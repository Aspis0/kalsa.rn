// The decode-hop leg rule (FABLE Amendment F, F3 rule v2) and its inputs:
// which legs may hop, each leg's thermal headroom (raw, or the EMA behind
// decode_headroom_tau_s), and the owner's selectable leg weighting.
// select_decode calls hop_decide() only at window boundaries while the hop
// is active; the safety and lane gates around it stay in select_decode.
//
// Owner constraints (F6d + owner vision 2026-10-03, "the governor is a
// three-legged surfer"): open every decode on the NPU; per window, move to
// the available leg whose own block headroom leads the current leg's by the
// hysteresis, NPU preferred on ties and returning under the same hysteresis.
// With any present leg's headroom unknown the rule falls back to v1's fixed
// alternation extended to the present legs. The heat-per-token weighting is
// a parameter defaulting to zero - pure headroom, v1 behaviour - so with the
// default leg mask (NPU+CPU, the two-model form) and default weights the
// decisions are byte-identical to v1.

#include "llama-governor-policy.h"

#include "llama-impl.h"

#include <cmath>

namespace {

const char * leg_label(llama_governor_engine engine) {
    switch (engine) {
        case llama_governor_engine::CPU: return "cpu";
        case llama_governor_engine::GPU: return "gpu";
        case llama_governor_engine::NPU: return "npu";
        case llama_governor_engine::GPU_COOLMODE: break;
    }
    return "?";
}

} // namespace

void llama_governor_policy::set_decode_headroom(float cpu_headroom_c, float npu_headroom_c,
                                                 float gpu_headroom_c) {
    set_decode_headroom(cpu_headroom_c, npu_headroom_c, gpu_headroom_c, ggml_time_us());
}

void llama_governor_policy::set_decode_headroom(float cpu_headroom_c, float npu_headroom_c,
                                                 float gpu_headroom_c, int64_t now_us) {
    decode_cpu_headroom_c_ = cpu_headroom_c;
    decode_npu_headroom_c_ = npu_headroom_c;
    decode_gpu_headroom_c_ = gpu_headroom_c;
    if (params_.decode_headroom_tau_s <= 0.0f) {
        return; // smoothing off: hop_decide reads the raw samples (today)
    }
    // Per-leg EMA, alpha = 1 - exp(-dt/tau) over the monotonic gap between
    // samples: the S23's min-zone headroom jumps tens of C between 32-token
    // windows (single-core spikes), which flipped the pure rule almost every
    // window. A finite sample whose leg average is not finite seeds it
    // instead of blending (first sample, or the one after a NaN - a NaN
    // parks the leg at NaN so the rule falls back until it re-seeds).
    const float raw[3] = { cpu_headroom_c, gpu_headroom_c, npu_headroom_c };
    double dt_s = 0.0;
    if (!decode_headroom_have_last_us_) {
        decode_headroom_last_us_ = now_us;
    } else if (now_us > decode_headroom_last_us_) {
        // Compare before subtracting: on a backwards (or equal) step keep
        // the last accepted timestamp and dt 0, so the next forward sample
        // measures real elapsed time from it instead of a stretched one.
        dt_s = static_cast<double>(static_cast<uint64_t>(now_us) -
                                   static_cast<uint64_t>(decode_headroom_last_us_)) / 1e6;
        decode_headroom_last_us_ = now_us;
    }
    const float alpha = static_cast<float>(1.0 - std::exp(-dt_s / params_.decode_headroom_tau_s));
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(raw[i])) {
            decode_headroom_smooth_[i] = std::numeric_limits<float>::quiet_NaN();
        } else if (!std::isfinite(decode_headroom_smooth_[i])) {
            decode_headroom_smooth_[i] = raw[i];
        } else {
            decode_headroom_smooth_[i] += alpha * (raw[i] - decode_headroom_smooth_[i]);
        }
    }
    decode_headroom_have_last_us_ = true;
}

void llama_governor_policy::set_decode_legs(bool npu, bool gpu, bool cpu) {
    decode_leg_present_[static_cast<int>(llama_governor_engine::NPU)] = npu;
    decode_leg_present_[static_cast<int>(llama_governor_engine::GPU)] = gpu;
    // H-05: the CPU leg is the fallback that always exists - never
    // excludable, whatever the caller (or a floor) says
    decode_leg_present_[static_cast<int>(llama_governor_engine::CPU)] = true;
    (void) cpu;
}

const char * llama_governor_policy::hop_decide(uint32_t window, uint32_t tokens_since_prefill) {
    // present legs in preference order: NPU first, so window 0 opens there
    // and exact ties prefer it
    llama_governor_engine present[3];
    int n_present = 0;
    if (decode_leg_present_[static_cast<int>(llama_governor_engine::NPU)]) {
        present[n_present++] = llama_governor_engine::NPU;
    }
    if (decode_leg_present_[static_cast<int>(llama_governor_engine::GPU)]) {
        present[n_present++] = llama_governor_engine::GPU;
    }
    if (decode_leg_present_[static_cast<int>(llama_governor_engine::CPU)]) {
        present[n_present++] = llama_governor_engine::CPU;
    }
    if (n_present == 0) {
        // unreachable while the CPU leg is never excludable; fail closed to
        // the CPU anyway so the decision is always a present, admitted leg
        decode_hop_leg_ = llama_governor_engine::CPU;
        return nullptr;
    }

    // What the rule reads: the EMA once smoothing is on, the raw sample
    // otherwise - tau 0 must stay byte-identical to the shipped path.
    const auto headroom_of = [this](llama_governor_engine engine) {
        const bool smoothed = params_.decode_headroom_tau_s > 0.0f;
        switch (engine) {
            case llama_governor_engine::CPU:
                return smoothed ? decode_headroom_smooth_[static_cast<int>(llama_governor_engine::CPU)]
                                : decode_cpu_headroom_c_;
            case llama_governor_engine::GPU:
                return smoothed ? decode_headroom_smooth_[static_cast<int>(llama_governor_engine::GPU)]
                                : decode_gpu_headroom_c_;
            case llama_governor_engine::NPU:
                return smoothed ? decode_headroom_smooth_[static_cast<int>(llama_governor_engine::NPU)]
                                : decode_npu_headroom_c_;
            case llama_governor_engine::GPU_COOLMODE: break;
        }
        return decode_cpu_headroom_c_;
    };
    // effective headroom: the heat weighting (battery C*s/token, a lab
    // constant per leg) subtracts weight*heat from a leg's raw headroom;
    // weight 0 is pure headroom, i.e. v1
    const auto effective_of = [this, &headroom_of](llama_governor_engine engine) {
        float heat = 0.0f;
        switch (engine) {
            case llama_governor_engine::CPU: heat = params_.decode_heat_per_token_cpu; break;
            case llama_governor_engine::GPU: heat = params_.decode_heat_per_token_gpu; break;
            case llama_governor_engine::NPU: heat = params_.decode_heat_per_token_npu; break;
            case llama_governor_engine::GPU_COOLMODE: break;
        }
        return headroom_of(engine) - params_.decode_heat_weight * heat;
    };

    const llama_governor_engine prev_leg = decode_hop_leg_;
    bool all_known = true;
    for (int i = 0; i < n_present; ++i) {
        if (!std::isfinite(headroom_of(present[i]))) {
            all_known = false;
        }
    }

    const char * hop_rule = "alternation";
    if (all_known) {
        hop_rule = "headroom";
        if (window == 0) {
            // open every decode on the NPU (present[0]; when the NPU is not
            // a candidate this falls to the next leg in preference order)
            decode_hop_leg_ = present[0];
        } else {
            // NPU_FIRST's loop (the shipped path, unchanged): best other
            // leg by effective headroom; the strict compare keeps the
            // earlier index on exact ties, which is the NPU preference
            int best = -1;
            for (int i = 0; i < n_present; ++i) {
                if (present[i] == decode_hop_leg_) { continue; }
                if (best < 0 || effective_of(present[i]) > effective_of(present[best])) {
                    best = i;
                }
            }
            if (best >= 0 &&
                effective_of(present[best]) - effective_of(decode_hop_leg_) >= k_hop_hysteresis_c) {
                decode_hop_leg_ = present[best];
            }
            if (params_.decode_leg_weighting == llama_governor_leg_weighting::GPU_BURST) {
                // GPU_BURST: a window the loop above keeps on the CPU goes
                // to the GPU instead when the GPU is a candidate with at
                // least the CPU's effective headroom - exactly one window:
                // the substitution makes GPU the incumbent, and from there
                // the CPU can only win by beating the GPU, which contradicts
                // gpu >= cpu, so no second substitution is reachable
                // back-to-back. The hold below states that wait; every
                // completed non-GPU window releases it.
                const bool wanted = decode_hop_leg_ == llama_governor_engine::CPU &&
                    decode_leg_present_[static_cast<int>(llama_governor_engine::GPU)] &&
                    effective_of(llama_governor_engine::GPU) >=
                        effective_of(llama_governor_engine::CPU);
                if (wanted && !decode_gpu_burst_blocked_) {
                    decode_hop_leg_ = llama_governor_engine::GPU;
                    hop_rule = "gpu_burst";
                    decode_gpu_burst_blocked_ = true;
                }
            }
        }
    } else {
        // v1's fixed alternation extended to the present legs: cycle the
        // list starting at its last element (the CPU), which for two legs is
        // exactly v1's odd/even window map
        decode_hop_leg_ = present[(window + n_present - 1) % n_present];
    }
    // Every completed non-GPU window - headroom, alternation or a burst
    // the hold suppressed - releases the burst hold, so the no-back-to-
    // back wait is exactly one such window, whatever decided it.
    if (decode_hop_leg_ != llama_governor_engine::GPU) {
        decode_gpu_burst_blocked_ = false;
    }

    // One line per leg change - when and why the work moved, naming the
    // three headrooms the rule used (the EMA values when tau > 0; nan = leg
    // absent or no zones), never a per-window tick.
    if (decode_hop_leg_ != prev_leg) {
        if (all_known) {
            LLAMA_LOG_INFO("governor: decode hop %s->%s at token %u (headroom cpu %.1f C,"
                           " npu %.1f C, gpu %.1f C, hysteresis %.0f C)\n",
                           leg_label(prev_leg), leg_label(decode_hop_leg_), tokens_since_prefill,
                           headroom_of(llama_governor_engine::CPU),
                           headroom_of(llama_governor_engine::NPU),
                           headroom_of(llama_governor_engine::GPU),
                           k_hop_hysteresis_c);
        } else {
            LLAMA_LOG_INFO("governor: decode hop %s->%s at token %u (alternation, zones unknown)\n",
                           leg_label(prev_leg), leg_label(decode_hop_leg_), tokens_since_prefill);
        }
    }
    return hop_rule;
}
