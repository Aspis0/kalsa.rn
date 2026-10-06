// The decode-hop leg rule (FABLE Amendment F, F3 rule v2) and its inputs:
// which legs may hop and each leg's thermal headroom. select_decode calls
// hop_decide() only at window boundaries while the hop is active; the safety
// and lane gates around it stay in select_decode.
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
    decode_cpu_headroom_c_ = cpu_headroom_c;
    decode_npu_headroom_c_ = npu_headroom_c;
    decode_gpu_headroom_c_ = gpu_headroom_c;
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

    const auto headroom_of = [this](llama_governor_engine engine) {
        switch (engine) {
            case llama_governor_engine::CPU: return decode_cpu_headroom_c_;
            case llama_governor_engine::GPU: return decode_gpu_headroom_c_;
            case llama_governor_engine::NPU: return decode_npu_headroom_c_;
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
            // best other leg by effective headroom; the strict compare keeps
            // the earlier index on exact ties, which is the NPU preference
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
        }
    } else {
        // v1's fixed alternation extended to the present legs: cycle the
        // list starting at its last element (the CPU), which for two legs is
        // exactly v1's odd/even window map
        decode_hop_leg_ = present[(window + n_present - 1) % n_present];
    }

    // One line per leg change - when and why the work moved, naming all
    // three headrooms (nan = leg absent or no zones), never a per-window tick.
    if (decode_hop_leg_ != prev_leg) {
        if (all_known) {
            LLAMA_LOG_INFO("governor: decode hop %s->%s at token %u (headroom cpu %.1f C,"
                           " npu %.1f C, gpu %.1f C, hysteresis %.0f C)\n",
                           leg_label(prev_leg), leg_label(decode_hop_leg_), tokens_since_prefill,
                           decode_cpu_headroom_c_, decode_npu_headroom_c_, decode_gpu_headroom_c_,
                           k_hop_hysteresis_c);
        } else {
            LLAMA_LOG_INFO("governor: decode hop %s->%s at token %u (alternation, zones unknown)\n",
                           leg_label(prev_leg), leg_label(decode_hop_leg_), tokens_since_prefill);
        }
    }
    return hop_rule;
}
