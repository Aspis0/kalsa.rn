#pragma once

#include "llama-ext.h"
#include "llama-governor-pacing-duty.h"

#include <atomic>
#include <cstdint>
#include <limits>

/** Copyable handle over an atomic prefill override: llama_governor_policy
 *  stays copy-assignable (the router tests reassign whole policies) while
 *  the binding's thread-pool push and a decode-thread read never race. */
struct atomic_prefill_mode {
    std::atomic<llama_governor_prefill_mode> value{llama_governor_prefill_mode::Auto};
    atomic_prefill_mode() = default;
    atomic_prefill_mode(llama_governor_prefill_mode mode) : value(mode) {}
    atomic_prefill_mode(const atomic_prefill_mode & other) : value(other.value.load()) {}
    atomic_prefill_mode & operator=(const atomic_prefill_mode & other) {
        value.store(other.value.load());
        return *this;
    }
};

bool llama_governor_expert_substitution_would_displace(
        float lambda, bool resident, float resident_score,
        float flash_winner_score, float score_range);

// Decode-hop hysteresis (llama-governor-policy-hop.cpp): another leg must
// beat the current one's effective headroom by this much before a window
// boundary moves the work, so sample noise cannot buy a flip-flopping KV
// commit. Starting value, to be tuned on the S23.
constexpr float k_hop_hysteresis_c = 4.0f;

// The C a leg is expected to give up while it runs (defined
// llama-governor-policy-hop.cpp, where the hop docks each idle candidate by it).
// The prefill placement docks by it under the same rule: a leg that is already
// running has paid the step and reads loaded, an idle one does not.
float load_step_of(const llama_governor_params & params, llama_governor_engine engine);

struct llama_governor_prefill_admission {
    llama_governor_decision decision = llama_governor_decision::Wait;
    llama_governor_engine engine = llama_governor_engine::CPU;
    uint32_t tokens = 0;
    uint32_t rule = 0;
};

// The three values the thermal log line stamps, read as one snapshot.
// Read without a lock, exactly like the three separate reads it replaced:
// the contract (llama-governor.h) is decode-thread-only - set_thermo_profile()
// is a decode-thread method and must not overlap decode() - but the binding
// is known to call it from its JSI ThreadPool (guarded only by
// throwIfContextBusy), so the writer race is real; pre-existing, not
// fixed here.
struct llama_governor_thermal_snapshot {
    llama_governor_thermal_state state;
    int32_t platform_status;
    bool from_platform;
};

struct llama_governor_decode_selection {
    llama_governor_engine engine = llama_governor_engine::CPU;
    bool requires_reload = false;
    bool wait = false;
    uint32_t rule = 0;
    // While the hop owns decode: static string naming what decided the
    // current window's leg ("headroom" / "gpu_burst" / "alternation");
    // null on every non-hop path, so a safety exit that moves the work
    // labels itself with no hop rule.
    const char * hop_rule = nullptr;
};

class llama_governor_policy {
public:
    explicit llama_governor_policy(const llama_governor_params & params);

    bool update_thermal(const llama_governor_thermo_profile & profile, int64_t now_ms);
    // Per-leg decode-hop headroom in C (NaN = unknown leg, no usable zone on
    // this phone); read by the hop rule in select_decode. The gpu leg is the
    // one-model governor's third leg and stays NaN for the two-model form.
    void set_decode_headroom(float cpu_headroom_c, float npu_headroom_c,
                             float gpu_headroom_c = std::numeric_limits<float>::quiet_NaN());
    // The same setter with an injected monotonic sample time in us: the
    // clock behind decode_headroom_tau_s's EMA. The public form calls it
    // with ggml_time_us(); host tests drive it directly to make the
    // smoothing deterministic.
    void set_decode_headroom(float cpu_headroom_c, float npu_headroom_c,
                             float gpu_headroom_c, int64_t now_us);
    // Which decode legs exist and may hop (rule v2): the one-model governor
    // pushes its leg table here (floors may drop a leg); the default
    // NPU+CPU is the two-model form's set, so v1 decisions are unchanged.
    void set_decode_legs(bool npu, bool gpu, bool cpu);
    // The prefill placement's own thermal input, in C to each leg's first
    // passive trip (llama_governor_set_prefill_headroom): the binding is
    // expected to push one right before a prefill batch. Per-turn state, but
    // not the clock reset_decode_hop() owns - that runs at prefill entry,
    // AFTER the sample - so it is dropped by clear_prefill_headroom() where
    // the turn's route re-arms instead. Raw, not smoothed: prefill decides
    // before any turn history exists. NaN = unknown leg, which leaves prefill
    // on the NPU.
    void set_prefill_headroom(float npu_headroom_c, float gpu_headroom_c);
    void clear_prefill_headroom();
    // The raw prefill sample as it stands right now: the turn's own until the route
    // re-arms (clear_prefill_headroom). Read it here rather than the members -
    // npu_hot_gpu_handoff is the only other reader, and a caller that wants the
    // turn's report wants this value, not the one a later push may replace.
    float prefill_npu_headroom_c() const { return prefill_npu_headroom_c_; }
    float prefill_gpu_headroom_c() const { return prefill_gpu_headroom_c_; }
    // The one-model governor's lane proof: its NPU leg exists and its
    // admission was set, which stands in for the LaunchConfig's fit /
    // readable-streams facts. npu_lane_enabled (the binding's platform
    // switch) stays the master gate either way.
    void set_npu_lane_capable(bool capable) { npu_lane_capable_ = capable; }
    // Arms a fresh hop leg decision for the next decode: the governor calls
    // this wherever it zeroes its decode-hop clock, and the headroom samples
    // go with the clock, so window 0 decides on this turn's reading (the
    // binding samples before the first decode of every turn), never on the
    // previous turn's.
    void reset_decode_hop();
    llama_governor_prefill_admission admit_prefill(
            llama_governor_engine requested, uint32_t prompt_tokens, float now_c,
            uint32_t n_batch = UINT32_MAX, bool thermal_handoff = false) const;
    // tokens_since_prefill is the caller's decode-hop clock: generated tokens
    // since the last prefill, which the governor restarts at every prefill
    // entry and at every reset_prefill_stats (the binding's per-completion
    // call, so a fully cached turn still opens window 0). It paces the hop
    // windows that the headroom and alternation rules decide on (see
    // select_decode).
    llama_governor_decode_selection select_decode(
            int64_t now_ms, uint32_t tokens_since_prefill = 0);

    // The optional outputs stamp the causal route facts: mode_used is the
    // single mode load of this call (reported even when safety preempts),
    // override_decided is true only when an override branch decided, and
    // thermal_handoff is true only when the NPU-hot GPU handoff decided - the
    // one GPU pick no other output distinguishes from a forced or fallback GPU.
    llama_governor_engine prefill_engine(
            llama_governor_prefill_mode * mode_used = nullptr,
            bool * override_decided = nullptr,
            bool * thermal_handoff = nullptr) const;
    // thermal_handoff is the same flag prefill_engine reports for the
    // request this admission is admitting, passed down so the rule it stamps is
    // the rule that chose the engine: the routing inputs can move under the
    // call (the mode is atomic and pushed cross-thread), and re-reading them
    // here would name a turn that did not happen.
    uint32_t prefill_rule() const;
    // The number for a placement ALREADY decided: prefill_rule for a caller
    // that holds the decision (the runtime latch, admit_prefill), so no caller
    // ever has to re-evaluate the routing inputs to label one.
    uint32_t prefill_rule_for(bool thermal_handoff, llama_governor_engine engine) const;
    // Bench route dev hook: mode is validated against
    // llama_governor_prefill_mode (0..2); false on anything else.
    bool set_prefill_override(int mode);
    llama_governor_thermal_state thermal_state() const;
    llama_governor_thermal_snapshot thermal_snapshot() const;
    // the constructor-time inputs (floor / heat knobs for the hop rule)
    const llama_governor_params & params() const { return params_; }
    // True while a platform-raised COOLMODE stands that battery heat does
    // not justify (sticky for the state's whole life, see update_thermal).
    bool state_from_platform() const;
    llama_governor_fit npu_fit() const;
    float current_temperature_c() const;
    uint32_t prefill_token_cap() const;
    uint64_t cpu_to_gpu_engagements() const;
    bool cache_budget_warning() const;
    float decode_coolmode_duty() const { return params_.decode_coolmode_duty; }
    // The platform thermal headroom forecast the duty curve reads (NaN = the
    // platform did not answer), normalized by update_thermal.
    float platform_thermal_headroom() const { return profile_.platform_thermal_headroom; }
    // The two battery-temperature lines the duty curve's second ramp reads
    // (llama-governor-pacing-duty.cpp), taken from the live profile: the cool
    // entry the classifier uses and the pause line above it.
    llama_governor_pacing_battery_lines decode_battery_lines() const;
    bool hot_plugged() const;

private:
    struct thresholds {
        float warm_enter;
        float warm_exit;
        float cool_enter;
        float cool_exit;
        float critical_enter;
        float critical_exit;
    };

    thresholds get_thresholds() const;
    llama_governor_thermal_state classify(float temperature) const;
    bool profile_is_valid(const llama_governor_thermo_profile & profile) const;
    bool dwell_elapsed(int64_t now_ms) const;
    bool can_leave(int64_t now_ms, float temperature, float exit_temperature) const;
    // The two predicates the NPU routing rules share (prefill_engine's lane
    // and the decode hop): the lane answers only when the binding opened it
    // and HTP proved it can read every weight stream the model needs, and
    // accelerators run only in the non-safety states.
    bool npu_lane_live() const;
    bool not_a_safety_state() const;
    // The GPU is loaded (params_.gpu_leg), fits, and is a measured prefill
    // backend on a generation the owner enabled for GPU prefill: the whole
    // "a GPU leg is really there to prefill on" fact. It gates both places
    // prefill_engine can reach a GPU leg on its own - the fallback and the
    // NPU-hot handoff - so the leg flag is read in one place.
    // prefill_engine's bench overrides read the leg directly instead -
    // bench A/Bs an unmeasured GPU.
    bool gpu_prefill_capable() const;
    // FIX 2 (owner 2026-10-16): the NPU lane is live but too hot to prefill on,
    // and the GPU - read at the temperature it will run at, by the hop's own
    // dock rule - is cooler than the NPU by more than the hop hysteresis and
    // clears the guard + hysteresis outright. FAST and WARM only: a COOLMODE
    // phone must not move prefill to the GPU (owner 2026-10-03). Unknown or
    // infinite headroom on either leg leaves prefill on the NPU.
    bool npu_hot_gpu_handoff() const;

    // Rule 4's leg choice (llama-governor-policy-hop.cpp): called only at
    // window boundaries while the hop is active; decides decode_hop_leg_,
    // returns the hop-rule label ("headroom"/"gpu_burst"/"alternation")
    // and logs one line per leg change.
    const char * hop_decide(uint32_t window, uint32_t tokens_since_prefill);
    void shift_smoothed_for_hop(llama_governor_engine prev_leg, llama_governor_engine new_leg);

    llama_governor_params params_;
    llama_governor_thermo_profile profile_;
    llama_governor_thermal_state state_ = llama_governor_thermal_state::Unknown;
    int64_t state_since_ms_ = 0;
    int64_t last_gpu_engagement_ms_ = -1;
    uint64_t cpu_to_gpu_engagements_ = 0;
    bool profile_valid_ = false;
    bool have_profile_ = false;
    bool state_from_platform_ = false;
    bool hot_plugged_ = false;
    bool cache_budget_warning_ = false;
    // /bench route dev hook; consulted only after the safety gates in
    // prefill_engine() - it requests, safety and admission still decide.
    // prefill_engine()'s answer (this override included, and LOWBAT's CPU
    // verdict) is read every admission but changes the route only at the
    // prefill latch, so a profile update mid-phase reroutes only at the
    // next latch; per-batch safety (abort/Wait) does not go through here. Atomic: the binding
    // pushes it from a thread-pool worker while a decode thread reads it -
    // a plain field would be a data race.
    atomic_prefill_mode prefill_override_;
    float t_idle_reference_c_ = 0.0f;
    bool have_t_idle_reference_ = false;
    llama_governor_engine last_decode_engine_ = llama_governor_engine::CPU;
    // Decode-hop leg state (the rule lives in select_decode /
    // llama-governor-policy-hop.cpp): per-leg headroom input (NaN =
    // unknown), the present-leg mask, the window the current leg was decided
    // for, and the leg itself. The window guard is what keeps decisions at
    // window boundaries - a caller polls select_decode per token.
    float decode_cpu_headroom_c_ = std::numeric_limits<float>::quiet_NaN();
    float decode_npu_headroom_c_ = std::numeric_limits<float>::quiet_NaN();
    float decode_gpu_headroom_c_ = std::numeric_limits<float>::quiet_NaN();
    // The same NPU/GPU margins, but the prefill placement's own sample: the
    // binding is expected to push one before a prefill batch, and nothing does
    // today, so both stay NaN and the handoff never fires. Deliberately NOT
    // dropped by reset_decode_hop(), which runs at prefill entry after the
    // sample; clear_prefill_headroom() drops them where the turn's route
    // re-arms, so an unsampled turn reads as unknown rather than stale.
    float prefill_npu_headroom_c_ = std::numeric_limits<float>::quiet_NaN();
    float prefill_gpu_headroom_c_ = std::numeric_limits<float>::quiet_NaN();
    // Per-leg EMA of those samples while decode_headroom_tau_s > 0,
    // indexed like llama_governor_engine (CPU, GPU, NPU); NaN = no finite
    // sample yet (or a NaN sample), so the next finite sample seeds rather
    // than blends. The timestamp pair is valid once a sample has been
    // smoothed - hop_decide reads exactly this array when tau > 0.
    float decode_headroom_smooth_[3] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN() };
    bool decode_headroom_have_last_us_ = false;
    int64_t decode_headroom_last_us_ = 0;
    // GPU_BURST mode: the hold set when a burst fires its one window and
    // cleared by every completed non-GPU window - that wait is the
    // no-back-to-back rule.
    bool decode_gpu_burst_blocked_ = false;
    // indexed like llama_governor_engine (CPU, GPU, NPU); the default is the
    // two-model form's set, so v1 decisions are byte-identical
    bool decode_leg_present_[3] = { true, false, true };
    // the one-model governor's lane proof (see set_npu_lane_capable)
    bool npu_lane_capable_ = false;
    // Window the current hop leg was decided for; the sentinel means no
    // window is decided yet (fresh policy or reset_decode_hop), so window 0
    // of a new decode always decides.
    uint32_t decode_hop_window_ = UINT32_MAX;
    llama_governor_engine decode_hop_leg_ = llama_governor_engine::CPU;
};
