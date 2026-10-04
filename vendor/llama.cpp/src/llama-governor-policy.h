#pragma once

#include "llama-ext.h"

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
    // current window's leg ("headroom" / "alternation"); null on every
    // non-hop path, so a safety exit that moves the work labels itself
    // with no hop rule.
    const char * hop_rule = nullptr;
};

class llama_governor_policy {
public:
    explicit llama_governor_policy(const llama_governor_params & params);

    bool update_thermal(const llama_governor_thermo_profile & profile, int64_t now_ms);
    // Per-leg decode-hop headroom in C (NaN = unknown leg, no usable zone on
    // this phone); read by the hop rule in select_decode.
    void set_decode_headroom(float cpu_headroom_c, float npu_headroom_c);
    // Arms a fresh hop leg decision for the next decode: the governor calls
    // this wherever it zeroes its decode-hop clock, and the headroom samples
    // go with the clock, so window 0 decides on this turn's reading (the
    // binding samples before the first decode of every turn), never on the
    // previous turn's.
    void reset_decode_hop();
    llama_governor_prefill_admission admit_prefill(
            llama_governor_engine requested, uint32_t prompt_tokens, float now_c,
            uint32_t n_batch = UINT32_MAX) const;
    // tokens_since_prefill is the caller's decode-hop clock: generated tokens
    // since the last prefill, which the governor restarts at every prefill
    // entry and at every reset_prefill_stats (the binding's per-completion
    // call, so a fully cached turn still opens window 0). It paces the hop
    // windows that the headroom and alternation rules decide on (see
    // select_decode).
    llama_governor_decode_selection select_decode(
            int64_t now_ms, uint32_t tokens_since_prefill = 0);

    // The optional outputs stamp the causal route facts: mode_used is the
    // single mode load of this call (reported even when safety preempts) and
    // override_decided is true only when an override branch decided.
    llama_governor_engine prefill_engine(
            llama_governor_prefill_mode * mode_used = nullptr,
            bool * override_decided = nullptr) const;
    uint32_t prefill_rule() const;
    // Bench route dev hook: mode is validated against
    // llama_governor_prefill_mode (0..2); false on anything else.
    bool set_prefill_override(int mode);
    llama_governor_thermal_state thermal_state() const;
    llama_governor_thermal_snapshot thermal_snapshot() const;
    // True while a platform-raised COOLMODE stands that battery heat does
    // not justify (sticky for the state's whole life, see update_thermal).
    bool state_from_platform() const;
    llama_governor_fit npu_fit() const;
    float current_temperature_c() const;
    uint32_t prefill_token_cap() const;
    uint64_t cpu_to_gpu_engagements() const;
    bool cache_budget_warning() const;
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
    // Decode-hop leg state (the rule lives in select_decode): per-leg
    // headroom input (NaN = unknown), the window the current leg was decided
    // for, and the leg itself. The window guard is what keeps decisions at
    // window boundaries - a caller polls select_decode per token.
    float decode_cpu_headroom_c_ = std::numeric_limits<float>::quiet_NaN();
    float decode_npu_headroom_c_ = std::numeric_limits<float>::quiet_NaN();
    // Window the current hop leg was decided for; the sentinel means no
    // window is decided yet (fresh policy or reset_decode_hop), so window 0
    // of a new decode always decides.
    uint32_t decode_hop_window_ = UINT32_MAX;
    llama_governor_engine decode_hop_leg_ = llama_governor_engine::CPU;
};
