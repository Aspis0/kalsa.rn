#pragma once

#include "llama-ext.h"

#include <atomic>
#include <cstdint>

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

struct llama_governor_decode_selection {
    llama_governor_engine engine = llama_governor_engine::CPU;
    bool requires_reload = false;
    bool wait = false;
    uint32_t rule = 0;
};

class llama_governor_policy {
public:
    explicit llama_governor_policy(const llama_governor_params & params);

    bool update_thermal(const llama_governor_thermo_profile & profile, int64_t now_ms);
    llama_governor_prefill_admission admit_prefill(
            llama_governor_engine requested, uint32_t prompt_tokens, float now_c,
            uint32_t n_batch = UINT32_MAX) const;
    llama_governor_decode_selection select_decode(int64_t now_ms);

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

    llama_governor_params params_;
    llama_governor_thermo_profile profile_;
    llama_governor_thermal_state state_ = llama_governor_thermal_state::Unknown;
    int64_t state_since_ms_ = 0;
    int64_t last_gpu_engagement_ms_ = -1;
    uint64_t cpu_to_gpu_engagements_ = 0;
    bool profile_valid_ = false;
    bool have_profile_ = false;
    bool hot_plugged_ = false;
    bool cache_budget_warning_ = false;
    // /bench route dev hook; consulted only after the safety gates in
    // prefill_engine() - it requests, safety and admission still decide.
    // prefill_engine()'s engine answer (this override included, and LOWBAT's
    // CPU verdict) is consumed by the runtime once per prefill latch, so a
    // profile update mid-phase reroutes only at the next latch; per-batch
    // safety (abort/Wait) does not go through here. Atomic: the binding
    // pushes it from a thread-pool worker while a decode thread reads it -
    // a plain field would be a data race.
    atomic_prefill_mode prefill_override_;
    float t_idle_reference_c_ = 0.0f;
    bool have_t_idle_reference_ = false;
    llama_governor_engine last_decode_engine_ = llama_governor_engine::CPU;
};
