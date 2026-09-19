#pragma once

#include "llama-ext.h"

#include <cstdint>

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
            llama_governor_engine requested, uint32_t prompt_tokens, float now_c) const;
    llama_governor_decode_selection select_decode(int64_t now_ms);

    llama_governor_engine prefill_engine() const;
    uint32_t prefill_rule() const;
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
    float t_idle_reference_c_ = 0.0f;
    bool have_t_idle_reference_ = false;
    llama_governor_engine last_decode_engine_ = llama_governor_engine::CPU;
};
