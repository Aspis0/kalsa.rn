#include "llama-governor-policy.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr int64_t k_dwell_ms = 10 * 60 * 1000;
constexpr int64_t k_flip_window_ms = 60 * 60 * 1000;
constexpr float k_warn_c = 40.0f;
constexpr float k_limit_c = k_warn_c - 0.5f;
constexpr float k_kill_c = 43.0f;
constexpr float k_trend_c_per_min = 1.5f;
constexpr uint32_t k_table_tokens[] = { 128, 512, 1024, 2048 };
constexpr float k_cpu_delta_c[] = { 1.9f, 3.25f, 3.25f, 5.15f };

bool valid_schema(const llama_governor_params & params) {
    return params.schema_version == 3 && params.capability_schema_version == 2 &&
           std::isfinite(params.admission_margin_c) && std::fabs(params.admission_margin_c - 0.5f) < 0.0001f;
}

float temperature_c(const llama_governor_thermo_profile & profile) {
    return profile.batt_temp_tenths_c / 10.0f;
}

} // namespace

bool llama_governor_expert_substitution_would_displace(
        float lambda, bool resident, float resident_score,
        float flash_winner_score, float score_range) {
    return lambda > 0.0f && lambda <= 1.0f && resident && std::isfinite(resident_score) &&
           std::isfinite(flash_winner_score) && std::isfinite(score_range) && score_range >= 0.0f &&
           resident_score <= flash_winner_score &&
           flash_winner_score - resident_score <= lambda * score_range;
}

llama_governor_policy::llama_governor_policy(const llama_governor_params & params) : params_(params) {
    cache_budget_warning_ = params_.cache_budget_bytes != 0 && params_.expert_cycle_bytes != 0 &&
                           params_.cache_budget_bytes < params_.expert_cycle_bytes;
}

llama_governor_policy::thresholds llama_governor_policy::get_thresholds() const {
    if (!profile_.plugged) {
        return { 38.0f, 36.0f, 39.5f, 36.5f, 42.0f, 34.0f };
    }
    return {
        std::min(profile_.t_idle_c + 3.0f, 42.0f), profile_.t_idle_c + 1.0f,
        std::min(profile_.t_idle_c + 4.5f, 42.0f), profile_.t_idle_c + 1.5f,
        std::min(profile_.t_idle_c + 7.0f, 42.0f), profile_.t_idle_c + 1.0f,
    };
}

bool llama_governor_policy::profile_is_valid(const llama_governor_thermo_profile & profile) const {
    if (!profile.sensor_valid || !std::isfinite(profile.trend_c_per_min)) {
        return false;
    }
    if (profile.batt_level_pct < 0 || profile.batt_level_pct > 100) {
        return false;
    }
    if (!profile.plugged) {
        return true;
    }
    return profile.t_idle_valid && std::isfinite(profile.t_idle_c) && profile.t_idle_c + 1.0f < 42.0f;
}

llama_governor_thermal_state llama_governor_policy::classify(float temperature) const {
    const auto limits = get_thresholds();
    if (temperature >= k_kill_c || temperature >= limits.critical_enter) {
        return llama_governor_thermal_state::CRITICAL;
    }
    if (temperature >= limits.cool_enter) {
        return llama_governor_thermal_state::COOLMODE;
    }
    if (temperature >= limits.warm_enter) {
        return llama_governor_thermal_state::WARM;
    }
    return llama_governor_thermal_state::FAST;
}

bool llama_governor_policy::dwell_elapsed(int64_t now_ms) const {
    return now_ms >= state_since_ms_ && now_ms - state_since_ms_ >= k_dwell_ms;
}

bool llama_governor_policy::can_leave(int64_t now_ms, float temperature, float exit_temperature) const {
    return temperature <= exit_temperature && dwell_elapsed(now_ms);
}

bool llama_governor_policy::update_thermal(const llama_governor_thermo_profile & profile, int64_t now_ms) {
    profile_ = profile;
    if (!profile_is_valid(profile)) {
        profile_valid_ = false;
        have_profile_ = true;
        state_ = llama_governor_thermal_state::Invalid;
        state_since_ms_ = now_ms;
        return false;
    }
    if (profile.plugged) {
        if (!have_t_idle_reference_) {
            t_idle_reference_c_ = profile.t_idle_c;
            have_t_idle_reference_ = true;
        } else if (std::fabs(profile.t_idle_c - t_idle_reference_c_) > 1.5f) {
            profile_valid_ = false;
            have_profile_ = true;
            state_ = llama_governor_thermal_state::Invalid;
            state_since_ms_ = now_ms;
            return false;
        }
    }

    profile_valid_ = true;
    have_profile_ = true;
    hot_plugged_ = profile.plugged && profile.t_idle_c >= 37.5f;
    const float temp = temperature_c(profile);
    const auto limits = get_thresholds();
    const auto old_state = state_;
    if (temp >= k_kill_c || temp >= limits.critical_enter) {
        state_ = llama_governor_thermal_state::CRITICAL;
    } else if (state_ == llama_governor_thermal_state::CRITICAL) {
        if (can_leave(now_ms, temp, limits.critical_exit)) {
            state_ = llama_governor_thermal_state::FAST;
        }
    } else if (profile.batt_level_pct < 25) {
        state_ = llama_governor_thermal_state::LOWBAT;
    } else if (state_ == llama_governor_thermal_state::LOWBAT) {
        if (dwell_elapsed(now_ms)) {
            state_ = classify(temp);
        }
    } else if (state_ == llama_governor_thermal_state::Unknown || state_ == llama_governor_thermal_state::Invalid) {
        state_ = classify(temp);
    } else if (state_ == llama_governor_thermal_state::COOLMODE) {
        if (can_leave(now_ms, temp, limits.cool_exit)) {
            state_ = llama_governor_thermal_state::FAST;
        }
    } else if (state_ == llama_governor_thermal_state::WARM) {
        if (temp >= limits.cool_enter ||
            (temp >= limits.warm_enter + 0.5f && profile.trend_c_per_min >= k_trend_c_per_min)) {
            state_ = llama_governor_thermal_state::COOLMODE;
        } else if (can_leave(now_ms, temp, limits.warm_exit)) {
            state_ = llama_governor_thermal_state::FAST;
        }
    } else if (state_ == llama_governor_thermal_state::FAST) {
        state_ = temp >= limits.cool_enter ? llama_governor_thermal_state::COOLMODE
                                           : temp >= limits.warm_enter ? llama_governor_thermal_state::WARM
                                                                        : state_;
    }
    if (state_ != old_state) {
        state_since_ms_ = now_ms;
    }
    return true;
}

llama_governor_engine llama_governor_policy::prefill_engine() const {
    if (!valid_schema(params_) || !profile_valid_ || !have_profile_ ||
        state_ == llama_governor_thermal_state::CRITICAL || state_ == llama_governor_thermal_state::Invalid ||
        state_ == llama_governor_thermal_state::LOWBAT || profile_.batt_level_pct < 45) {
        return llama_governor_engine::CPU;
    }
    if (params_.bench_force_gpu_prefill && params_.gpu_fit == llama_governor_fit::Fit) {
        return llama_governor_engine::GPU;
    }
    // measured: ALIVE #38: 8 Elite GPU prefill, G ttft 1434/1470 ms vs
    // C 16476/14412 ms (>=9.8x); decode 25.3/24.2 t/s >= C's.
    if ((params_.generation == llama_governor_generation::V75 ||
         params_.generation == llama_governor_generation::V79) &&
        params_.gpu_fit == llama_governor_fit::Fit && params_.gpu_prefill_measured) {
        return llama_governor_engine::GPU;
    }
    if (params_.generation == llama_governor_generation::V73 && params_.model_kind == llama_governor_model_kind::MoE &&
        params_.cool_prefill_eligible && params_.gpu_fit == llama_governor_fit::Fit && !hot_plugged_) {
        return llama_governor_engine::GPU_COOLMODE;
    }
    return llama_governor_engine::CPU;
}

uint32_t llama_governor_policy::prefill_rule() const {
    if (params_.generation == llama_governor_generation::NoHTP) {
        return 1;
    }
    const auto engine = prefill_engine();
    return engine == llama_governor_engine::GPU ? 5 :
           engine == llama_governor_engine::GPU_COOLMODE ? 8 : 9;
}

llama_governor_prefill_admission llama_governor_policy::admit_prefill(
        llama_governor_engine requested, uint32_t prompt_tokens, float now_c) const {
    llama_governor_prefill_admission result{};
    result.engine = llama_governor_engine::CPU;
    if (!profile_valid_ || !have_profile_ || state_ == llama_governor_thermal_state::Invalid ||
        state_ == llama_governor_thermal_state::CRITICAL || prompt_tokens == 0) {
        result.decision = state_ == llama_governor_thermal_state::Invalid ||
                          state_ == llama_governor_thermal_state::CRITICAL
            ? llama_governor_decision::Abort : llama_governor_decision::Wait;
        return result;
    }

    size_t table = sizeof(k_table_tokens) / sizeof(k_table_tokens[0]);
    while (table > 0 && k_table_tokens[table - 1] > prompt_tokens) {
        --table;
    }
    while (table > 0 && now_c + k_cpu_delta_c[table - 1] > k_limit_c) {
        --table;
    }
    if (table == 0) {
        if (prompt_tokens < k_table_tokens[0] &&
            now_c + k_cpu_delta_c[0] <= k_limit_c) {
            result.tokens = prompt_tokens;
            result.rule = requested == llama_governor_engine::NPU ? 2 : requested == llama_governor_engine::CPU ? 9 : 3;
            result.decision = requested == llama_governor_engine::CPU
                ? llama_governor_decision::Admit : llama_governor_decision::CPUFallback;
        } else {
            result.decision = llama_governor_decision::Wait;
        }
        return result;
    }

    result.tokens = k_table_tokens[table - 1];
    result.rule = requested == llama_governor_engine::NPU ? 2 : requested == llama_governor_engine::CPU ? 9 : 3;
    if (result.tokens != prompt_tokens) {
        result.decision = llama_governor_decision::Chunk;
    } else if (requested == llama_governor_engine::CPU) {
        result.decision = llama_governor_decision::Admit;
    } else {
        result.decision = llama_governor_decision::CPUFallback;
    }
    return result;
}

llama_governor_decode_selection llama_governor_policy::select_decode(int64_t now_ms) {
    llama_governor_decode_selection result{};
    result.rule = 3;
    if (!have_profile_ || !profile_valid_ || state_ == llama_governor_thermal_state::Unknown ||
        state_ == llama_governor_thermal_state::Invalid || state_ == llama_governor_thermal_state::CRITICAL) {
        result.wait = true;
        return result;
    }
    const bool budget_ok = last_gpu_engagement_ms_ < 0 || now_ms < last_gpu_engagement_ms_ ||
                           now_ms - last_gpu_engagement_ms_ >= k_flip_window_ms;
    const bool eligible = valid_schema(params_) && have_profile_ && profile_valid_ && profile_.batt_level_pct >= 45 &&
        state_ == llama_governor_thermal_state::COOLMODE && !hot_plugged_ &&
        params_.generation == llama_governor_generation::V73 &&
        params_.gpu_fit == llama_governor_fit::Fit && params_.cool_delta_measured && params_.kexp_cool_scope &&
        params_.cool_pays != llama_governor_cool_pays::No && params_.reload_budget_available && budget_ok;
    if (!eligible) {
        last_decode_engine_ = llama_governor_engine::CPU;
        return result;
    }

    result.engine = llama_governor_engine::GPU_COOLMODE;
    result.requires_reload = last_decode_engine_ != llama_governor_engine::GPU_COOLMODE;
    if (result.requires_reload) {
        last_gpu_engagement_ms_ = now_ms;
        ++cpu_to_gpu_engagements_;
    }
    last_decode_engine_ = result.engine;
    return result;
}

llama_governor_thermal_state llama_governor_policy::thermal_state() const { return state_; }
llama_governor_fit llama_governor_policy::npu_fit() const { return params_.npu_fit; }
float llama_governor_policy::current_temperature_c() const { return temperature_c(profile_); }
uint32_t llama_governor_policy::prefill_token_cap() const {
    return params_.npu_lane_enabled ? 512 : 0;
}
uint64_t llama_governor_policy::cpu_to_gpu_engagements() const { return cpu_to_gpu_engagements_; }
bool llama_governor_policy::cache_budget_warning() const { return cache_budget_warning_; }
bool llama_governor_policy::hot_plugged() const { return hot_plugged_; }
