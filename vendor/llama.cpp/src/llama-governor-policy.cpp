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
    // A battery below 0 C is a sensor fault, not a reading; accepted as
    // valid it would classify the session as cool and open admission.
    if (profile.batt_temp_tenths_c < 0) {
        return false;
    }
    if (profile.batt_level_pct < 0 || profile.batt_level_pct > 100) {
        return false;
    }
    if (!profile.plugged) {
        return true;
    }
    // 0 C or colder is not a real baseline (dumpsys reads 0.0 for no data);
    // the removed app gate refused it, and this gate owns that rule now.
    return profile.t_idle_valid && std::isfinite(profile.t_idle_c) && profile.t_idle_c > 0.0f &&
           profile.t_idle_c + 1.0f < 42.0f;
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
    } else {
        // An unplug ends the plugged baseline; the next plug latches a fresh one.
        have_t_idle_reference_ = false;
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
            state_ = !profile.plugged && profile.batt_level_pct < 25
                ? llama_governor_thermal_state::LOWBAT
                : llama_governor_thermal_state::FAST;
        }
    } else if (!profile.plugged && profile.batt_level_pct < 25) {
        state_ = llama_governor_thermal_state::LOWBAT;
    } else if (state_ == llama_governor_thermal_state::LOWBAT) {
        if (profile.plugged || dwell_elapsed(now_ms)) {
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
        state_ == llama_governor_thermal_state::LOWBAT) {
        return llama_governor_engine::CPU;
    }
    if (params_.bench_force_gpu_prefill && params_.gpu_fit == llama_governor_fit::Fit) {
        return llama_governor_engine::GPU;
    }
    // measured: ALIVE #38: 8 Elite GPU prefill, G ttft 1434/1470 ms vs
    // C 16476/14412 ms (>=9.8x); decode 25.3/24.2 t/s >= C's.
    // V73 carries the owner's 2026-09-21 enablement decision, not a measurement.
    // The generation list is duplicated in the app; the form refactor should carry it once.
    if ((params_.generation == llama_governor_generation::V73 ||
         params_.generation == llama_governor_generation::V75 ||
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
        llama_governor_engine requested, uint32_t prompt_tokens, float now_c, uint32_t n_batch) const {
    llama_governor_prefill_admission result{};
    result.engine = llama_governor_engine::CPU;
    // Non-finite now_c is unknown heat: both the warn-line test and the delta
    // comparisons read false for NaN, which would admit the largest row.
    if (!profile_valid_ || !have_profile_ || !std::isfinite(now_c) ||
        state_ == llama_governor_thermal_state::Invalid ||
        state_ == llama_governor_thermal_state::CRITICAL || prompt_tokens == 0) {
        result.decision = state_ == llama_governor_thermal_state::Invalid ||
                          state_ == llama_governor_thermal_state::CRITICAL
            ? llama_governor_decision::Abort : llama_governor_decision::Wait;
        return result;
    }

    size_t table = sizeof(k_table_tokens) / sizeof(k_table_tokens[0]);
    // Rows are capped to n_batch as well as to the prompt: the input batch may
    // be larger than n_batch, and every executed piece must fit one
    // llama_decode (which asserts n_tokens_all <= cparams.n_batch).
    while (table > 0 && (k_table_tokens[table - 1] > prompt_tokens ||
                         k_table_tokens[table - 1] > n_batch)) {
        --table;
    }
    // Owner decision 2026-09-24: below the warn line the smallest chunk always
    // passes, so admission can only Wait at or above it. The per-row deltas are
    // unmeasured defaults (40051f8ae) applied to CPU and GPU prefill alike;
    // without this floor they empty the table at now_c > 37.6 C and refuse
    // every prefill, even a 2-token prompt.
    const bool below_warn = now_c < k_warn_c;
    while (table > 0 && now_c + k_cpu_delta_c[table - 1] > k_limit_c) {
        if (table == 1 && below_warn) {
            break;
        }
        --table;
    }
    if (table == 0) {
        if (prompt_tokens < k_table_tokens[0] && prompt_tokens <= n_batch && below_warn) {
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
    // Row+1 cannot be partitioned: the leftover token would be a 1-token
    // batch, and a 1-token batch is a decode, not a prefill (decode_impl:
    // is_prefill = n_tokens > 1). Promote it whole unless that would exceed
    // n_batch. On the delta path the projection stays under k_limit_c; on the
    // owner's floor path (now_c < k_warn_c, decision 2026-09-24) row 128
    // passes by decision and 129 is that floor plus the one token that cannot
    // run alone - its projection may sit above the limit, which the floor
    // accepts.
    if (prompt_tokens - result.tokens == 1) {
        if (prompt_tokens <= n_batch) {
            result.tokens = prompt_tokens;
        } else if (table > 1) {
            // Only when n_batch is itself a table row can row+1 exceed the
            // cap; take the next row down so the remainder re-plans inside it.
            --table;
            result.tokens = k_table_tokens[table - 1];
        } else {
            // Cap is the smallest row (128) and the prompt is 129: 127 + 2
            // keeps both pieces runnable inside the cap.
            result.tokens = prompt_tokens - 2;
        }
    }
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
    const bool eligible = valid_schema(params_) && have_profile_ && profile_valid_ &&
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
