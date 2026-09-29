#include "llama-governor-policy.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr int64_t k_dwell_ms = 10 * 60 * 1000;
constexpr int64_t k_flip_window_ms = 60 * 60 * 1000;
// Owner decision 2026-09-25: pause at 43 C, kill at 44 C. k_limit_c is the
// admission projection ceiling - a row runs only while now_c + its delta
// stays at or under it - with one exception: the 2026-09-24 floor passes
// the smallest row below the pause line even when its projection exceeds
// the ceiling.
constexpr float k_warn_c = 43.0f;
constexpr float k_limit_c = k_warn_c - 0.5f;
constexpr float k_kill_c = 44.0f;
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

// Owner ladder (2026-09-29), hostile-audit fix: the platform status floors
// the thermal state - it never cools it and never brakes it. Every status
// 2..6 (MODERATE..SHUTDOWN) floors at COOLMODE, the state that prefers the
// cool path (select_decode hands out GPU_COOLMODE only there). Engine
// CRITICAL stays reachable from battery heat only: a transient Android
// status 5 must not reach fail("decode admission aborted") and kill the
// session; the platform emergency brake is the app's, which gates at
// Android CRITICAL between turns and recovers. LOWBAT outranks the floor:
// it guards a dying battery by forcing CPU, and a heat floor must not
// silently re-enable the accelerators.
llama_governor_thermal_state apply_platform_floor(
        llama_governor_thermal_state state, int32_t platform_status) {
    if (platform_status < 2) { // absent (-1), NONE (0), LIGHT (1): no escalation
        return state;
    }
    if (state == llama_governor_thermal_state::FAST ||
        state == llama_governor_thermal_state::WARM) {
        return llama_governor_thermal_state::COOLMODE;
    }
    return state;
}

// One normalization point for the wire value: a status outside -1..6 means
// the parser lied, so the field becomes absent (-1). Invalidating the
// profile here would feed the sticky abort path for a value the platform
// never sent.
int32_t normalize_platform_status(int32_t platform_status) {
    return platform_status < -1 || platform_status > 6 ? -1 : platform_status;
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
        return { 38.0f, 36.0f, 39.5f, 36.5f, k_kill_c, 34.0f };
    }
    // Review round 2026-09-25: critical_enter is the kill line in both
    // profiles. The plugged form min(t_idle + 7, kill) fired at 42 C for a
    // 35 C idle, preempting the 43 C pause; the pause at 43 handles hot
    // phones. Warm and cool routing caps stay pinned at 42 C.
    return {
        std::min(profile_.t_idle_c + 3.0f, 42.0f), profile_.t_idle_c + 1.0f,
        std::min(profile_.t_idle_c + 4.5f, 42.0f), profile_.t_idle_c + 1.5f,
        k_kill_c, profile_.t_idle_c + 1.0f,
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
    profile_.platform_thermal_status = normalize_platform_status(profile.platform_thermal_status);
    if (!profile_is_valid(profile)) {
        profile_valid_ = false;
        have_profile_ = true;
        state_ = llama_governor_thermal_state::Invalid;
        state_from_platform_ = false;
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
            state_from_platform_ = false;
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
    // The battery machine above owns every transition and its dwell; the
    // platform floor may only raise the result. state_since_ms_ below
    // stamps the net change: a raise that changes the state restamps it,
    // so the raised state leaves only through the same dwell gates as a
    // battery transition.
    const auto battery_state = state_;
    state_ = apply_platform_floor(state_, profile_.platform_thermal_status);
    // A platform-raised COOLMODE keeps the platform's attribution for its
    // whole life, including the dwell after the status drops: flipping to
    // battery there would buy GPU_COOLMODE decode plus a reload for heat
    // the platform reported. It clears only when the state leaves COOLMODE
    // or battery heat alone reaches the COOLMODE line.
    if (state_ != battery_state) {
        state_from_platform_ = true;
    } else if (state_ != llama_governor_thermal_state::COOLMODE ||
               temp >= limits.cool_enter) {
        state_from_platform_ = false;
    }
    if (state_ != old_state) {
        state_since_ms_ = now_ms;
    }
    return true;
}

llama_governor_engine llama_governor_policy::prefill_engine(
        llama_governor_prefill_mode * mode_used, bool * override_decided) const {
    // One atomic load per call: the binding may push a new mode from another
    // thread, and both outputs must describe the same decision.
    const auto mode = prefill_override_.value.load();
    if (mode_used != nullptr) {
        *mode_used = mode;
    }
    if (override_decided != nullptr) {
        *override_decided = false;
    }
    if (!valid_schema(params_) || !profile_valid_ || !have_profile_ ||
        state_ == llama_governor_thermal_state::CRITICAL || state_ == llama_governor_thermal_state::Invalid ||
        state_ == llama_governor_thermal_state::LOWBAT) {
        return llama_governor_engine::CPU; // safety preempts: override_decided stays false
    }
    // Bench route dev hook: the override requests the engine only after the
    // safety verdict above. Force-GPU keeps the gpu_fit==Fit gate (the same
    // gate bench_force_gpu_prefill uses below); a failed gate falls through
    // to the plan, which can only pick a Fit-guarded GPU path or CPU.
    if (mode == llama_governor_prefill_mode::CPU) {
        if (override_decided != nullptr) {
            *override_decided = true;
        }
        return llama_governor_engine::CPU;
    }
    if (mode == llama_governor_prefill_mode::GPU) {
        const bool fit = params_.gpu_fit == llama_governor_fit::Fit;
        if (override_decided != nullptr) {
            *override_decided = fit;
        }
        return fit ? llama_governor_engine::GPU : llama_governor_engine::CPU;
    }
    if (params_.bench_force_gpu_prefill && params_.gpu_fit == llama_governor_fit::Fit) {
        return llama_governor_engine::GPU;
    }
    // Owner rule (2026-09-28): NPU first for prefill, GPU when the NPU is
    // hot, CPU never while an accelerator qualifies - and the NPU is the
    // coolest prefill backend measured (S23 heat arms, lab
    // s23-heat-arms-engine commit 147b0351: 6.45 / 24.17 / 44.59 C*s per
    // 1k tokens NPU / GPU / CPU), so the lane stays eligible in FAST and in
    // COOLMODE; the arms are measured, the old "hop thresholds pending the
    // NPU heat arm" clause is closed. "GPU when the NPU is hot" needs an
    // NPU temperature input that does not exist yet: a later step, not
    // invented here. OFF unless npu_lane_enabled: npu_fit Fit, readable
    // HTP weights (MoE streams experts too). The bench/JS override above
    // still outranks it, and every safety state returned above still keeps CPU.
    if (params_.npu_lane_enabled &&
        params_.npu_fit == llama_governor_fit::Fit &&
        params_.htp_trunk_readable &&
        (params_.model_kind != llama_governor_model_kind::MoE || params_.htp_experts_readable) &&
        (state_ == llama_governor_thermal_state::FAST ||
         state_ == llama_governor_thermal_state::COOLMODE)) {
        return llama_governor_engine::NPU;
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
    // A platform-raised COOLMODE must not buy the cool prefill path;
    // only battery heat takes it.
    if (params_.generation == llama_governor_generation::V73 && params_.model_kind == llama_governor_model_kind::MoE &&
        params_.cool_prefill_eligible && params_.gpu_fit == llama_governor_fit::Fit && !hot_plugged_ &&
        !state_from_platform_) {
        return llama_governor_engine::GPU_COOLMODE;
    }
    // CPU stays the fallback only when no accelerator's own guard passed:
    // the safety states returned at the top of this function.
    return llama_governor_engine::CPU;
}

uint32_t llama_governor_policy::prefill_rule() const {
    if (params_.generation == llama_governor_generation::NoHTP) {
        return 1;
    }
    const auto engine = prefill_engine();
    return engine == llama_governor_engine::GPU ? 5 :
           engine == llama_governor_engine::GPU_COOLMODE ? 8 :
           engine == llama_governor_engine::NPU ? 2 : 9;
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
    // without this floor they empty the table at now_c > k_limit_c - 1.9 C
    // (40.6 C at the 2026-09-25 lines) and refuse every prefill, even a
    // 2-token prompt.
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
            result.engine = requested;
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
    // Every admitted piece runs on the requested engine; the runtime latch
    // reads this stamp to pick the turn's route.
    result.engine = requested;
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
    // GPU_COOLMODE is the hottest decode backend per token on the owner's
    // heat arms (NPU coolest, CPU between, GPU hottest), so a COOLMODE the
    // platform floor raised must not buy it: decode keeps today's engine.
    // Battery-derived COOLMODE eligibility is untouched.
    const bool eligible = valid_schema(params_) && have_profile_ && profile_valid_ &&
        state_ == llama_governor_thermal_state::COOLMODE && !state_from_platform_ && !hot_plugged_ &&
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
llama_governor_thermal_snapshot llama_governor_policy::thermal_snapshot() const {
    return { state_, profile_.platform_thermal_status, state_from_platform_ };
}
bool llama_governor_policy::state_from_platform() const { return state_from_platform_; }
llama_governor_fit llama_governor_policy::npu_fit() const { return params_.npu_fit; }
float llama_governor_policy::current_temperature_c() const { return temperature_c(profile_); }
uint32_t llama_governor_policy::prefill_token_cap() const {
    return params_.npu_lane_enabled ? 512 : 0;
}
uint64_t llama_governor_policy::cpu_to_gpu_engagements() const { return cpu_to_gpu_engagements_; }
bool llama_governor_policy::cache_budget_warning() const { return cache_budget_warning_; }
bool llama_governor_policy::hot_plugged() const { return hot_plugged_; }

bool llama_governor_policy::set_prefill_override(int mode) {
    if (mode < static_cast<int>(llama_governor_prefill_mode::Auto) ||
        mode > static_cast<int>(llama_governor_prefill_mode::GPU)) {
        return false;
    }
    prefill_override_.value.store(static_cast<llama_governor_prefill_mode>(mode));
    return true;
}
