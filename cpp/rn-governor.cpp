#include "rn-governor.h"
#include "llama-governor.h"
#include "rn-llama.h"

#include "ggml.h"

#include <stdexcept>

namespace rnllama {

rn_governor::rn_governor(llama_model * prefill_model, llama_model * decode_model,
                         llama_context_params prefill_params, llama_context_params decode_params,
                         const llama_governor_params & params)
    : gpu_fit_(params.gpu_fit) {
    // This is the same two-model call order as governor-bench. The v0.3 API
    // overload additionally enables the parsed policy inputs and thermo gate.
    std::string failure_reason;
    governor_ = llama_governor_init_with_params_internal(
        prefill_model, decode_model, prefill_params, decode_params, params, &failure_reason);
    if (governor_ == nullptr) {
        failure_reason_ = failure_reason.empty()
            ? "llama_governor_init_with_params returned null"
            : failure_reason;
        throw std::runtime_error(failure_reason_);
    }
    log_platform_thermal_availability();
    init_hop_reader(params);
}

rn_governor::rn_governor(llama_model * model, const llama_governor_leg * legs, uint32_t n_legs,
                         const llama_governor_params & params)
    : gpu_fit_(params.gpu_fit) {
    // One shared llama_model, one context per leg (F1). A null engine
    // governor logged its reason on the engine side (no out-param in the
    // one-model form); the caller falls back to the two-model load.
    governor_ = llama_governor_init_one_model_with_params(model, legs, n_legs, params);
    if (governor_ == nullptr) {
        failure_reason_ = "llama_governor_init_one_model_with_params returned null";
        throw std::runtime_error(failure_reason_);
    }
    log_platform_thermal_availability();
    // A present leg the engine did not build (its device refused, e.g. the
    // OpenCL import) is a runtime demotion: logged once here, the hop simply
    // never uses that leg, and the load stays one-copy on the legs that did
    // build. The CPU leg cannot be absent - the engine refuses without it.
    for (uint32_t i = 0; i < n_legs; ++i) {
        if (legs[i].engine == llama_governor_engine::CPU) {
            continue;
        }
        if (llama_governor_leg_ctx(governor_, legs[i].engine) == nullptr) {
            LOG_ERROR(
                "KALSA_GOVERNOR_FALLBACK {stage:\"onecopy_leg_absent\", models_loaded:1, engine:\"%s\"}",
                legs[i].engine == llama_governor_engine::NPU ? "NPU" : "GPU");
        }
    }
    init_hop_reader(params);
}

void rn_governor::init_hop_reader(const llama_governor_params & params) {
    // The leg reader exists only when the hop is configured, so a hop-less
    // governor does no sysfs work; the one-time counts say which legs the
    // headroom rule can decide (both zero -> the engine's alternation
    // fallback, e.g. Jelly/MTK denies the zones).
    if (params.decode_hop_tokens > 0) {
        decode_hop_tokens_ = params.decode_hop_tokens;
        decode_legs_ = std::make_unique<rn_thermal_legs>("/sys/class/thermal");
        LOG_INFO("governor: decode hop thermal legs cpu=%d npu=%d gpu=%d zones",
                 decode_legs_->cpu_zone_count(), decode_legs_->npu_zone_count(),
                 decode_legs_->gpu_zone_count());
    }
}

void rn_governor::log_platform_thermal_availability() const {
#if defined(__ANDROID__)
    if (const char * reason = platform_thermal_.unavailable_reason()) {
        LOG_INFO("KALSA_GOVERNOR_PLATFORM {source:\"unavailable\", reason:\"%s\"}", reason);
    }
#endif
}

rn_governor::~rn_governor() {
    llama_governor_free(governor_);
    governor_ = nullptr;
}

llama_context * rn_governor::active_ctx() const {
    if (governor_ == nullptr) {
        return nullptr;
    }
    auto * active = llama_governor_get_context(governor_);
    return active != nullptr ? active : llama_governor_get_prefill_context(governor_);
}

llama_context * rn_governor::prefill_ctx() const {
    return governor_ == nullptr ? nullptr : llama_governor_get_prefill_context(governor_);
}

int32_t rn_governor::decode(llama_batch batch) {
    if (failed_ || governor_ == nullptr) {
        failed_ = true;
        failure_reason_ = "governor is failed";
        return -1;
    }

    // Ahead of this step's admission, because the engine's platform floor
    // (apply_platform_floor) is read at admission time and decode is the one
    // place every step of every governor path passes through.
    refresh_platform_thermal();

    const int32_t result = llama_governor_decode(governor_, batch);
    if (governor_decode_failed(result, governor_->is_failed())) {
        failed_ = true;
        const char * reason = governor_->failure_reason();
        if (reason != nullptr) {
            failure_reason_ = reason;
        } else {
            failure_reason_ = "llama_governor_decode failed with rc=" + std::to_string(result);
        }
    }
    return result;
}

bool rn_governor::trim_sequence(llama_pos p) const {
    return governor_ != nullptr && llama_governor_trim_sequence(governor_, p);
}

bool rn_governor::engine_failed() const {
    return governor_ != nullptr && governor_->is_failed();
}

void rn_governor::clear_cache(bool clear_data) {
    // The engine restarts its hop clock here (llama-governor-runtime.cpp
    // clear_cache); keep the mirror in step.
    decode_tokens_since_prefill_ = 0;
    llama_governor_clear_cache(governor_, clear_data);
}

void rn_governor::reset_prefill_stats() {
    // The engine restarts its hop clock here too (the completion's
    // beginCompletion is one caller); keep the mirror in step.
    decode_tokens_since_prefill_ = 0;
    llama_governor_reset_prefill_stats(governor_);
}

void rn_governor::before_decode_batch(int n_tokens) {
    if (decode_legs_ == nullptr) {
        return;
    }
    // The engine's decode_tokens_since_prefill_ counts exactly these
    // standard-path batches (multi-token batches are prefills that restart
    // it; 1-token batches increment it after the run), and select_decode
    // decides window k on the pre-increment count k*N - so feeding when the
    // mirror hits a multiple of N puts a fresh sample under every window
    // decision, first window included.
    if (n_tokens > 1) {
        decode_tokens_since_prefill_ = 0;
        return;
    }
    if (decode_tokens_since_prefill_ % decode_hop_tokens_ == 0) {
        feed_decode_headroom();
    }
    ++decode_tokens_since_prefill_;
}

void rn_governor::feed_decode_headroom() {
    if (decode_legs_ == nullptr) {
        return;
    }
    const rn_thermal_legs::headroom legs = decode_legs_->sample();
    // The GPU value is NaN without gpu zones (no GPU leg): the engine's hop
    // treats an unknown leg as unavailable, which is exactly the two-model
    // and Jelly behaviour.
    llama_governor_set_decode_headroom(governor_, legs.cpu_headroom_c, legs.npu_headroom_c,
                                       legs.gpu_headroom_c);
}

bool rn_governor::set_thermo_profile(const llama_governor_thermo_profile & profile) {
    if (governor_ == nullptr || failed_) {
        return false;
    }
    std::lock_guard<std::mutex> lock(profile_mutex_);
    // -1 is "no app vote" (key absent, malformed, or the app's read failed).
    // That is a gap, not a cool-down, so the remembered status survives it
    // while a native read returned a status in the last 10 s. Native reads
    // run only inside a completion, so a profile sent after a longer pause
    // drops the carry and the native reader re-escalates (5 s debounce);
    // the engine's COOLMODE dwell keeps an entered COOLMODE meanwhile.
    llama_governor_thermo_profile merged = profile;
    const bool native_recent = platform_native_valid_us_ > 0 &&
        ggml_time_us() - platform_native_valid_us_ <= k_platform_thermal_carry_us;
    if (merged.platform_thermal_status < 0 && native_recent) {
        merged.platform_thermal_status = thermo_profile_.platform_thermal_status;
    }
    const bool status_changed =
        merged.platform_thermal_status != thermo_profile_.platform_thermal_status;
    profile_valid_ = llama_governor_set_thermo_profile(governor_, merged, ggml_time_us() / 1000);
    if (profile_valid_) {
        thermo_profile_ = merged;
        // An app profile cancels a pending native escalation only when it
        // changes the status.
        if (status_changed) {
            platform_thermal_send_state_ = {};
        }
        platform_refusal_logged_ = false;
    }
    return profile_valid_;
}

void rn_governor::refresh_platform_thermal() {
    {
        std::lock_guard<std::mutex> lock(profile_mutex_);
        if (!profile_valid_ || !rn_platform_thermal_should_read(ggml_time_us(),
                                                                platform_thermal_read_us_)) {
            return;
        }
    }
    // The HAL call can block on binder; profile updates remain available while
    // it runs, and the send decision uses whichever profile was accepted latest.
    const int32_t status = platform_thermal_.status();
    std::lock_guard<std::mutex> lock(profile_mutex_);
    if (!profile_valid_) {
        return;
    }
    if (status != k_platform_thermal_absent) {
        platform_native_valid_us_ = ggml_time_us();
    }
    const int32_t previous = thermo_profile_.platform_thermal_status;
    const auto decision = rn_platform_thermal_should_send(
        ggml_time_us(), status, previous, platform_thermal_send_state_);
    platform_thermal_send_state_ = decision.next_state;
    if (decision.pending_started) {
        LOG_INFO("KALSA_GOVERNOR_PLATFORM {status:%d, prev:%d, source:\"native\", sent:0}",
                 status, previous);
    }
    if (!decision.send) {
        return;
    }
    llama_governor_thermo_profile updated = thermo_profile_;
    updated.platform_thermal_status = status;
    if (!llama_governor_set_thermo_profile(governor_, updated, ggml_time_us() / 1000)) {
        profile_valid_ = false;
        platform_thermal_send_state_ = {};
        if (!platform_refusal_logged_) {
            LOG_WARNING("KALSA_GOVERNOR_PLATFORM {source:\"native\", sent:0, reason:\"engine_refused\"}");
            platform_refusal_logged_ = true;
        }
        return;
    }
    thermo_profile_ = updated;
    LOG_INFO("KALSA_GOVERNOR_PLATFORM {status:%d, prev:%d, source:\"native\", sent:1}",
             status, previous);
}

bool rn_governor::profile_valid() const {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    return profile_valid_;
}

bool rn_governor::set_prefill_override(int mode) {
    if (governor_ == nullptr || failed_) {
        return false;
    }
    return llama_governor_set_prefill_override(governor_, mode);
}

llama_governor_stats rn_governor::stats() const {
    llama_governor_stats result{};
    llama_governor_get_stats(governor_, &result);
    return result;
}

} // namespace rnllama
