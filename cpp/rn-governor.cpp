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
    profile_valid_ = llama_governor_set_thermo_profile(
        governor_, profile, ggml_time_us() / 1000);
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
