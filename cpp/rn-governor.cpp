#include "rn-governor.h"
#include "llama-governor.h"

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
    if (result != 0 && result != -2) {
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

void rn_governor::clear_cache(bool clear_data) {
    llama_governor_clear_cache(governor_, clear_data);
}

void rn_governor::reset_prefill_stats() {
    llama_governor_reset_prefill_stats(governor_);
}

bool rn_governor::set_thermo_profile(const llama_governor_thermo_profile & profile) {
    if (governor_ == nullptr || failed_) {
        return false;
    }
    profile_valid_ = llama_governor_set_thermo_profile(
        governor_, profile, lm_ggml_time_us() / 1000);
    return profile_valid_;
}

llama_governor_stats rn_governor::stats() const {
    llama_governor_stats result{};
    llama_governor_get_stats(governor_, &result);
    return result;
}

} // namespace rnllama
