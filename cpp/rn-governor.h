#pragma once

#include "llama-ext.h"
#include "llama.h"

#include <cstdint>
#include <string>

namespace rnllama {

class rn_governor {
public:
    rn_governor(llama_model * prefill_model, llama_model * decode_model,
                llama_context_params prefill_params, llama_context_params decode_params,
                const llama_governor_params & params);
    ~rn_governor();

    llama_context * active_ctx() const;
    llama_context * prefill_ctx() const;
    int32_t decode(llama_batch batch);
    void clear_cache(bool clear_data);
    void reset_prefill_stats();
    bool set_thermo_profile(const llama_governor_thermo_profile & profile);
    llama_governor_stats stats() const;

    bool failed() const { return failed_; }
    bool profile_valid() const { return profile_valid_; }
    llama_governor_fit gpu_fit() const { return gpu_fit_; }
    const std::string & failure_reason() const { return failure_reason_; }

private:
    llama_governor * governor_ = nullptr;
    llama_governor_fit gpu_fit_ = llama_governor_fit::Unknown;
    bool failed_ = false;
    bool profile_valid_ = false;
    std::string failure_reason_;
};

} // namespace rnllama
