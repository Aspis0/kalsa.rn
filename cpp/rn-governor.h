#pragma once

#include "llama-ext.h"
#include "llama.h"

#include <cstdint>
#include <string>

namespace rnllama {

// A decode rc is a real failure iff it is nonzero and either not the
// flow-control -2 or the engine governor has actually failed: the six
// flow-control -2s (thermal pause, caller chunking, reload-required, ...)
// never set the engine's failed state, while decode_impl sets it before
// returning any engine rc, including -2 for GGML_STATUS_ALLOC_FAILED.
inline bool governor_decode_failed(int32_t rc, bool engine_failed) {
    return rc != 0 && (rc != -2 || engine_failed);
}

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

    // The only sanctioned KV rewind under a governor (both contexts, both
    // watermarks); see llama_governor_trim_sequence. False only when even the
    // full clear of an untrimmable side failed.
    bool trim_sequence(llama_pos p) const;
    bool failed() const { return failed_; }
    // The engine governor's sticky state (not the shadow); out-of-line because
    // llama_governor is incomplete in this header.
    bool engine_failed() const;
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
