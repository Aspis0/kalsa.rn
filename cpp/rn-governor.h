#pragma once

#include "llama-ext.h"
#include "llama.h"
#include "rn-platform-thermal.h"
#include "rn-thermal-legs.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
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
    // The one-model form: one shared llama_model, one context per leg spec
    // (llama_governor_init_one_model_with_params). Absent legs (no devices,
    // or a device the model does not list) are the engine's call; the CPU
    // leg is required by the engine.
    rn_governor(llama_model * model, const llama_governor_leg * legs, uint32_t n_legs,
                const llama_governor_params & params);
    ~rn_governor();

    llama_context * active_ctx() const;
    llama_context * prefill_ctx() const;
    int32_t decode(llama_batch batch);
    void clear_cache(bool clear_data);
    void reset_prefill_stats();
    // The completion's decode path calls this before every standard decode
    // batch, on the decode thread: it keeps the binding mirror of the
    // engine's decode_tokens_since_prefill_ in step and feeds the fresh leg
    // sample each hop window opens on (see before_decode_batch in the .cpp).
    // No-op when the hop is off (no reader, no sysfs work).
    void before_decode_batch(int n_tokens);
    bool set_thermo_profile(const llama_governor_thermo_profile & profile);
    // Bench route dev hook: 0=auto, 1=cpu, 2=gpu (engine validates).
    bool set_prefill_override(int mode);
    llama_governor_stats stats() const;

    // The only sanctioned KV rewind under a governor (both contexts, both
    // watermarks); see llama_governor_trim_sequence. False only when even the
    // full clear of an untrimmable side failed.
    bool trim_sequence(llama_pos p) const;
    bool failed() const { return failed_; }
    // The engine governor's sticky state (not the shadow); out-of-line because
    // llama_governor is incomplete in this header.
    bool engine_failed() const;
    bool profile_valid() const;
    llama_governor_fit gpu_fit() const { return gpu_fit_; }
    const std::string & failure_reason() const { return failure_reason_; }

private:
    // The decode-hop thermal-leg reader both constructors share: built only
    // when decode_hop_tokens > 0, so a hop-less governor does no sysfs work.
    void init_hop_reader(const llama_governor_params & params);
    void log_platform_thermal_availability() const;
    // Samples the legs and hands the headroom to the engine; a no-op when
    // there is no reader (decode_hop_tokens == 0).
    void feed_decode_headroom();
    // Reads at the interval, then forwards a changed status under profile_mutex_.
    void refresh_platform_thermal();

    llama_governor * governor_ = nullptr;
    uint32_t decode_hop_tokens_ = 0;
    // Binding mirror of the engine's decode_tokens_since_prefill_: it counts
    // the same 1-token batches and restarts at the same three sites, so
    // window k always opens at a mirror value of k * decode_hop_tokens_.
    uint32_t decode_tokens_since_prefill_ = 0;
    llama_governor_fit gpu_fit_ = llama_governor_fit::Unknown;
    // Decode-hop leg reader, built only when decode_hop_tokens > 0.
    std::unique_ptr<rn_thermal_legs> decode_legs_;
    // Atomic: decode() writes it on the decode thread while the override
    // setter (allowed to overlap decode) reads it from a pool worker.
    std::atomic<bool> failed_{false};
    mutable std::mutex profile_mutex_;
    bool profile_valid_ = false;
    // The last profile accepted by the engine.
    llama_governor_thermo_profile thermo_profile_;
    // Native platform thermal reader, one per governor lifetime, and the
    // Wall-clock stamp of the last platform read.
    rn_platform_thermal platform_thermal_;
    int64_t platform_thermal_read_us_ = 0;
    rn_platform_thermal_send_state platform_thermal_send_state_;
    bool platform_refusal_logged_ = false;
    std::string failure_reason_;
};

} // namespace rnllama
