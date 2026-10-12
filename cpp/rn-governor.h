#pragma once

#include "llama-ext.h"
#include "llama.h"
#include "rn-battery-sysfs.h"
#include "rn-governor-samples.h"
#include "rn-platform-thermal.h"
#include "rn-thermal-legs.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

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
    // Stage 0 observation: the engine stats and the per-completion decode-hop
    // record (stamps, first sample, battery probe, window ring, trips) copied
    // under ONE engine_mutex_ hold, so both describe the same instant - the
    // JSI stats task reads them together.
    std::pair<llama_governor_stats, rn_governor_samples> stats_with_samples() const;

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
    // The one-time OPEN 4 probe, logged on its own tag: KALSA_GOVERNOR_PLATFORM
    // stays status-only by contract.
    void log_battery_sysfs() const;
    // Samples the legs and hands the headroom to the engine; a no-op when
    // there is no reader (decode_hop_tokens == 0). Called only from
    // before_decode_batch, which holds engine_mutex_. Returns the combined
    // reading so the observation record reuses it (one sysfs read per window).
    rn_thermal_legs::headroom feed_decode_headroom();
    // Stage 0 observation of one window boundary: the freshness snapshot, the
    // combined zone reading just taken, and the engine's applied values are
    // held PENDING - committed by commit_window only once the decode step
    // this boundary precedes actually ran, so a paused first batch leaves the
    // record empty. Caller holds engine_mutex_.
    void stage_window(int64_t now_ms, const rn_thermal_legs::headroom & legs);
    // Commits the pending window (plus the first-step latch and its battery
    // probe for the completion's first window) and advances the last-step
    // stamp; called only after a 1-token decode step returned rc 0. Caller
    // holds engine_mutex_.
    void commit_window(int64_t step_end_ms);
    // Completion start and cache clear: the record and any staged boundary
    // restart together. Caller holds engine_mutex_.
    void reset_observation_record();
    // Reads status and headroom on their own intervals, then forwards a
    // changed status or headroom under profile_mutex_.
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
    // Serializes the engine calls the threading contract forbids overlapping.
    // decode, set_decode_headroom and the two KV/stat rewinds run on the
    // completion thread; set_thermo_profile, clear_cache and stats run on JSI
    // promise workers, where throwIfContextBusy is a check, not a lock. Lock
    // order is profile_mutex_, then engine_mutex_, everywhere; nothing takes
    // profile_mutex_ while holding engine_mutex_.
    mutable std::mutex engine_mutex_;
    bool profile_valid_ = false;
    // The last profile accepted by the engine.
    llama_governor_thermo_profile thermo_profile_;
    // Native platform thermal reader, one per governor lifetime, and the
    // monotonic-clock stamp (ggml_time_us) of the last status and headroom reads.
    rn_platform_thermal platform_thermal_;
    int64_t platform_thermal_read_us_ = 0;
    int64_t platform_headroom_read_us_ = 0;
    rn_platform_thermal_headroom_state platform_headroom_state_;
    rn_platform_thermal_send_state platform_thermal_send_state_;
    // Monotonic stamp of the last native read that returned a status.
    int64_t platform_native_valid_us_ = 0;
    bool platform_refusal_logged_ = false;
    std::string failure_reason_;
    // Stage 0 observation record; the decode thread writes it under engine_mutex_.
    rn_governor_samples samples_;
    // Freshness of the thermo inputs for the window observation. The status
    // and app stamps are single relaxed atomics written wherever a profile
    // read answers (under profile_mutex_; the staging reads them lock-free
    // under engine_mutex_, because taking profile_mutex_ there would invert
    // the lock order above), so those ages are approximate and may mix
    // generations - Stage 0 never treats an age as exact. The forecast pair
    // is different: its value and stamp must belong to one read, so they are
    // plain fields both written and read under engine_mutex_ (the refresh
    // path takes that lock under profile_mutex_, the sanctioned order).
    std::atomic<int32_t> observed_status_{k_platform_thermal_absent};
    std::atomic<int64_t> observed_status_read_us_{0};
    std::atomic<int64_t> observed_app_sample_us_{0};
    float observed_headroom_ = std::numeric_limits<float>::quiet_NaN();
    int64_t observed_headroom_read_us_ = 0;
    // The window staged at the last boundary, pending until the step it
    // precedes ran (stage_window / commit_window).
    rn_window_sample pending_sample_{};
    rn_battery_sysfs pending_battery_{};
    int64_t pending_boundary_ms_ = 0;
    bool pending_first_window_ = false;
    bool pending_valid_ = false;
};

} // namespace rnllama
