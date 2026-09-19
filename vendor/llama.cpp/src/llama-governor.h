#pragma once

#include "llama-ext.h"
#include "llama-governor-metrics.h"
#include "llama-governor-policy.h"

#include <string>

struct llama_governor {
    llama_governor(llama_model * model_prefill, llama_model * model_decode,
                   llama_context_params params_prefill, llama_context_params params_decode);
    explicit llama_governor(llama_governor_params governor_params);
    ~llama_governor();

    // Threading contract: decode(), set_thermo_profile(), record_telemetry(),
    // note_expert_route(), and stats() are decode-thread methods and must not overlap.
    // stall_enter()/stall_exit() are the only worker-thread callbacks; they are mutex-protected.
    int32_t decode(llama_batch batch);
    llama_context * context() const;
    llama_context * prefill_context() const;
    llama_context * decode_context() const;
    llama_governor_stats stats() const;
    void clear_cache(bool clear_data);
    void reset_prefill_stats();
    bool set_thermo_profile(const llama_governor_thermo_profile & profile, int64_t now_ms);
    void record_telemetry(const llama_governor_telemetry_sample & sample);
    void stall_enter();
    void stall_exit();
    bool note_expert_route(bool resident, float resident_score,
                           float flash_winner_score, float score_range);
    const char * failure_reason() const { return failure_reason_; }

private:
    friend llama_governor * llama_governor_init_with_params_internal(
            llama_model *, llama_model *, llama_context_params, llama_context_params,
            llama_governor_params, std::string * failure_reason);

    llama_governor(llama_model * model_prefill, llama_model * model_decode,
                   llama_context_params params_prefill, llama_context_params params_decode,
                   llama_governor_params governor_params, bool policy_enabled);

    enum class phase { None, Prefill, Decode };
    // A turn chooses one prefill context at its first multi-token batch;
    // later batches cannot switch between CPU and GPU.
    enum class prefill_route { Undecided, GPU, CPU };
    // Internal control result; llama_decode uses 0, 1, 2, -1 and fatal values
    // below -1, so this cannot be mistaken for a native decode return.
    enum { k_prefill_chunked = -1001 };

    struct side_state {
        llama_pos watermark = 0; // exclusive next position not committed to the other side
        uint64_t n_reused = 0;
        uint64_t ubatches_submitted = 0;
        uint64_t graph_builds = 0;
        uint64_t calls = 0;
        int32_t n_splits = 0;
    };

    static bool sequence_end(llama_context * ctx, llama_seq_id seq_id, llama_pos & end);
    bool commit_side(llama_context * src_ctx, llama_context * dst_ctx,
                     side_state & src, side_state & dst, const char * direction);
    void record_side(llama_context * ctx, side_state & side, bool prefill, uint32_t n_tokens);
    void record_tally();
    void refresh_policy_stats();
    int32_t admit_prefill(llama_batch batch, bool allow_chunking);
    int32_t select_decode();
    int32_t decode_impl(llama_batch batch, bool allow_chunking);
    int32_t fail(const char * message);

    llama_context * ctx_prefill = nullptr;
    llama_context * ctx_decode = nullptr;
    llama_context * last_ctx = nullptr;
    side_state prefill_state;
    side_state decode_state;
    llama_governor_policy policy_;
    bool policy_enabled_ = false;
    llama_governor_stats stats_;
    llama_governor_counter_sample telemetry_;
    llama_governor_prefill_tally prefill_tally_;
    llama_governor_stall_union stall_union_;
    phase last_phase = phase::None;
    prefill_route prefill_route_ = prefill_route::Undecided;
    bool hot_plugged_announced_ = false;
    bool failed = false;
    const char * failure_reason_ = nullptr;
};

// Internal RN bridge: returns the constructor error without emitting a second
// fallback line. The app owner owns the structured fallback log.
llama_governor * llama_governor_init_with_params_internal(
        llama_model *, llama_model *, llama_context_params, llama_context_params,
        llama_governor_params, std::string * failure_reason);
