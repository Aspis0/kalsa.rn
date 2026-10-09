#pragma once

#include "llama-ext.h"
#include "llama-governor-metrics.h"
#include "llama-governor-policy.h"

#include <limits>
#include <string>
#include <vector>

struct llama_governor;

// Device name holding most of a context model's repeating layers, resolved
// once per context at construction (llama-governor.cpp; used by
// llama-governor-legs.cpp).
std::string context_layers_device(llama_context * ctx);

struct llama_governor {
    llama_governor(llama_model * model_prefill, llama_model * model_decode,
                   llama_context_params params_prefill, llama_context_params params_decode);
    explicit llama_governor(llama_governor_params governor_params);
    // One-model form (F1): legs over ONE shared llama_model, each built with
    // llama_init_from_model_with_legs. Throws on a load that cannot serve any
    // leg or lacks the CPU leg; a leg whose device the model does not list is
    // simply absent.
    llama_governor(llama_model * model, const llama_governor_leg * legs, uint32_t n_legs,
                   llama_governor_params governor_params);
    ~llama_governor();

    // Threading contract: decode(), set_thermo_profile(),
    // set_decode_headroom(), record_telemetry(), note_expert_route(), and
    // stats() are decode-thread methods and must not overlap.
    // set_prefill_override() is the one exception: it stores only the
    // atomic mode in the policy, so it may overlap decode(); the mode is
    // consumed at the next prefill latch (see admit_prefill).
    // stall_enter()/stall_exit() are the only worker-thread callbacks; they are mutex-protected.
    int32_t decode(llama_batch batch);
    // The ONLY sanctioned way to rewind KV under a governor: removes [p, end)
    // of sequence 0 on BOTH contexts and rewinds both watermarks. When every
    // side can express the partial rewind, both watermarks become
    // min(watermark, p). When a side cannot (hybrid recurrent rollback
    // window), that side's sequence is cleared entirely and BOTH watermarks
    // go to 0, so the next handoff re-commits [0, end) attention cells and
    // the wholesale recurrent state from the rewound side instead of
    // adopting stale cells; commit_side additionally rejects any destination
    // that still holds cells beyond the source end. A raw seq_rm on one
    // context (prefix reuse between turns) is exactly that bypass. Returns
    // false only when even the full clear could not drop a side.
    bool trim_sequence(llama_pos p);
    llama_context * context() const;
    llama_context * prefill_context() const;
    llama_context * decode_context() const;
    llama_governor_stats stats() const;
    void clear_cache(bool clear_data);
    void reset_prefill_stats();
    bool set_thermo_profile(const llama_governor_thermo_profile & profile, int64_t now_ms);
    // Decode-hop leg headroom (llama_governor_set_decode_headroom), gpu =
    // the one-model third leg (NaN keeps the two-model form); and the
    // present-leg mask for rule v2. Same decode-thread contract as
    // set_thermo_profile.
    bool set_decode_headroom(float cpu_headroom_c, float npu_headroom_c,
                             float gpu_headroom_c = std::numeric_limits<float>::quiet_NaN());
    bool set_decode_legs(bool npu, bool gpu, bool cpu);
    // Bench route dev hook: validates mode (0..2) into the policy override.
    // Atomic store only - no stats refresh (threading contract above); it
    // takes effect at the next prefill latch.
    bool set_prefill_override(int mode);
    void record_telemetry(const llama_governor_telemetry_sample & sample);
    void stall_enter();
    void stall_exit();
    bool note_expert_route(bool resident, float resident_score,
                           float flash_winner_score, float score_range);
    // The failed state is sticky and distinguishes real failures from the
    // flow-control -2 returns (thermal pause, chunking, reload-required),
    // which never set it - the binding keys its fallback on exactly that.
    bool is_failed() const { return failed; }
    const char * failure_reason() const { return failure_reason_; }

    // The context of one leg of the one-model form; NULL when the leg is
    // absent or this is the two-model form.
    llama_context * leg_ctx(llama_governor_engine engine) const;

private:
    friend struct llama_governor_pacing_test;
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
    void pace_decode(uint32_t n_tokens, uint64_t compute_us);
    int32_t admit_prefill(llama_batch batch, bool allow_chunking);
    int32_t select_decode();
    int32_t decode_impl(llama_batch batch, bool allow_chunking);
    int32_t fail(const char * message);

    // leg/context routing, shared by both forms (llama-governor-legs.cpp):
    // the accelerator context of today's ctx_prefill role (NPU leg first,
    // then the GPU leg, then the CPU leg; two-model: ctx_prefill itself,
    // ctx_decode on the CPU route)
    llama_context * prefill_ctx(bool cpu_route) const;
    // the decode context for the selected engine (one-model: its leg, the
    // CPU leg when absent; two-model: NPU on ctx_prefill, else ctx_decode)
    llama_context * decode_ctx(llama_governor_engine engine) const;
    // the side state of a present context; NULL for a foreign pointer
    side_state * side_of(llama_context * ctx);
    // every present context with its side state (2 in the two-model form,
    // 1..3 in the one-model form); returns the count
    size_t collect_sides(llama_context * ctxs[3], side_state * states[3]);
    // the engine whose leg owns this context in the one-model form
    llama_governor_engine leg_engine_of(llama_context * ctx) const;
    // the engine a decode TARGET runs on (two-model: ctx_prefill = NPU role)
    llama_governor_engine engine_of_ctx(llama_context * ctx) const;
    // stats_.decode_hop_pairs index for an unordered engine pair
    static size_t hop_pair_index(llama_governor_engine a, llama_governor_engine b);
    // rule v2 availability mask: present legs the floors did not drop
    void refresh_decode_legs();
    // turn-local per-leg samples the floor evidence is measured against
    // (refresh_decode_legs), indexed like llama_governor_engine
    uint64_t floor_mark_tokens_[3] = {};
    uint64_t floor_mark_us_[3] = {};
    // forced-leg rotation (see llama_governor_params::forced_leg_*)
    void forced_rotate();
    void init_forced_rotation(const llama_governor_params & governor_params);

    // one-model form storage; empty in the two-model form
    struct leg {
        llama_governor_engine engine = llama_governor_engine::CPU;
        llama_context * ctx = nullptr; // owned
        side_state state;
    };
    std::vector<leg> legs;

    // forced rotation spec, validated from llama_governor_params at
    // construction; empty means off
    std::vector<llama_governor_engine> forced_leg_sequence_;
    uint32_t forced_leg_tokens_ = 0;
    // window the current forced leg was decided for (UINT32_MAX = none yet);
    // reset wherever the decode-hop clock resets
    uint32_t forced_window_ = UINT32_MAX;
    llama_governor_engine forced_window_engine_ = llama_governor_engine::CPU;

    llama_context * ctx_prefill = nullptr;
    llama_context * ctx_decode = nullptr;
    llama_context * last_ctx = nullptr;
    // Backend device name of the device holding most of each context model's
    // repeating layers, resolved once in the constructor (llama-governor.cpp):
    // the route facts copy one of these per prefill chunk instead of counting
    // layers on the decode path.
    std::string prefill_layers_device_;
    std::string decode_layers_device_;
    side_state prefill_state;
    side_state decode_state;
    llama_governor_policy policy_;
    bool policy_enabled_ = false;
    llama_governor_stats stats_;
    uint64_t decode_paced_us_ = 0;
    void (*decode_sleep_fn_)(uint64_t) = nullptr;
    llama_governor_counter_sample telemetry_;
    llama_governor_prefill_tally prefill_tally_;
    llama_governor_stall_union stall_union_;
    phase last_phase = phase::None;
    prefill_route prefill_route_ = prefill_route::Undecided;
    // Decode engine chosen for the batch being routed (select_decode); NPU
    // sends the batch to ctx_prefill, the HTP-pinned context when the lane is
    // resolved, instead of ctx_decode.
    llama_governor_engine decode_engine_ = llama_governor_engine::CPU;
    // Generated tokens since the last prefill entry - the decode hop's
    // window clock, paced by both hop rules (headroom and alternation) and
    // restarted where the route latch re-arms.
    uint32_t decode_tokens_since_prefill_ = 0;
    // What decided the current hop window's leg ("headroom"/"alternation"),
    // copied from the policy selection; null while the hop is inactive, so a
    // safety exit that moves the work keeps the bare hop direction label.
    const char * decode_hop_rule_ = nullptr;
    // Snapshot of the /bench route override, taken with the route latch so
    // every route fact of one prefill latch reports the same mode and the
    // same causal decision, whatever a concurrent push does afterwards. The
    // latch re-arms only at the next prefill entry from another phase, at
    // clear_cache, or at a stats reset - which may pick up a newer mode.
    llama_governor_prefill_mode turn_prefill_mode_ = llama_governor_prefill_mode::Auto;
    bool turn_override_decided_ = false;
    bool hot_plugged_announced_ = false;
    bool failed = false;
    const char * failure_reason_ = nullptr;
    // Storage for the decode/commit-failure reason; failure_reason_ points
    // into it.
    char decode_failure_reason_[128] = "";
};

// Internal RN bridge: returns the constructor error without emitting a second
// fallback line. The app owner owns the structured fallback log.
llama_governor * llama_governor_init_with_params_internal(
        llama_model *, llama_model *, llama_context_params, llama_context_params,
        llama_governor_params, std::string * failure_reason);
