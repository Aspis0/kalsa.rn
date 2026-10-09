#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <limits>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Proactive memory pressure hook. The callback fires at most once per
// llama_decode call, before the batch (and any pending memory update) is
// processed, when memory utilization has reached trigger_frac; it re-arms
// automatically once utilization drops back below the threshold. Intended
// for proactive compaction: by the time a batch fails to fit, summarizing a
// chat gracefully is already too late.
//
// used_frac semantics per memory type:
//   - KV caches report the WORST STREAM's used/total cells: find_slot()
//     fails per stream, so one exhausted slot must fire even when other
//     slots are empty.
//   - iSWA memories report the base (non-sliding) part only. SWA cells are
//     recycled in place, so the sliding ring saturates at 1.0 in normal
//     single-sequence use and is never the binding constraint. Pure-SWA
//     layouts read 0.0 and never fire (a pure-SWA cache cannot fail for a
//     single sequence).
//   - hybrid memories report their attention (KV) part; the recurrent tail
//     has no meaningful fraction. Exhausting recurrent sequence slots is
//     NOT reported by this hook.
//   - recurrent memories and the DeepSeek DSA/DSV4 caches report a negative
//     value and never fire.
//
// Contract:
//   - install on the context that OWNS the memory. Shared-memory contexts
//     (MTP draft, Gemma4Assistant) report the owner's fraction but reject
//     every mutation, so they cannot act on the signal; installing on both
//     contexts of one physical cache double-fires.
//   - the setter is not thread-safe; call it before the first decode. The
//     callback runs on the decode thread and must not re-enter llama_decode
//     or free the context.
//   - the reported value is observed before pending memory updates (shifts,
//     copies), so with context shifting enabled it can include cells a
//     pending shift is about to drop.
//   - valid trigger_frac is [0, 1]; non-finite or > 1 disables the hook, a
//     negative value is treated as 0. Leaving the hook unset (the default)
//     costs one null check per decode.
typedef void (*llama_memory_pressure_cb)(float used_frac, void * user_data);

LLAMA_API void llama_memory_set_pressure_callback(
        struct llama_context * ctx,
        llama_memory_pressure_cb cb,
        void * user_data,
        float trigger_frac);

// Marks the entries that a joint decision head (clef) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

/** Fork extension (one model, one context per decode "leg"): construct a context whose device
 *  set is an explicit leg over the shared llama_model, instead of the model's load-time device
 *  list. Several contexts (legs) can live over one llama_model, each with its own backends,
 *  scheduler and memory; the model's weight buffers are neither duplicated nor moved.
 *
 *  leg_devices is a NULL-terminated array of devices:
 *    - NULL      = today's behaviour exactly (the context builds its backends from the model's
 *                  devices);
 *    - {NULL}    = the CPU leg (no offload device; the context runs entirely on CPU);
 *    - otherwise = the leg's offload devices, in order, each listed at most once and never a
 *                  CPU device (the CPU backend is always part of the context; the CPU leg is
 *                  the empty list). Every device must be one of the model's devices, else the
 *                  call fails. The context's backends are these devices plus the always-present
 *                  CPU backend; the FIRST device owns the leg's per-context buffers (KV cache,
 *                  recurrent state, output), where today they follow the model's per-layer
 *                  placement.
 *
 *  Load contract: "one of the model's devices" means the model's load list must name every
 *  device some leg will pick. The one-copy three-leg load lists devices {HTP0, GPUOpenCL} with
 *  n_gpu_layers covering all layers and tensor_split {1, 0} - every layer on HTP0, GPUOpenCL
 *  in the list but holding none (the load the S23 J(1) arm already uses) - and the legs pick
 *  {HTP0, GPUOpenCL}, {GPUOpenCL} and {} from that list. The NPU leg keeps HTP0 first (its KV
 *  and per-context buffers follow HTP0) and carries GPUOpenCL second so OpenCL takes what the
 *  HTP refuses - the tied Q6_K lm_head stays one flat copy in HTP0-HOST-W and is read in place
 *  by the GPU's AoS kernel inside the NPU decode (owner decision 2026-10-05: an NPU decode on
 *  the shared copy REQUIRES the OpenCL HOST leg; without it the lm_head falls back to the CPU
 *  and the leg is not the J(1) configuration).
 *
 *  Shared-model rule: every leg over one model must pass identical rope/YaRN context params
 *  (rope_scaling_type, rope_freq_scale, yarn_orig_ctx). Context initialization keeps an
 *  upstream behaviour that writes into the shared model: a custom YaRN rescale adjusts
 *  model->hparams.n_ctx_train (see llama_init_from_model_with_legs). Legs with differing YaRN
 *  values would each rewrite that one field - the last-constructed leg wins silently - so they
 *  must agree up front.
 *
 *  Returns NULL on failure, like llama_init_from_model. */
LLAMA_API llama_context * llama_init_from_model_with_legs(
        llama_model * model,
        llama_context_params params,
        const ggml_backend_dev_t * leg_devices);

/** Per-phase admission of the Hexagon backend the context runs on (one weight copy). `offload_min` is the
 *  number of tokens an op needs before a host weight offloads it (>= 1); `offload_ops` a bitmask of the
 *  GGML_HEXAGON_ADMIT_* op kinds a host weight may offload, 0..7 (0 = offload nothing), from
 *  ggml-hexagon.h; `gate_min` the number of tokens any CPU-runnable op needs to be supported by the
 *  Hexagon backend (0 = gate off). Returns true if a backend of the context accepted the call (the values
 *  were valid); false if none has the setter (CPU/GPU-only context) or the values were invalid. When a
 *  backend's admission changed, the scheduler is re-reserved at the next llama_decode (about 0.1 s on the
 *  S23: call it at a lane change, never per phase). The admission belongs to the device, so every context
 *  on that device re-reserves at its next decode. Must not run concurrently with a reserve, split or
 *  decode on any context of the same device: the governor calls it from its single decode thread, between
 *  phases. */
LLAMA_API bool llama_set_backend_admission(struct llama_context * ctx, int64_t offload_min, uint32_t offload_ops, int64_t gate_min);

enum class llama_kv_route {
    Direct,
    MirrorAndCopy,
    Reject,
};

LLAMA_API llama_kv_route llama_kv_route_query(
        const struct llama_context * dst_ctx,
        const struct llama_context * src_ctx);

/** Transfer counts for one contiguous KV range; V counts are source readbacks. */
struct llama_kv_route_cost {
    uint64_t v_layers = 0;
    uint64_t v_naive_gets = 0;
    uint64_t v_staged_gets = 0;
};

LLAMA_API bool llama_kv_route_query_cost(
        const struct llama_context * dst_ctx,
        const struct llama_context * src_ctx,
        struct llama_kv_route_cost * cost);

enum class llama_kv_commit_mode {
    Naive,
    Staged,
};

enum class llama_governor_engine {
    CPU,
    GPU,
    NPU,           // Selected by prefill_engine() when npu_lane_enabled (owner rule 2026-09-28).
    GPU_COOLMODE,
};

enum class llama_governor_generation {
    Unknown,
    NoHTP,
    V73,
    V75,
    V79,
    V81,           // Unqualified for GPU prefill: prefill_engine() admits only V73/V75/V79.
};

enum class llama_governor_model_kind {
    Unknown,
    Dense,
    Hybrid,
    MoE,
};

enum class llama_governor_fit {
    Unknown,
    NotFit,
    Fit,
};

enum class llama_governor_cool_pays {
    Unknown,
    No,
    Yes,
};

/** Which present leg the decode-hop window rule prefers when effective
 *  headroom decides (rule v2 owner knob, llama-governor-policy-hop.cpp). */
enum class llama_governor_leg_weighting {
    // Today's shipped rule: every decode opens on the NPU, and a later
    // window moves to the best other leg only when it leads the current
    // leg's effective headroom by the hysteresis (exact ties keep the
    // NPU > GPU > CPU order).
    NPU_FIRST,
    // NPU_FIRST, except a window it keeps on the CPU goes to the GPU when
    // the GPU's effective headroom is at least the CPU's - a burst is
    // exactly one window, and the next burst may start only after a
    // non-GPU window has run.
    GPU_BURST,
    // The present legs in order of heat per token, cheapest first; the
    // window stays on the first leg that is fit to run it: the current leg
    // while its headroom holds decode_guard_headroom_c, an idle candidate
    // once its loaded headroom clears that plus the hysteresis. No fit leg
    // falls back to NPU_FIRST's loop. Decides from measured headroom, not a
    // per-SoC ranking, so phones with the same chip may differ.
    HEAT_RANK,
};

enum class llama_governor_thermal_state {
    Unknown,
    FAST,
    WARM,
    COOLMODE,
    CRITICAL,
    LOWBAT,
    Invalid,
};

enum class llama_governor_decision {
    Admit,
    Chunk,
    CPUFallback,
    Wait,
    Abort,
};

/** Inputs owned by the app/device layer for one governor session. */
struct llama_governor_params {
    uint32_t schema_version = 3;            // LaunchConfig v0.3.
    uint32_t capability_schema_version = 2;
    llama_governor_generation generation = llama_governor_generation::Unknown;
    llama_governor_model_kind model_kind = llama_governor_model_kind::Unknown;
    llama_governor_fit gpu_fit = llama_governor_fit::Unknown;
    llama_governor_fit npu_fit = llama_governor_fit::Unknown;
    bool htp_trunk_readable = false;
    bool htp_experts_readable = false;
    bool npu_lane_enabled = false;
    bool gpu_prefill_measured = false;
    bool bench_force_gpu_prefill = false; // bench-only: force GPU prefill when gpu_fit==Fit; production leaves it false
    bool cool_prefill_eligible = false;
    bool cool_delta_measured = false;
    bool kexp_cool_scope = false;
    llama_governor_cool_pays cool_pays = llama_governor_cool_pays::Unknown;
    bool reload_budget_available = true;
    float admission_margin_c = 0.5f;
    uint64_t cache_budget_bytes = 0;
    uint64_t expert_cycle_bytes = 0;
    float expert_substitution_lambda = 0.0f;
    // Decode-hop prototype: when > 0, decode alternates CPU / live NPU lane
    // every decode_hop_tokens generated tokens (policy select_decode); 0
    // keeps CPU-only decode.
    uint32_t decode_hop_tokens = 0;
    // Rule v2, three legs (one-model governor): per-leg availability floors
    // in tok/s - a leg whose measured decode rate falls below its floor is
    // dropped from the hop candidates (0 = no floor, today's behaviour). The
    // heat-per-token weighting (battery C*s/token from the lab; NPU 0.15-0.17,
    // CPU 0.30, GPU 0.35-0.42) subtracts weight*heat from a leg's effective
    // headroom; the weight defaults to 0 = pure headroom = the v1 rule.
    float decode_floor_tps_npu = 0;
    float decode_floor_tps_gpu = 0;
    float decode_floor_tps_cpu = 0;
    float decode_heat_weight = 0;
    float decode_heat_per_token_npu = 0;
    float decode_heat_per_token_gpu = 0;
    float decode_heat_per_token_cpu = 0;
    // Per-leg load step (C): a leg's zone temperature jumps when the leg
    // starts working, so an idle leg's headroom overstates what it will
    // have once it runs. The rule subtracts the step from every leg that is
    // not the current one (the current leg already reads loaded); 0 = today's
    // rule; invalid values are rejected to 0 at construction.
    float decode_load_step_npu_c = 0;
    float decode_load_step_gpu_c = 0;
    float decode_load_step_cpu_c = 0;
    // Used only by HEAT_RANK: the headroom (C) below which the current leg is
    // left. Invalid values (NaN, negative, infinite) are rejected to 5 at
    // construction.
    float decode_guard_headroom_c = 5;
    // Forced decode-leg rotation (F5 proof hook; off by default): when
    // forced_leg_tokens > 0, decode follows forced_leg_sequence
    // (engines CPU/GPU/NPU) round-robin, one engine per window of
    // forced_leg_tokens generated tokens, overriding select_decode's rule.
    // An engine whose leg is absent runs its window on the CPU leg. NULL/0
    // with a nonzero token count is invalid; 0 keeps today's behaviour.
    const llama_governor_engine * forced_leg_sequence = nullptr;
    uint32_t forced_leg_sequence_n = 0;
    uint32_t forced_leg_tokens = 0;
    // Decode-hop rule v2 owner knobs (llama-governor-policy-hop.cpp):
    // decode_headroom_tau_s > 0 smooths each per-leg headroom sample with
    // an exponential moving average over the monotonic gap between samples
    // (seconds; 0 = raw samples, the shipped behaviour; invalid values are
    // rejected to raw at construction), and decode_leg_weighting selects
    // the window rule.
    float decode_headroom_tau_s = 0;
    llama_governor_leg_weighting decode_leg_weighting = llama_governor_leg_weighting::NPU_FIRST;
};

/** One successful battery poll. Temperature is the dumpsys tenths-of-degrees-C
 *  value. t_idle_c is the plugged temperature in whole degrees C, forwarded
 *  raw by the client: the policy latches the plugged reference from the first
 *  plugged sample and judges it (validity gate, 1.5 C drift, clear on unplug). */
struct llama_governor_thermo_profile {
    int32_t batt_temp_tenths_c = 0;
    int32_t batt_level_pct = 100;
    bool plugged = false;
    bool sensor_valid = false;
    bool t_idle_valid = false;
    float t_idle_c = 0.0f;
    float trend_c_per_min = 0.0f;
    // Android PowerManager.getCurrentThermalStatus(): -1 absent/unknown,
    // 0..6 NONE..SHUTDOWN; anything else is stored as absent (one clamp on
    // the policy side), never as a profile fault - an invalid profile is
    // what feeds the sticky abort. Absent means the platform never votes:
    // classification follows the battery alone, but the battery-driven rules
    // themselves moved on this branch (owner rule "NPU first": prefill may
    // pick NPU in battery COOLMODE too), so absent is not the old behavior.
    // The binding vendors the engine and compiles against this header, so
    // the default -1 keeps an unaware caller at "absent"; feeding a real
    // status requires the binding to parse the field, and until it does
    // the status stays absent.
    int32_t platform_thermal_status = -1;
};

/** Actual tensor_get/set counts for one commit. K is always naive, including in Staged mode. */
struct llama_kv_commit_stats {
    uint64_t transfers_k = 0;
    // V-only counts; staged commits do not contribute to transfers_naive.
    uint64_t transfers_naive = 0;
    uint64_t transfers_staged = 0;
    uint64_t v_gets_naive = 0;
    uint64_t v_sets_naive = 0;
    uint64_t v_gets_staged = 0;
    uint64_t v_sets_staged = 0;
    uint64_t staged_graph_builds = 0;
};

LLAMA_API bool llama_kv_commit(
        struct llama_context * dst_ctx,
        struct llama_context * src_ctx,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1);

LLAMA_API bool llama_kv_commit_ex(
        struct llama_context * dst_ctx,
        struct llama_context * src_ctx,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        llama_kv_commit_mode mode,
        struct llama_kv_commit_stats * stats);

/** Per-governor prefill override of the `/bench route` dev hook: a REQUEST
 *  consulted in prefill_engine() only after the safety early-returns.
 *  Also the encoding of llama_governor_set_prefill_override's mode argument
 *  and of llama_governor_route_chunk::{requested,actual}. */
enum class llama_governor_prefill_mode {
    Auto = 0, // no override: today's plan decides
    CPU = 1,
    GPU = 2,
};

/** One executed prefill chunk of a completion (bench route evidence). */
struct llama_governor_route_chunk {
    uint32_t index = 0;                 // 0-based within the completion
    llama_governor_prefill_mode requested = llama_governor_prefill_mode::Auto;
    llama_governor_prefill_mode actual = llama_governor_prefill_mode::CPU;
    uint32_t tokens = 0;
    uint64_t prefill_ms = 0;
    // Causal: true only when the prefill engine's decision came from the
    // override - the safety verdict did not preempt it and, for GPU, the
    // gpu_fit==Fit gate allowed it. LOWBAT+cpu and NotFit+gpu are false.
    bool forced = false;
    // ggml backend device name of the device that holds most of the layers of
    // the model behind the context that ran the chunk ("HTP0", "GPUOpenCL",
    // "CPU"). `actual` names the ROUTE, and a phone NPU load labels the route
    // GPU while Hexagon runs the matmuls, so consumers that need the device
    // read this. With the KV cache not offloaded (offload_kqv false, the NPU
    // prefill lane's no_kv_offload) the graph pins the nodes between the KV
    // store and the attention output to the CPU (llama-graph.cpp), so those
    // attention nodes do not run on this device. Fixed capacity,
    // NUL-terminated; longer names are truncated.
    char layers_device[32] = "";
};

/**
 * Statistics collected by llama_governor from existing context/KV APIs and
 * explicitly supplied router or telemetry inputs.
 * n_reused counts ubatches, not llama_decode calls. Graph-build counts are
 * ubatches_submitted - n_reused; llama has no public graph-build timer.
 * Split constancy is informative with CPU+GPU backends, but is necessarily
 * constant at one for the CPU-only contexts used by the step-5 test.
 * Delta-net hybrids copy a fixed recurrent state per commit (about 52.7 MB and
 * 99.65% of qwen35 bytes), so the §5 commit_ms/prefill_ms < 2% gate is not
 * trustworthy for short prefills until dirty-range recurrent commit is added.
 */
struct llama_governor_stats {
    uint64_t prefill_n_reused = 0;
    uint64_t decode_n_reused = 0;
    int32_t  prefill_n_splits = 0;
    int32_t  decode_n_splits = 0;
    uint64_t prefill_graph_builds = 0;
    uint64_t decode_graph_builds = 0;
    uint64_t commit_bytes = 0;
    uint64_t commit_us = 0;
    uint64_t commit_count = 0;
    uint64_t commit_naive_count = 0;
    uint64_t commit_staged_count = 0;
    uint64_t commit_transfers_k = 0;
    // V-only: K transfers are reported separately in commit_transfers_k.
    uint64_t commit_transfers_naive = 0;
    uint64_t commit_transfers_staged = 0;
    llama_governor_engine prefill_engine = llama_governor_engine::CPU;
    llama_governor_engine decode_engine = llama_governor_engine::CPU;
    llama_governor_thermal_state thermal_state = llama_governor_thermal_state::Unknown;
    llama_governor_fit npu_fit = llama_governor_fit::Unknown;
    uint32_t prefill_token_cap = 0; // 0 represents LaunchConfig null.
    bool decode_requires_reload = false;
    uint32_t last_router_rule = 0;
    uint64_t cpu_to_gpu_engagements = 0;
    // Decode-hop observability: decode-to-decode context switches executed,
    // tokens decoded on each context, the commit cost those switches paid
    // (already included in commit_bytes / commit_us above), and the windows
    // whose leg the headroom rule decided (zero while the alternation
    // fallback decides every window).
    uint64_t decode_hops = 0;
    uint64_t decode_tokens_cpu = 0;
    uint64_t decode_tokens_npu = 0;
    uint64_t decode_tokens_gpu = 0;   // one-model governor legs only (forced rotation)
    uint64_t decode_us_cpu = 0;       // per-leg decode time (one-model legs / rotation)
    uint64_t decode_us_npu = 0;
    uint64_t decode_us_gpu = 0;
    // hops and their commit cost per leg PAIR (unordered): 0 = NPU<->GPU,
    // 1 = NPU<->CPU, 2 = GPU<->CPU
    struct llama_governor_hop_pair {
        uint64_t hops = 0;
        uint64_t commit_bytes = 0;
        uint64_t commit_us = 0;
    };
    llama_governor_hop_pair decode_hop_pairs[3];
    uint64_t decode_hop_commit_bytes = 0;
    uint64_t decode_hop_commit_us = 0;
    uint64_t decode_hop_headroom_windows = 0;
    // the label of the last window decision ("headroom"/"alternation";
    // string literal, null while no window has been decided) - what a
    // free-running run prints as the reason per leg change
    const char * decode_hop_rule = nullptr;
    uint64_t stall_union_us = 0;
    uint64_t prefill_cpu_us = 0;
    uint64_t prefill_read_bytes = 0;
    uint64_t prefill_io_us = 0;
    uint64_t prefill_stall_us = 0;
    uint64_t prefill_management_us = 0;
    uint64_t decode_baseline_cpu_us = 0;
    uint64_t decode_baseline_read_bytes = 0;
    uint64_t decode_baseline_io_us = 0;
    uint64_t decode_baseline_stall_us = 0;
    uint64_t decode_baseline_management_us = 0;
    uint64_t expert_substitution_would_displace = 0;
    float expert_substitution_lambda = 0.0f;
    bool cache_budget_warning = false;
    uint64_t prefill_us = 0;
    uint64_t prefill_n = 0;
    uint32_t prefill_ctx_ngl = 0;
    char prefill_chunks[128] = {};
    // Per-completion route facts (reset by reset_prefill_stats, which the
    // binding calls at every completion start); overflow past the array
    // keeps the first entries, stops recording, and sets the flag so the
    // truncation is observable instead of silent.
    llama_governor_route_chunk route_chunks[64] = {};
    uint32_t route_chunk_count = 0;
    bool route_chunks_truncated = false;
    // Appended after the instrument-parsed fields above: the normalized
    // platform thermal status the classifier last saw (-1 absent) and which
    // input decided thermal_state - a pointer to the static strings
    // "battery" / "platform" set by refresh_policy_stats, never freed.
    int32_t platform_thermal_status = -1;
    const char * state_source = "battery";
};

/** Cumulative optional telemetry supplied by a streaming/backend integration. */
struct llama_governor_telemetry_sample {
    uint64_t cpu_us = 0;
    uint64_t read_bytes = 0;
    uint64_t read_us = 0;
    uint64_t stall_us = 0;
    uint64_t management_us = 0;
};

/**
 * Owns two independent contexts. Batches with n_tokens > 1 route to the
 * prefill context; batches with n_tokens == 1 route to the decode context.
 * Only sequence 0 is supported. A decode, commit, or route failure is sticky:
 * later calls keep failing rather than falling back to a single context.
 */
struct llama_governor;

/** Initialize a governor; model handles are borrowed for the governor lifetime. */
LLAMA_API struct llama_governor * llama_governor_init(
        struct llama_model * model_prefill,
        struct llama_model * model_decode,
        struct llama_context_params params_prefill,
        struct llama_context_params params_decode);

/** Initialize a governor with the v0.2 router inputs. */
LLAMA_API struct llama_governor * llama_governor_init_with_params(
        struct llama_model * model_prefill,
        struct llama_model * model_decode,
        struct llama_context_params params_prefill,
        struct llama_context_params params_decode,
        struct llama_governor_params governor_params);

/** One leg of the one-model governor: the engine the leg serves, the leg's
 *  device list (llama_init_from_model_with_legs) and its context params.
 *  The CPU leg ignores `devices` and is always the empty list; an NPU or GPU
 *  leg with devices == NULL is simply absent, and a leg whose first device
 *  the model does not list is absent too - the governor works with any
 *  subset that includes the CPU leg. GPU_COOLMODE is a decode state, not a
 *  leg. Params the legs must share (rope/YaRN) follow the
 *  llama_init_from_model_with_legs shared-model rule. */
struct llama_governor_leg {
    llama_governor_engine engine = llama_governor_engine::CPU;
    const ggml_backend_dev_t * devices = nullptr;
    llama_context_params params;
};

/** The one-model governor (F1): one shared llama_model, one context per leg
 *  built with llama_init_from_model_with_legs. The model handle is borrowed
 *  for the governor lifetime. Prefill runs where it runs in the two-model
 *  form (the NPU leg plays the ctx_prefill role, the GPU leg when there is
 *  no NPU, the CPU leg on the CPU route); decode follows the router, which
 *  the forced-leg rotation (llama_governor_params) can override. Returns
 *  NULL with a logged reason when no legs could be built or the CPU leg is
 *  missing. */
LLAMA_API struct llama_governor * llama_governor_init_one_model_with_params(
        struct llama_model * model,
        const struct llama_governor_leg * legs,
        uint32_t n_legs,
        struct llama_governor_params governor_params);

/** The context of one leg of a one-model governor; NULL when the leg is
 *  absent or the governor is the two-model form. */
LLAMA_API struct llama_context * llama_governor_leg_ctx(
        const struct llama_governor * governor,
        llama_governor_engine engine);

LLAMA_API void llama_governor_free(struct llama_governor * governor);

/**
 * Route by batch size (prefill for more than one token, decode for one).
 * Batches must use sequence 0; any failure permanently poisons the governor.
 */
LLAMA_API int32_t llama_governor_decode(
        struct llama_governor * governor,
        struct llama_batch batch);

/**
 * Rewind sequence 0 on both governor contexts, removing [p, end) where
 * supported. If both sides trim exactly, lower both commit watermarks to
 * min(watermark, p). If a side cannot be trimmed exactly, clear that side's
 * sequence entirely and reset both watermarks to zero after a successful
 * clear. Callers that trim a governor context's KV for prefix reuse MUST use
 * this instead of a raw llama_memory_seq_rm on one context, or the next phase
 * handoff fails with "watermark is invalid" and the other context keeps stale
 * cells. Returns false if the governor has failed or a required full clear
 * could not remove the sequence.
 */
LLAMA_API bool llama_governor_trim_sequence(
        struct llama_governor * governor,
        llama_pos p);

/** Update the session thermal state from a successful or failed poll. */
LLAMA_API bool llama_governor_set_thermo_profile(
        struct llama_governor * governor,
        struct llama_governor_thermo_profile profile,
        int64_t now_ms);

/** Update the decode-hop thermal headroom: for each leg, the margin in C
 *  between that leg's temperature and its first passive trip (the binding
 *  samples its thermal zones). NaN = unknown leg; with any present leg
 *  unknown the hop falls back to the fixed alternation, with all known it
 *  opens every decode on the NPU and moves a window only when another leg's
 *  effective headroom leads by the policy hysteresis (see select_decode).
 *  The gpu leg is the one-model governor's third leg; the two-value call
 *  keeps it NaN, which is exactly right for the two-model form. Same
 *  decode-thread contract as set_thermo_profile. */
LLAMA_API bool llama_governor_set_decode_headroom(
        struct llama_governor * governor,
        float cpu_headroom_c,
        float npu_headroom_c,
        float gpu_headroom_c = std::numeric_limits<float>::quiet_NaN());

/** Declare which decode legs exist and may hop (rule v2): the one-model
 *  governor sets this from its leg table (a leg the floors dropped is
 *  excluded); the default is NPU+CPU, the two-model form's set. The cpu
 *  flag is accepted for symmetry but the CPU leg is never excludable - it
 *  is the fallback that always exists. Same decode-thread contract as
 *  set_thermo_profile. */
LLAMA_API bool llama_governor_set_decode_legs(
        struct llama_governor * governor,
        bool npu,
        bool gpu,
        bool cpu);

/** Dev hook (/bench route): set the per-governor prefill override.
 *  mode: 0=auto (clears), 1=cpu, 2=gpu. Safe to call while another thread
 *  decodes: it stores the mode atomically. The prefill engine is chosen
 *  once per route latch - the first admission after a decode, clear_cache()
 *  or reset_prefill_stats() - and a latched route runs the whole prompt, so
 *  engine-changing inputs (this override, LOWBAT, gpu fit) take effect at
 *  the next latch, not at the next batch. Per-batch safety never waits for
 *  the latch: Invalid/CRITICAL abort, the >=40 C Wait, the small-prompt
 *  floor and the n_batch cap run on every admission. */
LLAMA_API bool llama_governor_set_prefill_override(
        struct llama_governor * governor,
        int mode);

/** Feed cumulative streaming/backend counters for phase-boundary attribution. */
LLAMA_API void llama_governor_record_telemetry(
        struct llama_governor * governor,
        struct llama_governor_telemetry_sample sample);

/** Critical-path stall hooks; nested intervals are unioned, not averaged. */
LLAMA_API void llama_governor_stall_enter(struct llama_governor * governor);
LLAMA_API void llama_governor_stall_exit(struct llama_governor * governor);

/** Record a residency observation at the expert-routing hook point. */
LLAMA_API bool llama_governor_note_expert_route(
        struct llama_governor * governor,
        bool resident,
        float resident_score,
        float flash_winner_score,
        float score_range);

LLAMA_API struct llama_context * llama_governor_get_context(
        const struct llama_governor * governor);

LLAMA_API struct llama_context * llama_governor_get_prefill_context(
        const struct llama_governor * governor);

LLAMA_API struct llama_context * llama_governor_get_decode_context(
        const struct llama_governor * governor);

/** Read governor counters; a null governor yields zeroed statistics. */
LLAMA_API void llama_governor_get_stats(
        const struct llama_governor * governor,
        struct llama_governor_stats * stats);

LLAMA_API void llama_governor_clear_cache(
        struct llama_governor * governor, bool clear_data);

LLAMA_API void llama_governor_reset_prefill_stats(struct llama_governor * governor);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
