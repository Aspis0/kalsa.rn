#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct lm_ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default lm_ggml_type for a given ftype.
LLAMA_API lm_ggml_type llama_ftype_get_default_type(llama_ftype ftype);

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
        const lm_ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        lm_ggml_tensor ** tensors,
        lm_ggml_type * result_types,
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
using llama_memory_breakdown = std::map<lm_ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API lm_ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

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
    NPU,           // Deferred: v0.2 never selects this lane.
    GPU_COOLMODE,
};

enum class llama_governor_generation {
    Unknown,
    NoHTP,
    V73,
    V75,
    V79,
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
};

/** One successful battery poll. Temperature is the dumpsys tenths-of-°C value. */
struct llama_governor_thermo_profile {
    int32_t batt_temp_tenths_c = 0;
    int32_t batt_level_pct = 100;
    bool plugged = false;
    bool sensor_valid = false;
    bool t_idle_valid = false;
    float t_idle_c = 0.0f;
    float trend_c_per_min = 0.0f;
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

LLAMA_API void llama_governor_free(struct llama_governor * governor);

/**
 * Route by batch size (prefill for more than one token, decode for one).
 * Batches must use sequence 0; any failure permanently poisons the governor.
 */
LLAMA_API int32_t llama_governor_decode(
        struct llama_governor * governor,
        struct llama_batch batch);

/** Update the session thermal state from a successful or failed poll. */
LLAMA_API bool llama_governor_set_thermo_profile(
        struct llama_governor * governor,
        struct llama_governor_thermo_profile profile,
        int64_t now_ms);

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

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);
