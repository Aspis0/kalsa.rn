#include "llama-governor.h"

#include "llama-context.h"
#include "llama-kv-commit.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

llama_governor::llama_governor(llama_model * model_prefill, llama_model * model_decode,
                               llama_context_params params_prefill, llama_context_params params_decode)
    : llama_governor(model_prefill, model_decode, params_prefill, params_decode,
                     llama_governor_params{}, false) {
}

llama_governor::llama_governor(llama_governor_params governor_params)
    : policy_(governor_params), policy_enabled_(true) {
    stats_.npu_fit = policy_.npu_fit();
    stats_.prefill_token_cap = policy_.prefill_token_cap();
    stats_.expert_substitution_lambda = governor_params.expert_substitution_lambda;
    stats_.cache_budget_warning = policy_.cache_budget_warning();
}

llama_governor::llama_governor(llama_model * model_prefill, llama_model * model_decode,
                               llama_context_params params_prefill, llama_context_params params_decode,
                               llama_governor_params governor_params, bool policy_enabled)
    : policy_(governor_params), policy_enabled_(policy_enabled) {
    if (!model_prefill || !model_decode) {
        throw std::runtime_error("governor requires two models");
    }
    if (params_prefill.ctx_other || params_decode.ctx_other) {
        throw std::runtime_error("governor contexts must have independent KV caches");
    }

    ctx_prefill = llama_init_from_model(model_prefill, params_prefill);
    if (!ctx_prefill) {
        throw std::runtime_error("failed to create governor prefill context");
    }
    ctx_decode = llama_init_from_model(model_decode, params_decode);
    if (!ctx_decode) {
        llama_free(ctx_prefill);
        ctx_prefill = nullptr;
        throw std::runtime_error("failed to create governor decode context");
    }
    if (llama_kv_route_query(ctx_decode, ctx_prefill) == llama_kv_route::Reject) {
        llama_free(ctx_decode);
        llama_free(ctx_prefill);
        ctx_decode = nullptr;
        ctx_prefill = nullptr;
        throw std::runtime_error("governor contexts have incompatible KV routes");
    }
    stats_.npu_fit = policy_.npu_fit();
    stats_.prefill_token_cap = policy_.prefill_token_cap();
    stats_.expert_substitution_lambda = governor_params.expert_substitution_lambda;
    stats_.cache_budget_warning = policy_.cache_budget_warning();
    stats_.prefill_ctx_ngl = ctx_prefill->get_model().n_gpu_layers();
    if (stats_.cache_budget_warning) {
        LLAMA_LOG_WARN("%s: cache budget is smaller than one expert cycle; streaming will thrash\n", __func__);
    }
}

llama_governor::~llama_governor() {
    llama_free(ctx_decode);
    llama_free(ctx_prefill);
}

bool llama_governor::sequence_end(llama_context * ctx, llama_seq_id seq_id, llama_pos & end) {
    const llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
    if (max_pos < 0) {
        end = 0;
        return true;
    }
    if (max_pos == std::numeric_limits<llama_pos>::max()) {
        return false;
    }
    end = max_pos + 1;
    return true;
}

bool llama_governor::commit_side(llama_context * src_ctx, llama_context * dst_ctx,
                                 side_state & src, side_state & dst, const char * direction) {
    const auto route = llama_kv_route_query(dst_ctx, src_ctx);
    if (route == llama_kv_route::Reject) {
        LLAMA_LOG_ERROR("%s: %s route rejected\n", __func__, direction);
        return false;
    }

    llama_pos end = 0;
    if (!sequence_end(src_ctx, 0, end) || end < src.watermark) {
        LLAMA_LOG_ERROR("%s: %s watermark is invalid\n", __func__, direction);
        return false;
    }
    if (end > src.watermark) {
        size_t copied_bytes = 0;
        // Keep staged selection on the same v_trans predicate used by mirror(); otherwise
        // FA-enabled caches would be counted as staged while copying V naively.
        const auto mode = route == llama_kv_route::MirrorAndCopy && llama_kv_context_can_stage_v(src_ctx)
            ? llama_kv_commit_mode::Staged : llama_kv_commit_mode::Naive;
        llama_kv_commit_stats commit_stats{};
        const int64_t t0 = lm_ggml_time_us();
        const bool ok = llama_kv_commit_with_stats(dst_ctx, src_ctx, 0, src.watermark, end,
                                                   &copied_bytes, mode, &commit_stats);
        stats_.commit_us += lm_ggml_time_us() - t0;
        if (!ok) {
            LLAMA_LOG_ERROR("%s: %s KV commit failed\n", __func__, direction);
            return false;
        }
        stats_.commit_bytes += copied_bytes;
        ++stats_.commit_count;
        stats_.commit_naive_count += mode == llama_kv_commit_mode::Naive;
        stats_.commit_staged_count += mode == llama_kv_commit_mode::Staged;
        stats_.commit_transfers_k += commit_stats.transfers_k;
        stats_.commit_transfers_naive += commit_stats.transfers_naive;
        stats_.commit_transfers_staged += commit_stats.transfers_staged;
    }

    src.watermark = end;
    dst.watermark = end;
    return true;
}

void llama_governor::record_side(llama_context * ctx, side_state & side, bool prefill, uint32_t n_tokens) {
    const uint64_t reused = static_cast<uint64_t>(ctx->perf_get_data().n_reused);
    const uint32_t n_ubatch = ctx->n_ubatch();
    side.ubatches_submitted += (n_tokens + n_ubatch - 1) / n_ubatch;
    side.n_reused = reused;
    side.graph_builds = side.ubatches_submitted - side.n_reused;
    side.n_splits = lm_ggml_backend_sched_get_n_splits(ctx->get_sched());
    ++side.calls;
    if (prefill) {
        stats_.prefill_n_reused = side.n_reused;
        stats_.prefill_n_splits = side.n_splits;
        stats_.prefill_graph_builds = side.graph_builds;
        stats_.prefill_n += n_tokens;
        const size_t used = std::strlen(stats_.prefill_chunks);
        if (used < sizeof(stats_.prefill_chunks)) {
            const char * separator = used == 0 ? (prefill_route_ == prefill_route::CPU ? "cpu:" : "") : "+";
            std::snprintf(stats_.prefill_chunks + used, sizeof(stats_.prefill_chunks) - used,
                          "%s%u", separator, n_tokens);
        }
    } else {
        stats_.decode_n_reused = side.n_reused;
        stats_.decode_n_splits = side.n_splits;
        stats_.decode_graph_builds = side.graph_builds;
    }
}

int32_t llama_governor::fail(const char * message) {
    failed = true;
    failure_reason_ = message;
    LLAMA_LOG_ERROR("%s: %s\n", __func__, message);
    return -1;
}

int32_t llama_governor::decode(llama_batch batch) {
    return decode_impl(batch, true);
}

int32_t llama_governor::decode_impl(llama_batch batch, bool allow_chunking) {
    if (failed) {
        return fail("governor is failed");
    }
    if (batch.n_tokens <= 0) {
        return fail("decode requires a non-empty batch");
    }
    if (batch.n_seq_id && batch.seq_id) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.n_seq_id[i] != 1 || !batch.seq_id[i] || batch.seq_id[i][0] != 0) {
                return fail("governor supports sequence 0 only");
            }
        }
    }

    const bool is_prefill = batch.n_tokens > 1;
    if (is_prefill && allow_chunking && last_phase != phase::Prefill) {
        prefill_route_ = prefill_route::Undecided;
    }

    if (policy_enabled_) {
        refresh_policy_stats();
        const int32_t policy_rc = is_prefill ? admit_prefill(batch, allow_chunking) : select_decode();
        if (policy_rc != 0) {
            return policy_rc == k_prefill_chunked ? 0 : policy_rc;
        }
    }

    const bool cpu_prefill = is_prefill && prefill_route_ == prefill_route::CPU;
    llama_context * target = cpu_prefill ? ctx_decode : (is_prefill ? ctx_prefill : ctx_decode);
    side_state * state = cpu_prefill ? &decode_state : (is_prefill ? &prefill_state : &decode_state);
    const phase next_phase = is_prefill ? phase::Prefill : phase::Decode;

    if (is_prefill && last_phase != phase::Prefill) {
        prefill_tally_.begin(telemetry_);
    } else if (!is_prefill && last_phase == phase::Prefill) {
        prefill_tally_.end(telemetry_);
        record_tally();
    }

    const bool cpu_route_handoff = prefill_route_ == prefill_route::CPU &&
        (is_prefill || last_phase == phase::Prefill);
    if (last_phase != phase::None && last_phase != next_phase && !cpu_route_handoff) {
        const bool ok = is_prefill
            ? commit_side(ctx_decode, ctx_prefill, decode_state, prefill_state, "decode-to-prefill")
            : commit_side(ctx_prefill, ctx_decode, prefill_state, decode_state, "prefill-to-decode");
        if (!ok) {
            return fail("route Reject during phase handoff");
        }
    }

    const int64_t t0 = is_prefill ? lm_ggml_time_us() : 0;
    const int32_t rc = llama_decode(target, batch);
    if (rc != 0) {
        LLAMA_LOG_ERROR("%s: llama_decode failed with rc=%d\n", __func__, rc);
        failed = true;
        return rc;
    }

    if (is_prefill) {
        stats_.prefill_us += static_cast<uint64_t>(lm_ggml_time_us() - t0);
    }

    record_side(target, *state, is_prefill, batch.n_tokens);
    last_ctx = target;
    last_phase = next_phase;
    return 0;
}
