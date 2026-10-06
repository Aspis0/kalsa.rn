#include "llama-governor.h"

#include "llama-context.h"
#include "llama-governor-device.h"
#include "llama-kv-commit.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// The device that holds most of the context model's repeating layers, i.e. the
// one the bulk of a chunk's matmuls run on. This is not dev_output(): that is
// the output layer's device, which under partial offload or a multi-device
// split is not where the bulk of the layers live. dev_layer() is sized to
// n_layer_all, so a zero-layer model contributes no names instead of throwing.
// Resolved once per context at construction; decode_impl copies the cached name.
// Shared with llama-governor-legs.cpp (declared in llama-governor.h).
std::string context_layers_device(llama_context * ctx) {
    const llama_model & model = ctx->get_model();
    std::vector<std::string> layer_devices;
    layer_devices.reserve(model.hparams.n_layer_all);
    for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
        layer_devices.emplace_back(ggml_backend_dev_name(model.dev_layer(static_cast<int>(il))));
    }
    return llama_governor_majority_device(layer_devices);
}

llama_governor::llama_governor(llama_model * model_prefill, llama_model * model_decode,
                               llama_context_params params_prefill, llama_context_params params_decode)
    : llama_governor(model_prefill, model_decode, params_prefill, params_decode,
                     llama_governor_params{}, false) {
}

llama_governor::llama_governor(llama_governor_params governor_params)
    : policy_(governor_params), policy_enabled_(true) {
    // F-08: forced rotation is a one-model proof hook; this test-only form
    // has no legs to rotate.
    if (governor_params.forced_leg_tokens != 0) {
        throw std::runtime_error("forced-leg rotation is a one-model proof hook");
    }
    stats_.npu_fit = policy_.npu_fit();
    stats_.prefill_token_cap = policy_.prefill_token_cap();
    stats_.expert_substitution_lambda = governor_params.expert_substitution_lambda;
    stats_.cache_budget_warning = policy_.cache_budget_warning();
}

llama_governor::llama_governor(llama_model * model_prefill, llama_model * model_decode,
                               llama_context_params params_prefill, llama_context_params params_decode,
                               llama_governor_params governor_params, bool policy_enabled)
    : policy_(governor_params), policy_enabled_(policy_enabled) {
    // F-08: forced rotation is a one-model proof hook; the two-model form
    // maps NPU to ctx_prefill and everything else to ctx_decode, so a forced
    // GPU window would be counted as GPU while running on the CPU context.
    if (governor_params.forced_leg_tokens != 0) {
        throw std::runtime_error("forced-leg rotation is a one-model proof hook");
    }
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
    prefill_layers_device_ = context_layers_device(ctx_prefill);
    decode_layers_device_ = context_layers_device(ctx_decode);
    stats_.npu_fit = policy_.npu_fit();
    stats_.prefill_token_cap = policy_.prefill_token_cap();
    stats_.expert_substitution_lambda = governor_params.expert_substitution_lambda;
    stats_.cache_budget_warning = policy_.cache_budget_warning();
    stats_.prefill_ctx_ngl = ctx_prefill->get_model().n_gpu_layers();
    if (stats_.cache_budget_warning) {
        LLAMA_LOG_WARN("%s: cache budget is smaller than one expert cycle; streaming will thrash\n", __func__);
    }
}

// Decode-leg label for the one-model hop and forced-rotation lines (the
// two-model form keeps its historical cpu/npu strings on its own code path).
static const char * engine_label(llama_governor_engine engine) {
    switch (engine) {
        case llama_governor_engine::CPU:         return "cpu";
        case llama_governor_engine::GPU:         return "gpu";
        case llama_governor_engine::NPU:         return "npu";
        case llama_governor_engine::GPU_COOLMODE: return "gpu-coolmode";
    }
    return "?";
}

llama_governor::~llama_governor() {
    llama_free(ctx_decode);
    llama_free(ctx_prefill);
    for (auto & l : legs) {
        llama_free(l.ctx);
    }
}

// Validates and copies the forced-leg rotation spec (llama_governor_params).
// A nonzero token count with no sequence, or a sequence naming GPU_COOLMODE,
// is a caller error and turns the governor away at construction.
void llama_governor::init_forced_rotation(const llama_governor_params & governor_params) {
    if (governor_params.forced_leg_tokens == 0) {
        if (governor_params.forced_leg_sequence != nullptr ||
            governor_params.forced_leg_sequence_n != 0) {
            LLAMA_LOG_WARN("%s: forced_leg_sequence ignored (forced_leg_tokens is 0)\n", __func__);
        }
        return;
    }
    if (governor_params.forced_leg_sequence == nullptr || governor_params.forced_leg_sequence_n == 0) {
        throw std::runtime_error("forced_leg_tokens > 0 requires a forced_leg_sequence");
    }
    for (uint32_t i = 0; i < governor_params.forced_leg_sequence_n; ++i) {
        const auto engine = governor_params.forced_leg_sequence[i];
        if (engine != llama_governor_engine::CPU &&
            engine != llama_governor_engine::GPU &&
            engine != llama_governor_engine::NPU) {
            throw std::runtime_error("forced_leg_sequence allows only CPU, GPU and NPU");
        }
        forced_leg_sequence_.push_back(engine);
    }
    forced_leg_tokens_ = governor_params.forced_leg_tokens;
}

// Forced decode-leg rotation (F5 proof and governor-bench hook): round-robin
// over the caller's engine sequence, one leg per window of
// forced_leg_tokens generated tokens, on the same window clock the decode
// hop uses. decode_impl calls it only after select_decode returned 0, so it
// overrides exactly the leg that rule chose - a safety state, a thermal wait
// or a required reload keeps winning. An engine whose leg is absent runs its
// window on the CPU leg (decode_ctx fallback).
void llama_governor::forced_rotate() {
    const uint32_t window = decode_tokens_since_prefill_ / forced_leg_tokens_;
    if (window != forced_window_) {
        forced_window_ = window;
        llama_governor_engine next = forced_leg_sequence_[window % forced_leg_sequence_.size()];
        if (!legs.empty() && leg_ctx(next) == nullptr) {
            // keep the log and the stats truthful about where the window runs
            LLAMA_LOG_WARN("%s: forced leg %s is absent; its windows run on the CPU leg\n",
                           __func__, engine_label(next));
            next = llama_governor_engine::CPU;
        }
        // One line per leg change, like the policy hop rule; window 0 follows
        // the prefill-to-decode transition, which logs its own line.
        if (window > 0 && next != forced_window_engine_) {
            LLAMA_LOG_INFO("governor: decode hop %s->%s at token %u (forced)\n",
                           engine_label(forced_window_engine_), engine_label(next),
                           decode_tokens_since_prefill_);
        }
        forced_window_engine_ = next;
    }
    // Applied on EVERY call: select_decode just ran and reset the engine to
    // its own choice, so the override must stand per token, not only at the
    // window boundary.
    decode_engine_ = forced_window_engine_;
    decode_hop_rule_ = "forced";
    stats_.decode_engine = forced_window_engine_;
    stats_.decode_requires_reload = false;
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
    llama_pos dst_end = 0;
    if (sequence_end(dst_ctx, 0, dst_end) && dst_end > end) {
        // The destination is stale beyond the source end (a GPU prewarm left
        // cells there while a CPU turn advanced the source, the 2026-09-24
        // S23 reject). Rewind through the sanctioned trim - the hybrid path
        // clears the untrimmable side wholesale and zeroes both watermarks,
        // so the copy below re-commits [0, end). The refuse stays as the
        // last line of defence for a trim that could not achieve even that.
        LLAMA_LOG_WARN("%s: %s destination ahead of source (%d > %d); sanctioned trim\n",
                       __func__, direction, (int) dst_end, (int) end);
        if (!trim_sequence(end) ||
            (sequence_end(dst_ctx, 0, dst_end) && dst_end > end)) {
            LLAMA_LOG_ERROR("%s: %s destination holds cells beyond the source end (%d > %d)\n",
                            __func__, direction, (int) dst_end, (int) end);
            return false;
        }
    }
    // The copy range is [DST.watermark, end) - what the DESTINATION is
    // missing, not what the source has added since the last commit. With two
    // contexts the watermarks are equal at every commit (both are set to end
    // below), so this is the historical behaviour; with three legs a leg
    // that has not participated yet (watermark 0) receives the whole
    // sequence, exactly F2's "the first GPU hop after a prefill receives the
    // entire turn".
    if (end > dst.watermark) {
        size_t copied_bytes = 0;
        // Keep staged selection on the same v_trans predicate used by mirror(); otherwise
        // FA-enabled caches would be counted as staged while copying V naively.
        const auto mode = route == llama_kv_route::MirrorAndCopy && llama_kv_context_can_stage_v(src_ctx)
            ? llama_kv_commit_mode::Staged : llama_kv_commit_mode::Naive;
        llama_kv_commit_stats commit_stats{};
        const int64_t t0 = ggml_time_us();
        bool ok = false;
        try {
            ok = llama_kv_commit_with_stats(dst_ctx, src_ctx, 0, dst.watermark, end,
                                            &copied_bytes, mode, &commit_stats);
        } catch (const std::exception & e) {
            // The OpenCL set_tensor path throws on device OOM (86f602f8a). An
            // escaping throw reaches JS as a generic rejection with no failed
            // flag, and the still-zeroed watermarks would make every next
            // turn retry the same full copy - latch it here instead.
            snprintf(decode_failure_reason_, sizeof(decode_failure_reason_),
                     "KV commit failed: %.80s", e.what());
            fail(decode_failure_reason_);
            return false;
        }
        stats_.commit_us += ggml_time_us() - t0;
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
    side.n_splits = ggml_backend_sched_get_n_splits(ctx->get_sched());
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

bool llama_governor::trim_sequence(llama_pos p) {
    // A latched commit failure may have left a half-copied side; the binding
    // calls this from loadPrompt before any governorFailed() check, so a
    // retry must not seq_rm anything. Touch nothing once failed.
    if (failed) {
        return false;
    }
    llama_context * side_ctxs[3];
    side_state * side_states[3];
    const size_t n_sides = collect_sides(side_ctxs, side_states);

    bool exact = true;
    bool untrimmable[3] = { false, false, false };
    for (size_t i = 0; i < n_sides; ++i) {
        llama_memory_t mem = llama_get_memory(side_ctxs[i]);
        // WHY pos_max decides here and not seq_rm's return value: for a shared
        // KV view seq_rm answers true without removing anything (the shared
        // early return in llama-kv-cache.cpp:427), so a "true" there never means
        // "trimmed". The trap is unreachable today: the governor ctor refuses
        // ctx_other, so a governor context is never a shared view.
        llama_memory_seq_rm(mem, 0, p, -1);
        const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
        if (pos_max != -1 && pos_max >= p) {
            // Hybrid: the recurrent rollback window refused, so neither the
            // recurrent state nor the attention cells beyond p were removed.
            untrimmable[i] = true;
            exact = false;
        }
    }
    if (exact) {
        for (size_t i = 0; i < n_sides; ++i) {
            side_states[i]->watermark = std::min(side_states[i]->watermark, p);
        }
        return true;
    }

    // A partial rewind is inexpressible on at least one side. Drop that
    // side's whole sequence and zero EVERY watermark, so the next handoff
    // re-commits [0, end) attention cells and the wholesale recurrent state
    // from the rewound side (llama-hybrid-commit.cpp) instead of letting
    // commit_side adopt the stale side.
    for (size_t i = 0; i < n_sides; ++i) {
        if (!untrimmable[i]) { continue; }
        llama_memory_t mem = llama_get_memory(side_ctxs[i]);
        llama_memory_seq_rm(mem, 0, -1, -1);
        if (llama_memory_seq_pos_max(mem, 0) != -1) {
            return false;
        }
    }
    for (size_t i = 0; i < n_sides; ++i) {
        side_states[i]->watermark = 0;
    }
    return true;
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
    // The route latch re-arms only on prefill entry from another phase (or
    // via clear_cache/reset_prefill_stats): within one prefill phase the
    // engine stays put. The hop clock restarts with the latch - here, and at
    // the binding's per-completion reset_prefill_stats call - so every turn
    // decides hop window 0 fresh, whichever rule runs (headroom opens it on
    // the NPU), even when the prompt was fully cached and its 1-token batch
    // never enters this prefill branch.
    if (is_prefill && allow_chunking && last_phase != phase::Prefill) {
        prefill_route_ = prefill_route::Undecided;
        decode_tokens_since_prefill_ = 0;
        forced_window_ = UINT32_MAX;
        policy_.reset_decode_hop();
        // each turn re-checks the floors against the legs' measured rates
        if (!legs.empty()) {
            refresh_decode_legs();
        }
    }

    if (policy_enabled_) {
        refresh_policy_stats();
        const int32_t policy_rc = is_prefill ? admit_prefill(batch, allow_chunking) : select_decode();
        if (policy_rc != 0) {
            return policy_rc == k_prefill_chunked ? 0 : policy_rc;
        }
        // Proof hook (F-07): the forced rotation replaces ONLY the leg choice
        // select_decode just made - the thermal wait/abort and reload gates
        // inside it have already spoken and win. Off (0) nothing changes.
        if (!is_prefill && forced_leg_tokens_ > 0) {
            forced_rotate();
        }
    }

    // The target context is chosen by (phase, prefill route, decode engine):
    // decode on the NPU is decode on ctx_prefill, the HTP-pinned context when
    // the lane is resolved - the binding forces matching q8_0 KV on both
    // contexts, so a committed cache is directly readable on either side and
    // a hop never needs a reload. The one-model form routes the same roles
    // through its legs (prefill_ctx / decode_ctx, llama-governor-legs.cpp).
    const bool cpu_prefill = is_prefill && prefill_route_ == prefill_route::CPU;
    llama_context * target = is_prefill ? prefill_ctx(cpu_prefill)
                                        : decode_ctx(decode_engine_);
    side_state * state = side_of(target);
    const phase next_phase = is_prefill ? phase::Prefill : phase::Decode;

    if (is_prefill && last_phase != phase::Prefill) {
        prefill_tally_.begin(telemetry_);
    } else if (!is_prefill && last_phase == phase::Prefill) {
        prefill_tally_.end(telemetry_);
        record_tally();
    }

    // One switch, one commit: whenever a batch moves to another context the
    // KV follows it - the prefill<->decode handoffs and the mid-turn decode
    // hops are the same event on this one code path (this also covers the
    // CPU-route prefill that follows a GPU prewarm: a prefill route switch
    // commits like any other batch that moves contexts). A CPU-route prefill,
    // a CPU decode and an NPU decode that follows its own NPU prefill all
    // stay on their context and never commit.
    const bool decode_hop = !is_prefill && last_phase == phase::Decode && target != last_ctx;
    if (last_ctx != nullptr && target != last_ctx) {
        // Name the decider in the hop label ("... (headroom)"/"... (alternation)"/
        // "... (forced)"); a safety exit that moves the work keeps the bare direction.
        char hop_direction[64];
        char hop_labeled[96];
        const char * direction = nullptr;
        if (is_prefill) {
            direction = last_phase == phase::Decode ? "decode-to-prefill" : "prefill route switch";
        } else if (last_phase == phase::Decode) {
            if (!legs.empty()) {
                std::snprintf(hop_direction, sizeof(hop_direction), "decode hop %s->%s",
                              engine_label(leg_engine_of(last_ctx)), engine_label(decode_engine_));
                direction = hop_direction;
            } else {
                direction = decode_engine_ == llama_governor_engine::NPU
                    ? "decode hop cpu-to-npu" : "decode hop npu-to-cpu";
            }
        } else {
            direction = "prefill-to-decode";
        }
        if (decode_hop && decode_hop_rule_ != nullptr) {
            // hop_labeled, not hop_direction: direction may point into it
            std::snprintf(hop_labeled, sizeof(hop_labeled), "%s (%s)",
                          direction, decode_hop_rule_);
            direction = hop_labeled;
        }
        side_state * src = side_of(last_ctx);
        const uint64_t commit_bytes_before = stats_.commit_bytes;
        const uint64_t commit_us_before = stats_.commit_us;
        if (!commit_side(last_ctx, target, *src, *state, direction)) {
            // commit_side may already have latched a specific reason (the KV
            // commit catch); keep it instead of overwriting with the generic
            // route Reject.
            return failed ? -1 : fail("route Reject during context switch");
        }
        if (decode_hop) {
            const uint64_t bytes = stats_.commit_bytes - commit_bytes_before;
            const uint64_t us = stats_.commit_us - commit_us_before;
            stats_.decode_hop_commit_bytes += bytes;
            stats_.decode_hop_commit_us += us;
            auto & pair = stats_.decode_hop_pairs[hop_pair_index(engine_of_ctx(last_ctx), decode_engine_)];
            ++pair.hops;
            pair.commit_bytes += bytes;
            pair.commit_us += us;
        }
    }

    const int64_t t0 = ggml_time_us();
    const int32_t rc = llama_decode(target, batch);
    if (rc != 0) {
        LLAMA_LOG_ERROR("%s: llama_decode failed with rc=%d\n", __func__, rc);
        // fail() would collapse the rc to -1; callers must still see the raw
        // engine code (e.g. -2 for GGML_STATUS_ALLOC_FAILED), so set the
        // sticky state and reason directly.
        snprintf(decode_failure_reason_, sizeof(decode_failure_reason_),
                 "llama_decode failed with rc=%d", rc);
        failure_reason_ = decode_failure_reason_;
        failed = true;
        return rc;
    }

    if (is_prefill) {
        stats_.prefill_us += static_cast<uint64_t>(ggml_time_us() - t0);
        // One route fact per executed prefill chunk (bench hook evidence),
        // stamped from the snapshot taken with the route latch: one latch,
        // one mode. A prefill -> decode -> prefill sequence inside one reset
        // interval re-latches and can record two modes. Overflow keeps the
        // first entries and raises the truncation flag instead of failing.
        if (stats_.route_chunk_count <
            sizeof(stats_.route_chunks) / sizeof(stats_.route_chunks[0])) {
            auto & chunk = stats_.route_chunks[stats_.route_chunk_count];
            chunk.index = stats_.route_chunk_count;
            chunk.requested = turn_prefill_mode_;
            chunk.actual = cpu_prefill ? llama_governor_prefill_mode::CPU
                                       : llama_governor_prefill_mode::GPU;
            chunk.tokens = static_cast<uint32_t>(batch.n_tokens);
            chunk.prefill_ms = static_cast<uint64_t>((ggml_time_us() - t0) / 1000);
            // Causal: true only when prefill_engine() took the override
            // branch (no safety preemption, Fit gate passed for GPU).
            chunk.forced = turn_override_decided_;
            // The CPU route runs on ctx_decode, so the device fact follows the
            // context that executed the chunk, not `actual`.
            std::snprintf(chunk.layers_device, sizeof(chunk.layers_device), "%s",
                          cpu_prefill ? decode_layers_device_.c_str()
                                      : prefill_layers_device_.c_str());
            ++stats_.route_chunk_count;
        } else {
            stats_.route_chunks_truncated = true;
        }
    }

    record_side(target, *state, is_prefill, batch.n_tokens);
    if (decode_hop) {
        ++stats_.decode_hops;
    }
    if (!is_prefill) {
        ++decode_tokens_since_prefill_;
        const uint64_t decode_us = static_cast<uint64_t>(ggml_time_us() - t0);
        // GPU is a decode engine only in the one-model form (the v2 hop rule
        // or forced rotation); the two-model router never selects it for
        // decode without a reload, so its counting is unchanged.
        switch (decode_engine_) {
            case llama_governor_engine::NPU:
                stats_.decode_tokens_npu += batch.n_tokens;
                stats_.decode_us_npu += decode_us;
                break;
            case llama_governor_engine::GPU:
                stats_.decode_tokens_gpu += batch.n_tokens;
                stats_.decode_us_gpu += decode_us;
                break;
            default:
                stats_.decode_tokens_cpu += batch.n_tokens;
                stats_.decode_us_cpu += decode_us;
                break;
        }
    }
    last_ctx = target;
    last_phase = next_phase;
    return 0;
}
