#include "llama-governor.h"

#include "llama-impl.h"

#include <algorithm>
#include <string>
#include <stdexcept>
#include <vector>

void llama_governor::record_tally() {
    const auto tally = prefill_tally_.snapshot(telemetry_);
    stats_.prefill_cpu_us = tally.delta.cpu_us;
    stats_.prefill_read_bytes = tally.delta.read_bytes;
    stats_.prefill_io_us = tally.delta.read_us;
    stats_.prefill_stall_us = tally.delta.stall_us;
    stats_.prefill_management_us = tally.delta.management_us;
    if (tally.closed) {
        stats_.decode_baseline_cpu_us = tally.post.cpu_us;
        stats_.decode_baseline_read_bytes = tally.post.read_bytes;
        stats_.decode_baseline_io_us = tally.post.read_us;
        stats_.decode_baseline_stall_us = tally.post.stall_us;
        stats_.decode_baseline_management_us = tally.post.management_us;
    }
}

void llama_governor::refresh_policy_stats() {
    if (prefill_route_ == prefill_route::Undecided) {
        stats_.prefill_engine = policy_.prefill_engine();
    } else if (prefill_route_ == prefill_route::CPU) {
        stats_.prefill_engine = llama_governor_engine::CPU;
    }
    stats_.thermal_state = policy_.thermal_state();
    stats_.npu_fit = policy_.npu_fit();
    stats_.prefill_token_cap = policy_.prefill_token_cap();
    stats_.cpu_to_gpu_engagements = policy_.cpu_to_gpu_engagements();
    stats_.cache_budget_warning = policy_.cache_budget_warning();
    stats_.last_router_rule = policy_.prefill_rule();
}

int32_t llama_governor::admit_prefill(llama_batch batch, bool allow_chunking) {
    if (prefill_route_ == prefill_route::CPU) {
        return 0;
    }

    const auto requested = policy_.prefill_engine();
    const auto admission = policy_.admit_prefill(
            requested, static_cast<uint32_t>(batch.n_tokens),
            policy_.current_temperature_c());
    stats_.last_router_rule = admission.rule;
    if (admission.decision == llama_governor_decision::Abort) {
        return fail("prefill admission aborted");
    }
    const bool cpu_fallback = admission.decision == llama_governor_decision::CPUFallback &&
        policy_.thermal_state() != llama_governor_thermal_state::FAST;
    if (prefill_route_ == prefill_route::Undecided &&
        (admission.decision == llama_governor_decision::Wait || cpu_fallback)) {
        prefill_route_ = prefill_route::CPU;
        stats_.prefill_engine = llama_governor_engine::CPU;
        stats_.prefill_chunks[0] = '\0';
        if (admission.decision == llama_governor_decision::Wait) {
            LLAMA_LOG_WARN("%s: prefill admission is waiting; routing this turn to CPU\n", __func__);
        }
        return 0;
    }
    if (prefill_route_ == prefill_route::Undecided) {
        prefill_route_ = prefill_route::GPU;
        stats_.prefill_engine = requested;
    }
    if (admission.decision == llama_governor_decision::Wait ||
        admission.decision == llama_governor_decision::CPUFallback) {
        // The first batch selected GPU; keep this turn on that context.
        return 0;
    }
    if (admission.tokens >= static_cast<uint32_t>(batch.n_tokens)) {
        return 0;
    }
    if (!allow_chunking || !batch.token) {
        LLAMA_LOG_INFO("%s: prefill requires caller chunk of %u tokens\n", __func__, admission.tokens);
        return -2;
    }

    std::vector<int32_t> chunk_sizes;
    int32_t remaining = batch.n_tokens;
    while (remaining > 0) {
        const auto next = policy_.admit_prefill(
                policy_.prefill_engine(), static_cast<uint32_t>(remaining),
                policy_.current_temperature_c());
        if (next.decision == llama_governor_decision::Abort || next.tokens == 0) {
            LLAMA_LOG_INFO("%s: prompt cannot be partitioned into safe tabled chunks\n", __func__);
            return -2;
        }
        if (next.decision == llama_governor_decision::Wait) {
            // No GPU chunk has run in this call yet; keep the turn's first route.
            return 0;
        }
        const int32_t count = std::min<int32_t>(next.tokens, remaining);
        if (count <= 1) {
            return -2;
        }
        chunk_sizes.push_back(count);
        remaining -= count;
    }

    int32_t offset = 0;
    for (const int32_t count : chunk_sizes) {
        llama_batch chunk = batch;
        chunk.n_tokens = count;
        chunk.token = batch.token + offset;
        chunk.pos = batch.pos ? batch.pos + offset : nullptr;
        chunk.n_seq_id = batch.n_seq_id ? batch.n_seq_id + offset : nullptr;
        chunk.seq_id = batch.seq_id ? batch.seq_id + offset : nullptr;
        chunk.logits = batch.logits ? batch.logits + offset : nullptr;
        const int32_t rc = decode_impl(chunk, false);
        if (rc != 0) {
            return rc;
        }
        offset += count;
    }
    return k_prefill_chunked;
}

int32_t llama_governor::select_decode() {
    const auto selection = policy_.select_decode(lm_ggml_time_us() / 1000);
    stats_.decode_engine = selection.engine;
    stats_.decode_requires_reload = selection.requires_reload;
    stats_.last_router_rule = selection.rule;
    stats_.cpu_to_gpu_engagements = policy_.cpu_to_gpu_engagements();
    if (selection.wait) {
        if (policy_.thermal_state() == llama_governor_thermal_state::Invalid ||
            policy_.thermal_state() == llama_governor_thermal_state::CRITICAL) {
            return fail("decode admission aborted");
        }
        LLAMA_LOG_WARN("%s: decode admission is waiting for a valid thermal profile\n", __func__);
        return -2;
    }
    if (selection.requires_reload) {
        LLAMA_LOG_INFO("%s: decode engine change requires a full context reload\n", __func__);
        return -2;
    }
    return 0;
}

llama_governor_stats llama_governor::stats() const {
    auto result = stats_;
    const auto tally = prefill_tally_.snapshot(telemetry_);
    result.prefill_cpu_us = tally.delta.cpu_us;
    result.prefill_read_bytes = tally.delta.read_bytes;
    result.prefill_io_us = tally.delta.read_us;
    result.prefill_stall_us = tally.delta.stall_us;
    result.prefill_management_us = tally.delta.management_us;
    result.stall_union_us = stall_union_.total(static_cast<uint64_t>(lm_ggml_time_us()));
    return result;
}

void llama_governor::clear_cache(bool clear_data) {
    if (ctx_prefill != nullptr) {
        llama_memory_clear(llama_get_memory(ctx_prefill), clear_data);
    }
    if (ctx_decode != nullptr) {
        llama_memory_clear(llama_get_memory(ctx_decode), clear_data);
    }
    prefill_state = side_state{};
    decode_state = side_state{};
    last_ctx = nullptr;
    last_phase = phase::None;
    prefill_route_ = prefill_route::Undecided;

    stats_.commit_bytes = 0;
    stats_.commit_us = 0;
    stats_.commit_count = 0;
    stats_.commit_naive_count = 0;
    stats_.commit_staged_count = 0;
    stats_.commit_transfers_k = 0;
    stats_.commit_transfers_naive = 0;
    stats_.commit_transfers_staged = 0;
}

void llama_governor::reset_prefill_stats() {
    prefill_route_ = prefill_route::Undecided;
    stats_.prefill_us = 0;
    stats_.prefill_n = 0;
    stats_.prefill_chunks[0] = '\0';
}

bool llama_governor::set_thermo_profile(const llama_governor_thermo_profile & profile, int64_t now_ms) {
    if (!policy_enabled_) {
        return false;
    }
    const bool ok = policy_.update_thermal(profile, now_ms);
    refresh_policy_stats();
    if (!ok) {
        LLAMA_LOG_WARN("%s: invalid thermal profile; accelerator admission is closed\n", __func__);
    } else if (policy_.hot_plugged() && !hot_plugged_announced_) {
        LLAMA_LOG_WARN("%s: hot-plugged profile is CPU-only with no cool mode\n", __func__);
        hot_plugged_announced_ = true;
    } else if (!policy_.hot_plugged()) {
        hot_plugged_announced_ = false;
    }
    return ok;
}

void llama_governor::record_telemetry(const llama_governor_telemetry_sample & sample) {
    telemetry_ = { sample.cpu_us, sample.read_bytes, sample.read_us, sample.stall_us, sample.management_us };
    if (prefill_tally_.active()) {
        record_tally();
    }
}

void llama_governor::stall_enter() {
    stall_union_.enter(static_cast<uint64_t>(lm_ggml_time_us()));
}

void llama_governor::stall_exit() {
    stall_union_.exit(static_cast<uint64_t>(lm_ggml_time_us()));
}

bool llama_governor::note_expert_route(bool resident, float resident_score,
                                       float flash_winner_score, float score_range) {
    if (!llama_governor_expert_substitution_would_displace(
            stats_.expert_substitution_lambda, resident, resident_score,
            flash_winner_score, score_range)) {
        return false;
    }
    ++stats_.expert_substitution_would_displace;
    return true;
}

llama_context * llama_governor::context() const { return last_ctx; }
llama_context * llama_governor::prefill_context() const { return ctx_prefill; }
llama_context * llama_governor::decode_context() const { return ctx_decode; }

llama_governor * llama_governor_init(llama_model * model_prefill, llama_model * model_decode,
                                     llama_context_params params_prefill, llama_context_params params_decode) {
    try {
        return new llama_governor(model_prefill, model_decode, params_prefill, params_decode);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return nullptr;
    }
}

llama_governor * llama_governor_init_with_params(llama_model * model_prefill, llama_model * model_decode,
                                                 llama_context_params params_prefill,
                                                 llama_context_params params_decode,
                                                 llama_governor_params governor_params) {
    return llama_governor_init_with_params_internal(
        model_prefill, model_decode, params_prefill, params_decode, governor_params, nullptr);
}

llama_governor * llama_governor_init_with_params_internal(
        llama_model * model_prefill, llama_model * model_decode,
        llama_context_params params_prefill, llama_context_params params_decode,
        llama_governor_params governor_params, std::string * failure_reason) {
    try {
        return new llama_governor(model_prefill, model_decode, params_prefill, params_decode,
                                  governor_params, true);
    } catch (const std::exception & err) {
        if (failure_reason != nullptr) {
            *failure_reason = err.what();
        } else {
            LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        }
        return nullptr;
    }
}

void llama_governor_free(llama_governor * governor) { delete governor; }

int32_t llama_governor_decode(llama_governor * governor, llama_batch batch) {
    return governor ? governor->decode(batch) : -1;
}

bool llama_governor_set_thermo_profile(llama_governor * governor,
                                       llama_governor_thermo_profile profile, int64_t now_ms) {
    return governor && governor->set_thermo_profile(profile, now_ms);
}

void llama_governor_record_telemetry(llama_governor * governor, llama_governor_telemetry_sample sample) {
    if (governor) {
        governor->record_telemetry(sample);
    }
}

void llama_governor_stall_enter(llama_governor * governor) {
    if (governor) {
        governor->stall_enter();
    }
}

void llama_governor_stall_exit(llama_governor * governor) {
    if (governor) {
        governor->stall_exit();
    }
}

bool llama_governor_note_expert_route(llama_governor * governor, bool resident,
                                      float resident_score, float flash_winner_score, float score_range) {
    return governor && governor->note_expert_route(resident, resident_score, flash_winner_score, score_range);
}

llama_context * llama_governor_get_context(const llama_governor * governor) {
    return governor ? governor->context() : nullptr;
}

llama_context * llama_governor_get_prefill_context(const llama_governor * governor) {
    return governor ? governor->prefill_context() : nullptr;
}

llama_context * llama_governor_get_decode_context(const llama_governor * governor) {
    return governor ? governor->decode_context() : nullptr;
}

void llama_governor_get_stats(const llama_governor * governor, llama_governor_stats * stats) {
    if (stats) {
        *stats = governor ? governor->stats() : llama_governor_stats{};
    }
}

void llama_governor_clear_cache(llama_governor * governor, bool clear_data) {
    if (governor != nullptr) {
        governor->clear_cache(clear_data);
    }
}

void llama_governor_reset_prefill_stats(llama_governor * governor) {
    if (governor != nullptr) {
        governor->reset_prefill_stats();
    }
}
