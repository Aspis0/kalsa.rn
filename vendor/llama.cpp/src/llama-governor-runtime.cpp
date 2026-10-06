#include "llama-governor.h"

#include "llama-context.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

const char * thermal_state_name(llama_governor_thermal_state state) {
    switch (state) {
        case llama_governor_thermal_state::Unknown:  return "Unknown";
        case llama_governor_thermal_state::FAST:     return "FAST";
        case llama_governor_thermal_state::WARM:     return "WARM";
        case llama_governor_thermal_state::COOLMODE: return "COOLMODE";
        case llama_governor_thermal_state::CRITICAL: return "CRITICAL";
        case llama_governor_thermal_state::LOWBAT:   return "LOWBAT";
        case llama_governor_thermal_state::Invalid:  return "Invalid";
    }
    return "?";
}

// One definition: this line is parsed as an API (prefix and existing keys
// must stay stable; new keys may only be appended).
void log_thermal_pause(const char * func, float now_c,
                       const llama_governor_thermal_snapshot & snap) {
    LLAMA_LOG_WARN("%s: no safe prefill chunk under the thermal ceiling; pausing "
                   "(now_c=%.1f state=%s platform_status=%d state_source=%s)\n",
                   func, now_c, thermal_state_name(snap.state), snap.platform_status,
                   snap.from_platform ? "platform" : "battery");
}

} // namespace

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
    // One snapshot: the stat triple agrees with the pause-log triple.
    const auto thermal = policy_.thermal_snapshot();
    stats_.thermal_state = thermal.state;
    stats_.platform_thermal_status = thermal.platform_status;
    stats_.state_source = thermal.from_platform ? "platform" : "battery";
    stats_.npu_fit = policy_.npu_fit();
    stats_.prefill_token_cap = policy_.prefill_token_cap();
    stats_.cpu_to_gpu_engagements = policy_.cpu_to_gpu_engagements();
    stats_.cache_budget_warning = policy_.cache_budget_warning();
    stats_.last_router_rule = policy_.prefill_rule();
}

int32_t llama_governor::admit_prefill(llama_batch batch, bool allow_chunking) {
    // Every executed piece must fit one llama_decode, which asserts
    // n_tokens_all <= cparams.n_batch; the input batch itself may be larger
    // (partitioning is this function's job). Any present context can execute
    // the pieces (CPU vs GPU route), so cap to the smallest n_batch of the
    // present contexts (both governor forms).
    uint32_t n_batch = UINT32_MAX; // test-only governor without contexts
    {
        llama_context * side_ctxs[3];
        side_state * side_states[3];
        const size_t n_sides = collect_sides(side_ctxs, side_states);
        for (size_t i = 0; i < n_sides; ++i) {
            n_batch = std::min(n_batch, side_ctxs[i]->get_cparams().n_batch);
        }
    }
    llama_governor_prefill_mode mode_used = llama_governor_prefill_mode::Auto;
    bool override_decided = false;
    const auto requested = policy_.prefill_engine(&mode_used, &override_decided);
    const float now_c = policy_.current_temperature_c();
    const auto admission = policy_.admit_prefill(
            requested, static_cast<uint32_t>(batch.n_tokens), now_c, n_batch);
    stats_.last_router_rule = admission.rule;
    if (admission.decision == llama_governor_decision::Abort) {
        return fail("prefill admission aborted");
    }
    if (admission.decision == llama_governor_decision::Wait) {
        // No chunk fits under the thermal ceiling. Reducing tokens is the only
        // allowed fallback; never switch to another engine. Pause this turn.
        log_thermal_pause(__func__, now_c, policy_.thermal_snapshot());
        return -2;
    }
    if (prefill_route_ == prefill_route::Undecided) {
        // The engine is decided once per latch: engine-changing inputs
        // (the override, LOWBAT, gpu fit) are read on every admission but
        // applied to the route only while it is Undecided, so a LOWBAT
        // update between two same-phase prefills keeps the latched engine -
        // switching mid-prompt would split the prompt's KV across the two
        // contexts. Per-batch safety (abort, Wait, floor, cap) is
        // admit_prefill's job and runs on every call.
        // One latch, one mode: the latch re-arms at the next prefill entry
        // from another phase, at clear_cache, or at a stats reset.
        // engine equals requested; a non-CPU admission (GPU_COOLMODE
        // included) keeps the GPU route - owner rule: the CPU heats more,
        // do not take the GPU away.
        prefill_route_ = admission.engine == llama_governor_engine::CPU ? prefill_route::CPU : prefill_route::GPU;
        stats_.prefill_engine = admission.engine;
        // Snapshot the override inputs with the route: every fact of this
        // latch reports the same mode and the same causal decision.
        turn_prefill_mode_ = mode_used;
        turn_override_decided_ = override_decided;
    }
    if (admission.decision == llama_governor_decision::CPUFallback) {
        // Whole admission of a non-CPU request on its requested engine; the
        // enum name is a historical misnomer - it does not fall back to CPU.
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
                requested, static_cast<uint32_t>(remaining),
                policy_.current_temperature_c(), n_batch);
        if (next.decision == llama_governor_decision::Wait) {
            // Defensive: under the threading contract the profile cannot change
            // inside one decode, and an admission that reached the planner
            // (Chunk) implies below k_warn_c, so no remainder can ever Wait
            // here (no deterministic test exists for that reason). If one
            // somehow does, pause like the first admission: -2 with the
            // thermal warn and rule-0 stamp, before the tokens==0 guard below
            // would swallow the Wait into a silent, mislabelled -2.
            stats_.last_router_rule = next.rule;
            log_thermal_pause(__func__, policy_.current_temperature_c(), policy_.thermal_snapshot());
            return -2;
        }
        if (next.decision == llama_governor_decision::Abort || next.tokens == 0) {
            LLAMA_LOG_INFO("%s: prompt cannot be partitioned into safe tabled chunks\n", __func__);
            return -2;
        }
        const int32_t count = std::min<int32_t>(next.tokens, remaining);
        if (count <= 1) {
            // Unreachable while the policy never returns a chunk that leaves a
            // 1-token remainder; keep it loud if that invariant ever breaks.
            LLAMA_LOG_INFO("%s: prompt cannot be partitioned into runnable chunks\n", __func__);
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
    const auto selection = policy_.select_decode(ggml_time_us() / 1000, decode_tokens_since_prefill_);
    // The engine routes this batch: NPU decode runs on ctx_prefill (decode_impl).
    decode_engine_ = selection.engine;
    decode_hop_rule_ = selection.hop_rule;
    stats_.decode_engine = selection.engine;
    stats_.decode_requires_reload = selection.requires_reload;
    stats_.last_router_rule = selection.rule;
    stats_.cpu_to_gpu_engagements = policy_.cpu_to_gpu_engagements();
    // hop_rule is set only on the token that decides a window, so this counts
    // windows, in the same clear_cache interval as the other decode_hop_* stats.
    if (selection.hop_rule != nullptr) {
        stats_.decode_hop_rule = selection.hop_rule;
        if (std::strcmp(selection.hop_rule, "headroom") == 0) {
            ++stats_.decode_hop_headroom_windows;
        }
    }
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
    result.stall_union_us = stall_union_.total(static_cast<uint64_t>(ggml_time_us()));
    return result;
}

void llama_governor::clear_cache(bool clear_data) {
    {
        llama_context * side_ctxs[3];
        side_state * side_states[3];
        const size_t n_sides = collect_sides(side_ctxs, side_states);
        for (size_t i = 0; i < n_sides; ++i) {
            llama_memory_clear(llama_get_memory(side_ctxs[i]), clear_data);
            *side_states[i] = side_state{};
        }
    }
    last_ctx = nullptr;
    last_phase = phase::None;
    prefill_route_ = prefill_route::Undecided;
    decode_tokens_since_prefill_ = 0;
    forced_window_ = UINT32_MAX;
    policy_.reset_decode_hop();
    // a fully cached turn's 1-token batch never enters the prefill branch,
    // so the floor refresh lives here too (one-model only)
    if (!legs.empty()) {
        refresh_decode_legs();
    }

    stats_.commit_bytes = 0;
    stats_.commit_us = 0;
    stats_.commit_count = 0;
    stats_.commit_naive_count = 0;
    stats_.commit_staged_count = 0;
    stats_.commit_transfers_k = 0;
    stats_.commit_transfers_naive = 0;
    stats_.commit_transfers_staged = 0;
    stats_.decode_hops = 0;
    stats_.decode_tokens_cpu = 0;
    stats_.decode_tokens_npu = 0;
    stats_.decode_tokens_gpu = 0;
    stats_.decode_us_cpu = 0;
    stats_.decode_us_npu = 0;
    stats_.decode_us_gpu = 0;
    for (auto & pair : stats_.decode_hop_pairs) {
        pair = llama_governor_stats::llama_governor_hop_pair{};
    }
    stats_.decode_hop_commit_bytes = 0;
    stats_.decode_hop_commit_us = 0;
    stats_.decode_hop_headroom_windows = 0;
    stats_.decode_hop_rule = nullptr;
    // the floor evidence is measured against these marks; with the counters
    // zeroed the marks must go too or the next delta underflows
    for (int i = 0; i < 3; ++i) {
        floor_mark_tokens_[i] = 0;
        floor_mark_us_[i] = 0;
    }
}

void llama_governor::reset_prefill_stats() {
    prefill_route_ = prefill_route::Undecided;
    // The hop clock is per-turn routing state like the latch: the binding
    // calls this at every completion start, so a turn whose prompt was fully
    // cached (its 1-token batch never enters the prefill branch) still
    // decides hop window 0 fresh, whichever rule runs.
    decode_tokens_since_prefill_ = 0;
    forced_window_ = UINT32_MAX;
    policy_.reset_decode_hop();
    // a fully cached turn's 1-token batch never enters the prefill branch,
    // so the floor refresh lives here too (one-model only)
    if (!legs.empty()) {
        refresh_decode_legs();
    }
    stats_.prefill_us = 0;
    stats_.prefill_n = 0;
    stats_.prefill_chunks[0] = '\0';
    stats_.route_chunk_count = 0;
    stats_.route_chunks_truncated = false;
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

bool llama_governor::set_decode_headroom(float cpu_headroom_c, float npu_headroom_c,
                                          float gpu_headroom_c) {
    if (!policy_enabled_) {
        return false;
    }
    policy_.set_decode_headroom(cpu_headroom_c, npu_headroom_c, gpu_headroom_c);
    return true;
}

bool llama_governor::set_decode_legs(bool npu, bool gpu, bool cpu) {
    if (!policy_enabled_) {
        return false;
    }
    policy_.set_decode_legs(npu, gpu, cpu);
    return true;
}

bool llama_governor::set_prefill_override(int mode) {
    if (!policy_enabled_) {
        return false;
    }
    // Atomic mode store only. This method may overlap decode(), so it must
    // not touch stats_ or any other state decode writes - do not re-add the
    // stats refresh here. The mode is consumed at the next prefill latch;
    // decode_impl refreshes stats on the decode path in the meantime.
    return policy_.set_prefill_override(mode);
}

void llama_governor::record_telemetry(const llama_governor_telemetry_sample & sample) {
    telemetry_ = { sample.cpu_us, sample.read_bytes, sample.read_us, sample.stall_us, sample.management_us };
    if (prefill_tally_.active()) {
        record_tally();
    }
}

void llama_governor::stall_enter() {
    stall_union_.enter(static_cast<uint64_t>(ggml_time_us()));
}

void llama_governor::stall_exit() {
    stall_union_.exit(static_cast<uint64_t>(ggml_time_us()));
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
// two-model: the two named contexts; one-model: the accelerator-role leg
// (NPU, else GPU, else CPU) and the CPU leg
llama_context * llama_governor::prefill_context() const {
    return legs.empty() ? ctx_prefill : prefill_ctx(false);
}
llama_context * llama_governor::decode_context() const {
    return legs.empty() ? ctx_decode : leg_ctx(llama_governor_engine::CPU);
}

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

llama_governor * llama_governor_init_one_model_with_params(
        llama_model * model, const llama_governor_leg * legs, uint32_t n_legs,
        llama_governor_params governor_params) {
    try {
        return new llama_governor(model, legs, n_legs, governor_params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return nullptr;
    }
}

llama_context * llama_governor_leg_ctx(const llama_governor * governor, llama_governor_engine engine) {
    return governor ? governor->leg_ctx(engine) : nullptr;
}

void llama_governor_free(llama_governor * governor) { delete governor; }

int32_t llama_governor_decode(llama_governor * governor, llama_batch batch) {
    return governor ? governor->decode(batch) : -1;
}

bool llama_governor_trim_sequence(llama_governor * governor, llama_pos p) {
    return governor ? governor->trim_sequence(p) : false;
}

bool llama_governor_set_thermo_profile(llama_governor * governor,
                                       llama_governor_thermo_profile profile, int64_t now_ms) {
    return governor && governor->set_thermo_profile(profile, now_ms);
}

bool llama_governor_set_decode_headroom(llama_governor * governor,
                                        float cpu_headroom_c, float npu_headroom_c,
                                        float gpu_headroom_c) {
    return governor && governor->set_decode_headroom(cpu_headroom_c, npu_headroom_c, gpu_headroom_c);
}

bool llama_governor_set_decode_legs(llama_governor * governor, bool npu, bool gpu, bool cpu) {
    return governor && governor->set_decode_legs(npu, gpu, cpu);
}

bool llama_governor_set_prefill_override(llama_governor * governor, int mode) {
    return governor && governor->set_prefill_override(mode);
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
