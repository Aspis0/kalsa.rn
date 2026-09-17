#include "rn-completion.h"
#include "rn-llama.h"
#include "rn-tts.h"
#include "rn-mtmd.hpp"
#include "rn-common.hpp"
#include "llama-ext.h"  // llama_get_ctx_other (mem-shared MTP draft detection)

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <limits>

// Include multimodal support
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include "tools/mtmd/clip.h"

namespace rnllama {

static bool benchRawTopProbs(
    llama_context* ctx, const llama_vocab* vocab, int32_t count,
    std::vector<completion_token_output::raw_token_prob>& result) {
    result.clear();
    if (count <= 0 || ctx == nullptr || vocab == nullptr) return false;
    llama_synchronize(ctx);
    const float* logits = llama_get_logits_ith(ctx, -1);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (logits == nullptr || n_vocab <= 0) return false;

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    if (!std::isfinite(max_logit)) return false;

    double exp_sum = 0.0;
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (int32_t i = 0; i < n_vocab; ++i) {
        const float logit = logits[i];
        exp_sum += std::exp((double)logit - max_logit);
        candidates.push_back({(llama_token)i, logit, 0.0f});
    }
    if (!(exp_sum > 0.0)) return false;
    const size_t top_count = std::min<size_t>(count, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + top_count, candidates.end(),
        [](const llama_token_data& a, const llama_token_data& b) { return a.logit > b.logit; });
    const double log_norm = (double)max_logit + std::log(exp_sum);
    result.reserve(top_count);
    for (size_t i = 0; i < top_count; ++i) {
        result.push_back({candidates[i].id, (float)((double)candidates[i].logit - log_norm)});
    }
    return true;
}

// Constructor
llama_rn_context_completion::llama_rn_context_completion(llama_rn_context* parent)
    : parent_ctx(parent) {
}

// Destructor
llama_rn_context_completion::~llama_rn_context_completion() {
    resetSpeculative();
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }
}

void llama_rn_context_completion::rewind() {
    resetSpeculative();
    mtp_capability_logged = false;
    is_interrupted = false;
    parent_ctx->params.antiprompt.clear();
    parent_ctx->params.sampling.grammar = {};
    parent_ctx->params.sampling.grammar_lazy = false;
    parent_ctx->params.sampling.grammar_triggers.clear();
    parent_ctx->params.sampling.preserved_tokens.clear();
    parent_ctx->params.sampling.generation_prompt.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    generated_token_ids.clear();
    num_draft_tokens = 0;
    num_draft_tokens_accepted = 0;
    resetGenerationTimings();
    prefill_text = "";
    generated_text = "";
    generated_text.reserve(parent_ctx->params.n_ctx);
    utf8_gate.reset();
    truncated = false;
    context_full = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    incomplete = false;
    n_remain = 0;
    n_past = 0;
    parent_ctx->params.sampling.n_prev = parent_ctx->n_ctx;
    if (parent_ctx->isVocoderEnabled()) {
        parent_ctx->tts_wrapper->audio_tokens.clear();
        parent_ctx->tts_wrapper->next_token_uses_guide_token = true;
        parent_ctx->tts_wrapper->guide_tokens.clear();
    }
}

bool llama_rn_context_completion::initSampling() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
    }
    ctx_sampling = common_sampler_init(parent_ctx->model, parent_ctx->params.sampling);
    return ctx_sampling != nullptr;
}

// ----------------------------------------------------------------------------
// Prompt state cache (recurrent / hybrid / SWA prefix reuse)
//
// These memories can't roll back in place, so instead of full-clearing on
// divergence we snapshot the non-rollbackable part (PARTIAL_ONLY) during prompt
// ingest and restore the longest snapshot that prefixes the new prompt.
// Single-sequence completion path only; the slot manager passes no callbacks
// and keeps the full-clear behaviour.
// ----------------------------------------------------------------------------

void llama_rn_context_completion::probeStateCache() {
    if (state_cache_probed) {
        return;
    }
    state_cache_probed = true;

    // Pull the host-configured bounds (set at model init) into effect.
    state_cache_budget_bytes = parent_ctx->state_cache_budget_bytes;
    // 0 = no count cap (budget-bound only); <0/unset keeps the default.
    if (parent_ctx->state_cache_max_checkpoints > 0) {
        state_cache_max_checkpoints = (size_t) parent_ctx->state_cache_max_checkpoints;
    } else if (parent_ctx->state_cache_max_checkpoints == 0) {
        state_cache_max_checkpoints = std::numeric_limits<size_t>::max();
    }

    const llama_model *model = parent_ctx->model;
    if (model == nullptr || state_cache_budget_bytes == 0) {
        // A zero budget is an explicit opt-out (keep the full-reprocess fallback).
        state_cache_enabled = false;
        return;
    }
    // Recurrent/hybrid only. Pure-SWA never fails seq_rm in this fork, so a
    // checkpoint could never be restored — capturing would be pure cost.
    // TODO: SWA reuse past a slid window (deep edit in a chat longer than the
    // window) is unguarded here; a fix would need a pos_min trigger, not seq_rm.
    state_cache_enabled =
        llama_model_is_recurrent(model) ||
        llama_model_is_hybrid(model);
    if (state_cache_enabled) {
        LOG_INFO("prompt state cache enabled (recurrent/hybrid model)");
    }
}

void llama_rn_context_completion::evictStateCheckpoints() {
    // Oldest first, but always keep the smallest-position snapshot: that is the
    // first message boundary (system-prompt end) a brand-new session shares.
    auto total_bytes = [&]() {
        size_t n = 0;
        for (const auto &c : state_checkpoints) n += c.size_bytes();
        return n;
    };
    auto smallest_pos = [&]() {
        size_t idx = 0;
        for (size_t i = 1; i < state_checkpoints.size(); i++) {
            if (state_checkpoints[i].n_tokens() < state_checkpoints[idx].n_tokens()) idx = i;
        }
        return idx;
    };
    while (state_checkpoints.size() > 1 &&
           (state_checkpoints.size() > state_cache_max_checkpoints ||
            total_bytes() > state_cache_budget_bytes)) {
        const size_t keep = smallest_pos();
        // Evict the oldest snapshot that is not the pinned stable-prefix one.
        size_t victim = (keep == 0 && state_checkpoints.size() > 1) ? 1 : 0;
        state_checkpoints.erase(state_checkpoints.begin() + victim);
    }
}

void llama_rn_context_completion::clearStateCheckpoints() {
    state_checkpoints.clear();
    // Boundary positions index into the current prompt; invalidated together.
    boundary_ckpts.clear();
    prompt_checkpoint_pending = false;
}

void llama_rn_context_completion::keepOnlyStablePrefixCheckpoint() {
    // Boundary positions index into the prompt that is going away.
    boundary_ckpts.clear();
    prompt_checkpoint_pending = false;
    if (state_checkpoints.size() <= 1) {
        return;
    }
    size_t keep = 0;
    for (size_t i = 1; i < state_checkpoints.size(); i++) {
        if (state_checkpoints[i].n_tokens() < state_checkpoints[keep].n_tokens()) {
            keep = i;
        }
    }
    rn_state_checkpoint stable = std::move(state_checkpoints[keep]);
    state_checkpoints.clear();
    state_checkpoints.push_back(std::move(stable));
}

void llama_rn_context_completion::eraseStateCheckpointAt(size_t n_tokens) {
    state_checkpoints.erase(
        std::remove_if(state_checkpoints.begin(), state_checkpoints.end(),
            [&](const rn_state_checkpoint &c) { return c.n_tokens() == n_tokens; }),
        state_checkpoints.end());
}

void llama_rn_context_completion::eraseStateCheckpointsAfter(size_t n_tokens) {
    const size_t old_size = state_checkpoints.size();
    state_checkpoints.erase(
        std::remove_if(state_checkpoints.begin(), state_checkpoints.end(),
            [&](const rn_state_checkpoint &c) { return c.n_tokens() > n_tokens; }),
        state_checkpoints.end());
    if (state_checkpoints.size() != old_size) {
        LOG_VERBOSE("invalidated %zu state checkpoint(s) after position %zu",
            old_size - state_checkpoints.size(), n_tokens);
    }
}

void llama_rn_context_completion::captureStateCheckpoint() {
    // The memory holds exactly embd[0, n_past).
    if (n_past <= 0) {
        return;
    }
    captureStateCheckpoint(embd, (size_t) n_past);
}

void llama_rn_context_completion::captureStateCheckpoint(
        const std::vector<llama_token> &seq, size_t n) {
    if (!state_cache_enabled || !state_cache_capture_allowed || parent_ctx->active_ctx() == nullptr) {
        return;
    }
    if (n == 0 || n > seq.size()) {
        return;
    }
    // Label n must be the live frontier. Capturing seq[0, n) while pos_max+1 > n
    // stores the long recurrent tail under a short name; recover then seq_rm(k)
    // fails on hybrid (beyond n_rs_seq) and loadPrompt used to ignore that and
    // decode at the old pos_max+1 (S23: restore 2441, recover 1827, save
    // n_tokens=2360 pos_max=2973).
    auto * cap_kv = llama_get_memory(parent_ctx->active_ctx());
    if (cap_kv != nullptr) {
        const llama_pos cap_pos_max = llama_memory_seq_pos_max(cap_kv, 0);
        if (cap_pos_max + 1 != (llama_pos) n) {
            rnllama::log("WARNING", __func__, __LINE__,
                "KALSA_KVRESUME skip_ckpt n=%zu pos_max=%d", n, (int) cap_pos_max);
            return;
        }
    }
    // Already hold this exact snapshot (e.g. just restored): skip the readback.
    for (const auto &c : state_checkpoints) {
        if (c.n_tokens() == n &&
            std::equal(c.tokens.begin(), c.tokens.end(), seq.begin())) {
            return;
        }
    }

    const size_t size = llama_state_seq_get_size_ext(
        parent_ctx->active_ctx(), /*seq_id*/ 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (size == 0) {
        return;
    }

    // A hybrid's SWA cells grow ~linearly until the window fills (unlike its
    // fixed-size recurrent cells). Extrapolate to saturation and disable if one
    // snapshot would exceed half the budget. (Pure-SWA is gated off upstream.)
    if (parent_ctx->model != nullptr && llama_model_n_swa(parent_ctx->model) > 0) {
        const size_t n_swa = (size_t) llama_model_n_swa(parent_ctx->model);
        const size_t fill  = std::min(n, n_swa);
        const size_t saturated = size * n_swa / fill;
        if (saturated > state_cache_budget_bytes / 2) {
            LOG_WARNING(
                "state cache disabled: a saturated snapshot would be ~%.1f MiB "
                "(measured %.1f MiB at %zu/%zu window fill) vs a %.1f MiB budget",
                saturated / (1024.0 * 1024.0), size / (1024.0 * 1024.0),
                fill, n_swa, state_cache_budget_bytes / (1024.0 * 1024.0));
            state_cache_enabled = false;
            clearStateCheckpoints();
            return;
        }
    }

    rn_state_checkpoint ckpt;
    ckpt.tokens.assign(seq.begin(), seq.begin() + n);
    try {
        ckpt.data.resize(size);
    } catch (const std::bad_alloc &) {
        // Skip the capture rather than aborting the completion.
        LOG_WARNING("state checkpoint alloc failed (n_tokens=%zu, %.1f MiB)",
            n, size / (1024.0 * 1024.0));
        return;
    }
    const size_t written = llama_state_seq_get_data_ext(
        parent_ctx->active_ctx(), ckpt.data.data(), size, /*seq_id*/ 0,
        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (written == 0) {
        // Can fail under memory pressure; the old snapshot at this boundary
        // is still valid and must survive a failed re-capture.
        LOG_WARNING("state checkpoint capture failed (n_tokens=%zu)", n);
        return;
    }
    ckpt.data.resize(written);

    // Replace any snapshot at this boundary only after a successful capture
    // (a stale same-length one would shadow the current tokens).
    eraseStateCheckpointAt(n);
    state_checkpoints.push_back(std::move(ckpt));
    evictStateCheckpoints();
    LOG_VERBOSE("captured state checkpoint: n_tokens=%zu, size=%.1f KiB, total=%zu",
        n, written / 1024.0, state_checkpoints.size());
}

int llama_rn_context_completion::findStateCheckpoint(
        const std::vector<llama_token> &target, size_t max_len) const {
    // Pick the longest snapshot whose tokens are a prefix of `target` and whose
    // length does not exceed `max_len` (the verified shared-prefix length).
    int best = -1;
    size_t best_len = 0;
    for (size_t i = 0; i < state_checkpoints.size(); i++) {
        const auto &c = state_checkpoints[i];
        const size_t n = c.n_tokens();
        if (n == 0 || n > max_len || n > target.size()) {
            continue;
        }
        if (n <= best_len) {
            continue; // can't beat the current best
        }
        if (std::equal(c.tokens.begin(), c.tokens.end(), target.begin())) {
            best = (int) i;
            best_len = n;
        }
    }
    return best;
}

bool llama_rn_context_completion::recoverStateCheckpoint(
        const std::vector<llama_token> &target, size_t max_reuse,
        size_t total_tokens, llama_pos &n_past_out) {
    if (parent_ctx->active_ctx() == nullptr) {
        return false;
    }
    auto * kv = llama_get_memory(parent_ctx->active_ctx());
    // Longest snapshot with n_tokens <= n_common (max_reuse). min(n_common,
    // total-1) rejected a restore checkpoint of length n_common when
    // n_common == total (exact match; one token reserved to eval).
    // A short label on a long KV fails hybrid seq_rm; do not keep n_past=k.
    size_t search_max = max_reuse;
    while (true) {
        const int ckpt_idx = findStateCheckpoint(target, search_max);
        if (ckpt_idx < 0 || !restoreStateCheckpoint((size_t) ckpt_idx)) {
            return false;
        }
        llama_pos k = (llama_pos) state_checkpoints[ckpt_idx].n_tokens();
        const bool seq_rm_ok = llama_memory_seq_rm(kv, 0, k, -1);
        const llama_pos pos_max = llama_memory_seq_pos_max(kv, 0);
        if (seq_rm_ok && pos_max + 1 == k) {
            if (total_tokens > 0 && (size_t) k >= total_tokens) {
                k = (llama_pos) total_tokens - 1;
            }
            n_past_out = k;
            return true;
        }
        rnllama::log("WARNING", __func__, __LINE__,
            "KALSA_KVRESUME recover_trim_failed k=%d seq_rm_ok=%d pos_max=%d",
            (int) k, (int) seq_rm_ok, (int) pos_max);
        eraseStateCheckpointAt((size_t) k);
        if (k <= 0) {
            return false;
        }
        search_max = (size_t) k - 1;
    }
}

bool llama_rn_context_completion::restoreStateCheckpoint(size_t index) {
    if (index >= state_checkpoints.size() || parent_ctx->active_ctx() == nullptr) {
        return false;
    }
    const auto &c = state_checkpoints[index];
    const size_t read = llama_state_seq_set_data_ext(
        parent_ctx->active_ctx(), c.data.data(), c.data.size(), /*dest_seq_id*/ 0,
        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (read == 0) {
        LOG_WARNING("state checkpoint restore failed (n_tokens=%zu)", c.n_tokens());
        return false;
    }
    return true;
}

void llama_rn_context_completion::truncatePrompt(std::vector<llama_token> &prompt_tokens) {
    const int n_left = parent_ctx->n_ctx - parent_ctx->params.n_keep;
    const int n_block_size = n_left / 2;
    const int erased_blocks = (prompt_tokens.size() - parent_ctx->params.n_keep - n_block_size) / n_block_size;

    // Keep n_keep tokens at start of prompt (at most n_ctx - 4)
    std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + parent_ctx->params.n_keep);

    new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + parent_ctx->params.n_keep + erased_blocks * n_block_size, prompt_tokens.end());

    LOG_INFO("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, old_size: %d, new_size: %d",
        parent_ctx->n_ctx,
        parent_ctx->params.n_keep,
        n_left,
        prompt_tokens.size(),
        new_tokens.size()
    );

    truncated = true;
    prompt_tokens = new_tokens;
}

std::vector<llama_pos> llama_rn_context_completion::computeMessageBoundaries(
        const std::vector<llama_token> &tokens, llama_pos min_gap) const {
    std::vector<llama_pos> out;
    if (parent_ctx->model == nullptr) {
        return out;
    }
    const auto vocab = llama_model_get_vocab(parent_ctx->model);
    // A boundary is the first content token after a run of chat-template
    // delimiter tokens (CONTROL/USER_DEFINED, template-agnostic; whitespace
    // keeps a run open). History up to a boundary is identical in every future
    // prompt sharing the conversation to that message — an exact restore point.
    //
    // min_gap: a snapshot < min_gap tokens past the previous restore point
    // (position 0 counts as the first) saves less reprocess than the slot is
    // worth. min_gap == 1 keeps every boundary (used to find the last one).
    llama_pos last_accepted = 0;
    bool run_has_delim = false;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] < 0) {
            run_has_delim = false; // media placeholder — not a vocab id
            continue;
        }
        const unsigned attr = (unsigned) llama_vocab_get_attr(vocab, tokens[i]);
        if ((attr & ((unsigned) LLAMA_TOKEN_ATTR_CONTROL |
                     (unsigned) LLAMA_TOKEN_ATTR_USER_DEFINED)) != 0) {
            run_has_delim = true;
            continue;
        }
        if (!run_has_delim) {
            continue;
        }
        // Whitespace between delimiters keeps the run open.
        const std::string piece = common_token_to_piece(parent_ctx->active_ctx(), tokens[i]);
        if (!piece.empty() && piece.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        const llama_pos pos = (llama_pos) i;
        if (pos - last_accepted >= min_gap) {
            out.push_back(pos);
            last_accepted = pos;
        }
        run_has_delim = false;
    }
    return out;
}

void llama_rn_context_completion::loadPrompt(const std::vector<std::string> &media_paths, bool allow_state_cache) {
    bool has_media = !media_paths.empty();
    // embedding()/rerank() drive throwaway prompts through this same path; keep their
    // state out of the chat's checkpoint cache (see state_cache_capture_allowed).
    state_cache_capture_allowed = allow_state_cache;

    // Check if this is an encoder-decoder model (like T5)
    const bool is_enc_dec = llama_model_has_encoder(parent_ctx->model);
    const auto vocab = llama_model_get_vocab(parent_ctx->model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    if (!has_media) {
        std::vector<llama_token> text_tokens;
        // Text-only path - use modified tokenization for encoder-decoder models
        text_tokens = ::common_tokenize(parent_ctx->active_ctx(), parent_ctx->params.prompt, add_bos || is_enc_dec, true);
        num_prompt_tokens = text_tokens.size();

        // Upstream dumps every prompt token id here, and rnllama::log reaches
        // logcat ungated in release builds (only LOG_VERBOSE is compiled out),
        // so that line publishes the whole conversation. Counts carry the same
        // diagnostic value. Do not restore the per-token loop.
        LOG_INFO("%s: prompt_n=%zu n_ctx=%d embd=%zu",
            __func__, num_prompt_tokens, parent_ctx->n_ctx, embd.size());

        if (parent_ctx->params.n_keep < 0) {
            parent_ctx->params.n_keep = (int)num_prompt_tokens;
        }
        parent_ctx->params.n_keep = std::min(parent_ctx->n_ctx - 4, parent_ctx->params.n_keep);

        // Handle truncation if needed
        if (num_prompt_tokens >= (size_t)parent_ctx->n_ctx) {
            // truncatePrompt keeps n_keep then a later block — not a prefix
            // JS will resend. Same-chat live KV must stay. No K-shift.
            if (parent_ctx->params.ctx_shift) {
                rnllama::log("WARNING", __func__, __LINE__,
                    "KALSA_KVSHIFT refused prompt=%zu n_ctx=%d keep_prefix=1",
                    num_prompt_tokens, parent_ctx->n_ctx);
            }
            context_full = true;
            return;
        }

        // NOTE: Do NOT feed prompt tokens into the sampler.
        // The penalty sampler should only track generated tokens.
        // Feeding prompt tokens causes <|im_end|> (which appears
        // in every ChatML prompt) to be penalised by repeat_penalty
        // / frequency_penalty, preventing EOS and producing
        // extremely verbose output on Qwen-family models.

        // n_common = shared prefix with the live cache; it bounds how far a
        // checkpoint may be trusted.
        size_t n_common = is_enc_dec ? 0 : find_common_prefix_length(embd, text_tokens);
        // Kalsa diag: always print prefix-compare inputs. Silence of KVDIAG0
        // (gated on n_common==0 && !embd.empty()) left open whether embd was
        // empty or comparison ran on a different completion object than restore.
        rnllama::log("WARNING", __func__, __LINE__,
            "KALSA_KVPREFIX embd=%zu text_tokens=%zu n_common=%zu mtp_draft_mem_shared=%d is_enc_dec=%d this=%p",
            embd.size(), text_tokens.size(), n_common,
            (int) mtp_draft_mem_shared, (int) is_enc_dec, (const void *) this);
        // Real divergence only: shared prefix is a proper subset of both sides.
        // Counts are logged in every build. The token ids and the detokenized
        // snippets are the user's own words, and WARNING-level logcat is collected
        // by Android bug reports, so both are compiled in for debug builds only:
        // RNLLAMA_ANDROID_ENABLE_LOGGING is set by CMake when CMAKE_BUILD_TYPE is
        // Debug (rnllama/CMakeLists.txt). NDEBUG cannot serve here — that target
        // sets it unconditionally. Token ids are NOT exempt, which this code used
        // to assume: detokenizing them recovers the words exactly.
        if (n_common < embd.size() && n_common < text_tokens.size()) {
            const size_t pre = 8;
            const size_t post = 12;
            const size_t shared_lo = n_common > pre ? n_common - pre : 0;
            const size_t embd_hi = std::min(n_common + post, embd.size());
            const size_t text_hi = std::min(n_common + post, text_tokens.size());
            rnllama::log("WARNING", __func__, __LINE__,
                "KALSA_KVDIVERGE n_common=%zu shared_lo=%zu embd_hi=%zu text_hi=%zu",
                n_common, shared_lo, embd_hi, text_hi);
#ifdef RNLLAMA_ANDROID_ENABLE_LOGGING
            std::string shared_ids, embd_ids, text_ids;
            for (size_t i = shared_lo; i < n_common; i++) {
                shared_ids += std::to_string(embd[i]) + " ";
            }
            for (size_t i = n_common; i < embd_hi; i++) {
                embd_ids += std::to_string(embd[i]) + " ";
            }
            for (size_t i = n_common; i < text_hi; i++) {
                text_ids += std::to_string(text_tokens[i]) + " ";
            }
            auto clip200 = [](std::string s) {
                if (s.size() > 200) s.resize(200);
                return s;
            };
            const std::string shared_txt = n_common > shared_lo
                ? clip200(tokens_to_str(parent_ctx->active_ctx(), embd.cbegin() + shared_lo, embd.cbegin() + n_common))
                : std::string();
            const std::string embd_txt = embd_hi > n_common
                ? clip200(tokens_to_str(parent_ctx->active_ctx(), embd.cbegin() + n_common, embd.cbegin() + embd_hi))
                : std::string();
            const std::string text_txt = text_hi > n_common
                ? clip200(tokens_to_str(parent_ctx->active_ctx(), text_tokens.cbegin() + n_common, text_tokens.cbegin() + text_hi))
                : std::string();
            rnllama::log("WARNING", __func__, __LINE__,
                "KALSA_KVDIVERGE ids shared=[%s] embd=[%s] text=[%s] shared_txt=[%s] embd_txt=[%s] text_txt=[%s]",
                shared_ids.c_str(), embd_ids.c_str(), text_ids.c_str(),
                shared_txt.c_str(), embd_txt.c_str(), text_txt.c_str());
#endif
        }
        // A mem-shared MTP draft leaves stale speculative cells in the target's
        // SHARED KV window that upstream never cleans (TAG_KV_CACHE_SHARE_CELLS,
        // see initMTP). Reusing any prefix — even the plain seq_rm fast path —
        // then inherits that polluted state and shifts the distribution. Force a
        // full reprocess for these drafts (the pre-reuse MTP behaviour); it is
        // detected on the first generation turn, so this engages from turn 2 on.
        if (mtp_draft_mem_shared) {
            n_common = 0;
        }
        // Kalsa diag: a zero common prefix while a cache EXISTS means the
        // re-rendered prompt diverges at position 0 — the case that silently
        // wastes a restored session. Print both heads so the divergence can be
        // named instead of guessed (mirrors the KVDIAG idea from the MoE work).
        if (n_common == 0 && !embd.empty() && !text_tokens.empty()) {
            std::string a, b;
            for (size_t i = 0; i < 12 && i < embd.size(); i++) {
                a += std::to_string(embd[i]) + " ";
            }
            for (size_t i = 0; i < 12 && i < text_tokens.size(); i++) {
                b += std::to_string(text_tokens[i]) + " ";
            }
            LOG_WARNING("KALSA_KVDIAG0 cache_len=%zu prompt_len=%zu cache_head=[%s] prompt_head=[%s]",
                embd.size(), text_tokens.size(), a.c_str(), b.c_str());
        }

        n_past = (llama_pos) n_common;

        embd = text_tokens;
        if (n_past == num_prompt_tokens) {
            // we have to evaluate at least 1 token to generate logits.
            n_past--;
        }

        // Manage KV cache
        probeStateCache();
        auto * kv = llama_get_memory(parent_ctx->active_ctx());
        if (mtp_draft_mem_shared) {
            llama_memory_clear(kv, false);  // drop the polluted shared window
        }
        bool cache_remove_success = llama_memory_seq_rm(kv, 0, n_past, -1);

        // Recurrent/hybrid/SWA: seq_rm fails beyond the rollback window; restore
        // the longest matching snapshot and reprocess only the diverged tail.
        if (!cache_remove_success) {
            // Steal vs ggml-org/llama.cpp #20955 / #25592: seq_rm fails by
            // design on hybrid. Restore nearest checkpoint with n_tokens <=
            // n_common (user-turn 37s). Extract n_common=2 vs ckpt=2301 is an
            // expected miss — then drop live KV, keep RAM snapshots.
            // mtp_draft_mem_shared still cleared the window above.
            if (recoverStateCheckpoint(text_tokens, n_common, num_prompt_tokens, n_past)) {
                // WARNING, not INFO: a filtered device logcat keeps W and drops
                // I, so the 2026-09-17 run could not tell a successful seq_rm
                // from a snapshot restore -- the two readings that decide
                // whether partial reuse works on this model at all. Counts
                // only; ids and text never leave (see app 4553062).
                rnllama::log("WARNING", __func__, __LINE__,
                    "KALSA_KVREUSE checkpoint n_past=%d prompt=%zu n_common=%zu",
                    (int) n_past, num_prompt_tokens, n_common);
            } else {
                std::string sizes;
                for (const auto &c : state_checkpoints) {
                    sizes += std::to_string(c.n_tokens()) + ",";
                }
                LOG_WARNING("KALSA_KVDIAG n_common=%zu total=%zu search_max=%zu checkpoints=[%s] keep_prefix=0",
                    n_common, (size_t) num_prompt_tokens,
                    n_common,
                    sizes.c_str());
                // Drop live KV so this prompt does not append on a hybrid tail
                // seq_rm could not trim. Keep RAM snapshots: extract's n_common=2
                // miss used to wipe the chat 1827/1829 blobs, then loadSession
                // relabeled the restored 2441 tail as those lengths.
                llama_memory_clear(kv, false);
                n_past = 0;
            }
        }

        // Frontier capture: the reused state already rests at n_past, so snapshot
        // it here — one readback, no decode split. tokens[0,n_past) is the verified
        // shared prefix, so it is token-exact. Restore point for a later
        // regenerate/edit of this turn.
        if (state_cache_enabled && state_cache_capture_allowed && n_past > 0) {
            captureStateCheckpoint(text_tokens, (size_t) n_past);
        }

        // Cold ingest only lays boundary snapshots as it decodes (amortized once
        // per session; seeds the system-prefix anchor). Warm turns don't split —
        // the frontier capture above covers them, so the tail decodes in one batch.
        const bool cold_ingest = n_past == 0;
        boundary_ckpts.clear();
        if (state_cache_enabled && state_cache_capture_allowed) {
            if (cold_ingest) {
                // Boundaries + the frontier (last boundary, even if min_gap would
                // drop it) so a fresh session's first turn has a restore point.
                // Delimiter-less templates find none — the prefill_interval
                // fallback in nextToken covers long ones.
                boundary_ckpts = computeMessageBoundaries(text_tokens, state_ckpt_min_gap);
                const auto all = computeMessageBoundaries(text_tokens, /*min_gap*/ 1);
                if (!all.empty() &&
                    (boundary_ckpts.empty() || boundary_ckpts.back() != all.back())) {
                    boundary_ckpts.push_back(all.back());
                }
            } else if (!cache_remove_success && (llama_pos) n_common > n_past &&
                       (llama_pos) n_common < (llama_pos) num_prompt_tokens) {
                // Restore path (seq_rm failed): capture the advanced frontier at
                // n_common as we reprocess, so the checkpoint moves forward each
                // turn (a rollback-0 model would otherwise freeze at its first
                // snapshot). One split, only where seq_rm can't reuse in place.
                boundary_ckpts.push_back((llama_pos) n_common);
            }
        }
        // Eviction only keeps {first} + the newest few; don't serialize (or split
        // prefill batches for) boundaries that can't survive this ingest.
        if (boundary_ckpts.size() > state_cache_max_checkpoints) {
            const size_t keep_tail = state_cache_max_checkpoints - 1;
            std::vector<llama_pos> filtered;
            filtered.reserve(state_cache_max_checkpoints);
            filtered.push_back(boundary_ckpts.front());
            filtered.insert(filtered.end(),
                            boundary_ckpts.end() - keep_tail, boundary_ckpts.end());
            boundary_ckpts = std::move(filtered);
        }
        // Cold stays pending even with no boundaries (interval fallback); warm
        // only when we armed one.
        prompt_checkpoint_pending = state_cache_enabled && state_cache_capture_allowed &&
                                    (cold_ingest || !boundary_ckpts.empty());

        LOG_VERBOSE("prompt ingested, n_past: %d, cached: %s, to_eval: %s",
            n_past,
            tokens_to_str(parent_ctx->active_ctx(), embd.cbegin(), embd.cbegin() + n_past).c_str(),
            tokens_to_str(parent_ctx->active_ctx(), embd.cbegin() + n_past, embd.cend()).c_str()
        );
    } else {
        // Multimodal path - process all media paths
        processMedia(parent_ctx->params.prompt, media_paths);
        num_prompt_tokens = embd.size();
        // Placeholder tokens are not vocab ids; no delimiter scan on media prompts.
        boundary_ckpts.clear();
        // Don't arm prompt-region snapshots: processMedia already ingested the
        // whole prompt and captured via its callback; nothing left for nextToken.
        prompt_checkpoint_pending = false;
    }

    // Handle encoder-decoder models (like T5) with special encoding phase
    if (is_enc_dec && !has_media) {
        // For encoder-decoder models, we need to encode the input tokens first
        if (embd.size() > n_past) {
            // Encode tokens in batches using n_batch as chunk size
            int n_past_batch = n_past;
            int n_remaining = embd.size() - n_past;

            while (n_remaining > 0) {
                int n_eval = n_remaining;
                if (n_eval > parent_ctx->params.n_batch) {
                    n_eval = parent_ctx->params.n_batch;
                }

                int ret = llama_encode(parent_ctx->active_ctx(), llama_batch_get_one(embd.data() + n_past_batch, n_eval));
                if (ret < 0) {
                    LOG_ERROR("Failed to encode token batch, code: %d, n_eval: %d, n_past_batch: %d", ret, n_eval, n_past_batch);
                    has_next_token = false;
                    return;
                }

                n_past_batch += n_eval;
                n_remaining -= n_eval;
                n_past += n_eval;
            }
        }
        // Update token count for encoding
        num_prompt_tokens = embd.size();

        // Add decoder start token for encoder-decoder models
        llama_token decode_bos = llama_model_decoder_start_token(parent_ctx->model);
        if (decode_bos == LLAMA_TOKEN_NULL) {
            decode_bos = llama_vocab_bos(vocab);
        }

        // Add the decoder start token to begin generation
        embd.emplace_back(decode_bos);
        common_sampler_accept(ctx_sampling, decode_bos, false);

        LOG_INFO("[DEBUG] T5 encoding complete, added decoder BOS token: %d", decode_bos);
    }

    has_next_token = true;

    LOG_INFO("[DEBUG] Input processed: n_past=%d, embd.size=%zu, num_prompt_tokens=%zu, has_media=%d",
            n_past, embd.size(), num_prompt_tokens, has_media ? 1 : 0);
}

void llama_rn_context_completion::beginCompletion() {
    beginCompletion(COMMON_CHAT_FORMAT_CONTENT_ONLY, COMMON_REASONING_FORMAT_NONE);
}

void llama_rn_context_completion::beginCompletion(int chat_format, common_reasoning_format reasoning_format, const std::string &generation_prompt, const std::string &chat_parser) {
    // number of tokens to keep when resetting context
    n_remain = parent_ctx->params.n_predict;
    parent_ctx->resetGovernorPrefillStats();
    llama_perf_context_reset(parent_ctx->active_ctx());
    resetGenerationTimings();
    is_predicting = true;

    current_chat_format = chat_format;
    current_reasoning_format = reasoning_format;
    current_generation_prompt = generation_prompt;
    current_chat_parser = chat_parser;
}

void llama_rn_context_completion::endCompletion() {
    generated_text += utf8_gate.finish();
    incomplete = false;
    // Trim the undecoded final token. On a stop-word / token-budget stop the last
    // sampled token is already pushed to embd but never decoded.
    if (n_past > 0 && n_past < (llama_pos) embd.size()) {
        embd.resize(n_past);
    }
    is_predicting = false;
}

void llama_rn_context_completion::resetGenerationTimings() {
    t_start_generation = 0;
    t_token_generation = 0.0;
}

void llama_rn_context_completion::startGenerationTiming() {
    if (t_start_generation == 0) {
        t_start_generation = lm_ggml_time_us();
    }
}

void llama_rn_context_completion::updateGenerationTiming() {
    if (t_start_generation != 0 && num_tokens_predicted > 0) {
        t_token_generation = (lm_ggml_time_us() - t_start_generation) / 1e6;
    }
}

bool llama_rn_context_completion::shouldUseMTP() const {
    if (parent_ctx == nullptr || parent_ctx->active_ctx() == nullptr || parent_ctx->model == nullptr) {
        return false;
    }
    if (parent_ctx->hasGovernor()) {
        return false;
    }
    // spec_n_past == -1 latches a context whose MTP init failed: the plain
    // path serves the request from loadPrompt's embd/n_past (see the catch
    // in initMTP) until rewind() re-arms the latch.
    if (spec_n_past == -1) {
        return false;
    }
    // Kalsa patch: DFLASH rides the same draft-speculative init/decode path as
    // MTP (common_speculative dispatches per-type); without this the whole
    // speculative machinery never initializes for a pure draft-dflash config.
    const auto & types = parent_ctx->params.speculative.types;
    const bool has_draft_type =
        std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end() ||
        std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != types.end();
    if (!has_draft_type || parent_ctx->params.speculative.draft.n_max <= 0) {
        return false;
    }
    if (llama_model_has_encoder(parent_ctx->model)) {
        if (!mtp_capability_logged) {
            mtp_capability_logged = true;
            LOG_ERROR("MTP speculative decoding is only supported for decoder-only models, running plain");
        }
        return false;
    }
    const auto n_mtp = parent_ctx->params.speculative.draft.n_max;
    if ((llama_model_is_recurrent(parent_ctx->model) || llama_model_is_hybrid(parent_ctx->model)) &&
        llama_n_rs_seq(parent_ctx->active_ctx()) < (uint32_t) n_mtp) {
        if (!mtp_capability_logged) {
            mtp_capability_logged = true;
            LOG_ERROR("MTP for recurrent or hybrid models must be enabled when loading the model "
                      "with speculative.type='draft-mtp' and speculative.n_max/spec_draft_n_max set; running plain");
        }
        return false;
    }
    return true;
}

void llama_rn_context_completion::resetSpeculative() {
    if (spec != nullptr) {
        common_speculative_free(spec);
        spec = nullptr;
    }
    spec_ctx.reset();
    if (spec_batch_initialized) {
        llama_batch_free(spec_batch);
        spec_batch = {};
        spec_batch_initialized = false;
    }
    spec_prompt.clear();
    spec_id_last = LLAMA_TOKEN_NULL;
    spec_n_past = 0;
    spec_draft.clear();
    spec_pending_tokens.clear();
    draft_rollback_fenced = false;
}

// One error line, then plain. The target is untouched at every call site, so
// loadPrompt's embd/n_past stay valid and nextToken's plain while-loop decodes
// the prompt tail.
void llama_rn_context_completion::fallBackToPlain(const char * why) {
    if (!mtp_capability_logged) {
        mtp_capability_logged = true;
        LOG_ERROR("%s", why);
    }
    resetSpeculative();
    spec_n_past = -1;
}

void llama_rn_context_completion::initMTP() {
    if (!shouldUseMTP()) {
        return;
    }
    if (embd.empty()) {
        throw std::runtime_error("MTP speculative decoding requires a non-empty prompt");
    }

    try {
        resetSpeculative();
        spec_ctx.reset(parent_ctx->createMTPDraftContext(parent_ctx->params));
        if (spec_ctx == nullptr) {
            throw std::runtime_error("failed to create MTP draft context");
        }

        parent_ctx->params.speculative.draft.ctx_tgt = parent_ctx->active_ctx();
        parent_ctx->params.speculative.draft.ctx_dft = spec_ctx.get();

        spec = common_speculative_init(parent_ctx->params.speculative, 1);
        if (spec == nullptr) {
            throw std::runtime_error("failed to initialize MTP speculative decoding");
        }
    } catch (const std::exception& e) {
        // createMTPDraftContext bottoms out in llama_init_from_model, which
        // throws for pure recurrent and hybrid-SWA architectures (ctx_other
        // unsupported). Nothing above has touched the target -- embd/n_past
        // from loadPrompt stay valid -- so the plain path serves this turn.
        std::string why = "this model cannot create an MTP draft context (" + std::string(e.what()) + "); running plain";
        fallBackToPlain(why.c_str());
        return;
    }

    // Probe the draft clear before anything else consumes this draft: on this
    // pin the draft's KV cache is shared with the target for dense models, and
    // a shared cache refuses both seq_rm and init_batch (llama-kv-cache.cpp:411
    // and :757, TAG_KV_CACHE_SHARE_CELLS), so common_speculative_process could
    // never decode a proposal from it. When kalsallama lifts the decode fence
    // for shared cells, this probe must be re-evaluated: seq_rm may still be
    // refused while decode works, and this latch would then disable MTP that
    // could run.
    if (!rn_seq_rm(spec_ctx.get(), 0, -1, -1)) {
        fallBackToPlain("MTP: this engine cannot decode from a KV cache shared with the target (llama-kv-cache.cpp init_batch fence, TAG_KV_CACHE_SHARE_CELLS); running plain");
        return;
    }

    spec_batch = llama_batch_init(llama_n_batch(parent_ctx->active_ctx()), 0, 1);
    spec_batch_initialized = true;

    // A mem-shared draft (e.g. gemma4/EAGLE3) shares the target's KV cells, where
    // upstream state save/restore is a no-op — restoring a checkpoint would leave
    // stale speculative cells and corrupt the output. Disable the cache for it
    // (full-reprocess each turn) until upstream supports shared-cell restore.
    if (llama_get_ctx_other(spec_ctx.get()) == parent_ctx->active_ctx()) {
        if (state_cache_enabled) {
            LOG_INFO("state cache disabled for this turn: mem-shared MTP draft "
                     "(shared-cell state restore is unsupported upstream)");
        }
        mtp_draft_mem_shared = true;
        state_cache_enabled = false;
        clearStateCheckpoints();
    }

    evalMTPPrompt();
    startGenerationTiming();
}

void llama_rn_context_completion::evalMTPPrompt() {
    const llama_seq_id seq_id = 0;
    const size_t n_prompt = embd.size();

    spec_prompt.clear();
    spec_pending_tokens.clear();
    spec_draft.clear();
    spec_id_last = embd.back();

    if (n_prompt > 1) {
        spec_prompt.assign(embd.begin(), embd.end() - 1);
    }

    const int32_t n_batch = std::max<int32_t>(1, llama_n_batch(parent_ctx->active_ctx()));

    // Reuse whatever prefix loadPrompt already left in the (shared) target
    // memory: it set n_past to the reused position. Decode only the diverged tail.
    size_t offset = std::min((size_t) std::max<llama_pos>(0, n_past), spec_prompt.size());
    const size_t start_offset = offset;

    while (offset < spec_prompt.size()) {
        size_t decode_to = spec_prompt.size();
        // Cold ingest only: stop at the next message boundary to snapshot there
        // (boundary_ckpts is empty on warm turns, so the tail decodes whole).
        {
            const auto next_boundary = std::upper_bound(
                boundary_ckpts.begin(), boundary_ckpts.end(), (llama_pos) offset);
            if (next_boundary != boundary_ckpts.end() &&
                (size_t) *next_boundary < spec_prompt.size()) {
                decode_to = std::min(decode_to, (size_t) *next_boundary);
            }
        }

        common_batch_clear(spec_batch);

        const size_t n_eval = std::min<size_t>(n_batch, decode_to - offset);
        for (size_t i = 0; i < n_eval; ++i) {
            // MTP consumes pre-norm embeddings from every target row, but prompt logits are unused.
            // Keep one output row per decode batch to preserve the usual llama.cpp graph shape.
            const bool needs_logits = i + 1 == n_eval;
            common_batch_add(spec_batch, spec_prompt[offset + i],
                             (llama_pos) (offset + i), { seq_id }, needs_logits);
        }

        const int ret = parent_ctx->decode(spec_batch);
        if (ret != 0) {
            // Memory holds only [0, offset); trim embd so a later prefix match
            // can't claim never-decoded cells (mirrors nextToken).
            embd.resize(std::min(embd.size(), offset));
            n_past = (llama_pos) offset;
            throw std::runtime_error("failed to evaluate MTP prompt batch, ret=" + std::to_string(ret));
        }
        if (!common_speculative_process(spec, spec_batch)) {
            embd.resize(std::min(embd.size(), offset));
            n_past = (llama_pos) offset;
            throw std::runtime_error("failed to process MTP prompt batch");
        }

        offset += n_eval;

        // Cold ingest only: snapshot at boundary positions the moment we reach
        // them. Warm turns captured their frontier in loadPrompt already.
        if (std::binary_search(boundary_ckpts.begin(), boundary_ckpts.end(),
                               (llama_pos) offset)) {
            captureStateCheckpoint(spec_prompt, offset);
        }
    }

    mtp_prompt_reprocessed = spec_prompt.size() - start_offset;

    spec_n_past = (llama_pos) spec_prompt.size();
    n_past = spec_n_past;

    common_speculative_begin(spec, seq_id, spec_prompt);
}

bool llama_rn_context_completion::refillMTPTokens() {
    const llama_seq_id seq_id = 0;

    if (spec_id_last == LLAMA_TOKEN_NULL || stopped_eos || stopped_limit || context_full) {
        return false;
    }
    if (parent_ctx->params.n_predict >= 0 && n_remain == 0) {
        stopped_limit = true;
        has_next_token = false;
        return false;
    }

    const int32_t n_ctx = parent_ctx->params.n_ctx;
    if (spec_n_past + 1 >= n_ctx) {
        context_full = true;
        has_next_token = false;
        return false;
    }

    spec_draft.clear();

    const int32_t remaining =
        parent_ctx->params.n_predict < 0 ? std::numeric_limits<int32_t>::max() : (int32_t) n_remain;
    const int32_t n_draft_remaining = remaining == std::numeric_limits<int32_t>::max()
        ? parent_ctx->params.speculative.draft.n_max
        : std::max<int32_t>(0, remaining - 1);
    const int32_t n_draft_ctx = std::max<int32_t>(0, n_ctx - (int32_t) spec_n_past - 1);
    const int32_t n_draft_batch = std::max<int32_t>(0, llama_n_batch(parent_ctx->active_ctx()) - 1);
    const int32_t n_draft_limit = std::min<int32_t>(
        parent_ctx->params.speculative.draft.n_max,
        std::min<int32_t>(n_draft_remaining, std::min<int32_t>(n_draft_ctx, n_draft_batch)));

    if (n_draft_limit > 0) {
        common_speculative_get_draft_params(spec, seq_id) = {
            /* .drafting = */ true,
            /* .n_max    = */ n_draft_limit,
            /* .n_past   = */ spec_n_past,
            /* .id_last  = */ spec_id_last,
            /* .prompt   = */ &spec_prompt,
            /* .result   = */ &spec_draft,
        };
        common_speculative_draft(spec);

        if ((int32_t) spec_draft.size() > n_draft_limit) {
            spec_draft.resize(n_draft_limit);
        }

        // Draft-side removal is advisory: every proposal is verified by the
        // target sampler, so a failure here never corrupts output. The init
        // probe latches fenced caches, so this site is reachable only if the
        // cache's sharing changes after init.
        if (!draft_rollback_fenced && !rn_seq_rm(spec_ctx.get(), seq_id, spec_n_past, -1)) {
            draft_rollback_fenced = true;
            LOG_INFO("MTP: draft memory refuses rollbacks (shared KV cache); proposals stay verified by the target");
        }
    }

    const size_t n_draft = spec_draft.size();
    num_draft_tokens += n_draft;

    common_batch_clear(spec_batch);
    common_batch_add(spec_batch, spec_id_last, spec_n_past, { seq_id }, true);
    for (size_t i = 0; i < n_draft; ++i) {
        common_batch_add(spec_batch, spec_draft[i],
                         spec_n_past + (llama_pos) i + 1, { seq_id }, true);
    }

    const int ret = parent_ctx->decode(spec_batch);
    if (ret != 0) {
        throw std::runtime_error("failed to evaluate MTP target batch, ret=" + std::to_string(ret));
    }
    if (!common_speculative_process(spec, spec_batch)) {
        throw std::runtime_error("failed to process MTP target batch");
    }

    auto accepted = common_sampler_sample_and_accept_n(ctx_sampling, parent_ctx->active_ctx(), spec_draft);
    if (accepted.empty()) {
        return false;
    }

    size_t accepted_count = accepted.size();
    bool saw_eos = false;
    const llama_vocab* vocab = llama_model_get_vocab(parent_ctx->model);
    for (size_t i = 0; i < accepted.size(); ++i) {
        if (llama_vocab_is_eog(vocab, accepted[i])) {
            accepted_count = i + 1;
            saw_eos = true;
            break;
        }

        completion_token_output output;
        output.tok = accepted[i];
        output.text = common_token_to_piece(parent_ctx->active_ctx(), accepted[i]);
        spec_pending_tokens.push_back(std::move(output));
    }

    const size_t n_accepted_draft = saw_eos
        ? accepted_count - 1
        : accepted.size() - 1;
    if (n_draft > 0) {
        const size_t n_accepted = std::min(n_accepted_draft, n_draft);
        num_draft_tokens_accepted += n_accepted;
        common_speculative_accept(spec, seq_id, (uint16_t) n_accepted);
    }

    for (size_t i = 0; i < accepted_count; ++i) {
        spec_prompt.push_back(spec_id_last);
        spec_id_last = accepted[i];
    }

    spec_n_past += (llama_pos) accepted_count;
    n_past = spec_n_past;

    if (!rn_seq_rm(parent_ctx->active_ctx(), seq_id, spec_n_past, -1)) {
        throw std::runtime_error("MTP: failed to truncate sequence " + std::to_string(seq_id) +
                                 " to pos " + std::to_string(spec_n_past) + " in the target context");
    }
    if (!draft_rollback_fenced && !rn_seq_rm(spec_ctx.get(), seq_id, spec_n_past, -1)) {
        // Draft-side removal is advisory (see the first rollback site in
        // refillMTPTokens); fall through to the saw_eos / n_predict tail
        // below.
        draft_rollback_fenced = true;
        LOG_INFO("MTP: draft memory refuses rollbacks (shared KV cache); proposals stay verified by the target");
    }

    if (saw_eos) {
        stopped_eos = true;
        has_next_token = false;
    }

    if (parent_ctx->params.n_predict >= 0) {
        const size_t emitted = spec_pending_tokens.size();
        n_remain = emitted >= n_remain ? 0 : n_remain - emitted;
        if (n_remain == 0 && !saw_eos) {
            stopped_limit = true;
            has_next_token = false;
        }
    }

    return !spec_pending_tokens.empty();
}

completion_token_output llama_rn_context_completion::nextTokenMTP() {
    completion_token_output result;
    result.tok = -1;

    if (spec == nullptr && spec_n_past != -1) {
        initMTP();
    }
    startGenerationTiming();

    if (spec_pending_tokens.empty() && !refillMTPTokens()) {
        return result;
    }

    result = std::move(spec_pending_tokens.front());
    spec_pending_tokens.pop_front();
    num_tokens_predicted++;
    updateGenerationTiming();
    has_next_token = !spec_pending_tokens.empty() || (!stopped_eos && !stopped_limit && !context_full);
    return result;
}

completion_token_output llama_rn_context_completion::nextToken()
{
    if (shouldUseMTP()) {
        return nextTokenMTP();
    }

    completion_token_output result;
    result.tok = -1;

    if (embd.size() >= (size_t)parent_ctx->params.n_ctx)
    {
        if (!parent_ctx->params.ctx_shift) {
            // If context shifting is disabled, stop generation
            LOG_WARNING("context full, n_ctx: %d, tokens: %d", parent_ctx->params.n_ctx, embd.size());
            has_next_token = false;
            context_full = true;
            return result;
        }

        // seq_rm + seq_add remaps positions and frees cells (pos_add when
        // pos<0). llama_kv_cache::update then applies K-shift
        // (llama-kv-cache.cpp L912). Hybrid seq_add hits attn+recr
        // (llama-memory-hybrid.cpp). S23 2bbef14 T20C t4: n_common=7379 then
        // tokensCached=4304; t5 n_common=0 heads disjoint. JS resends the
        // original prefix — keep live KV. Do not clearStateCheckpoints.
        // Do not set context_full: JS anchored forceRebuild would clearCache.
        rnllama::log("WARNING", __func__, __LINE__,
            "KALSA_KVSHIFT refused embd=%zu n_ctx=%d n_past=%d n_keep=%d keep_prefix=1",
            embd.size(), parent_ctx->params.n_ctx, (int) n_past,
            parent_ctx->params.n_keep);
        if (n_past >= 0 && embd.size() > (size_t) n_past) {
            embd.resize((size_t) n_past);
        }
        has_next_token = false;
        truncated = true;
        return result;
    }

    // Interval fallback (0 = off): only for cold prompts with no message
    // boundaries (prompt_checkpoint_pending is cold-ingest-only, see loadPrompt).
    const llama_pos prefill_interval =
        (state_cache_enabled && prompt_checkpoint_pending && boundary_ckpts.empty())
            ? (llama_pos) state_ckpt_prefill_interval
            : 0;

    bool tg = true;
    while (n_past < embd.size())
    {
        llama_pos decode_to = (llama_pos) embd.size();
        // Cold ingest only: split the decode at snapshot positions so we can
        // capture there (boundary / interval). Warm turns leave boundary_ckpts
        // empty and prompt_checkpoint_pending false, so the tail decodes whole.
        if (prompt_checkpoint_pending) {
            const auto next_boundary = std::upper_bound(
                boundary_ckpts.begin(), boundary_ckpts.end(), n_past);
            if (next_boundary != boundary_ckpts.end()) {
                decode_to = std::min(decode_to, *next_boundary);
            }
        }
        if (prefill_interval > 0) {
            const llama_pos next_ckpt = (n_past / prefill_interval + 1) * prefill_interval;
            if (next_ckpt < (llama_pos) num_prompt_tokens) {
                decode_to = std::min(decode_to, next_ckpt);
            }
        }
        int n_eval = (int)(decode_to - n_past);
        tg = ((int) embd.size() - n_past) == 1;
        if (n_eval > parent_ctx->params.n_batch)
        {
            n_eval = parent_ctx->params.n_batch;
        }
        if (parent_ctx->decode(llama_batch_get_one(&embd[n_past], n_eval)))
        {
            LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d, embd: %s",
                n_eval,
                n_past,
                parent_ctx->params.cpuparams.n_threads,
                tokens_to_str(parent_ctx->active_ctx(), embd.cbegin() + n_past, embd.cend()).c_str()
            );
            // Trim embd to what the memory actually contains so a later prefix
            // match can't claim never-written cells.
            embd.resize(n_past);
            has_next_token = false;
            return result;
        }
        n_past += n_eval;

        if(is_interrupted) {
            LOG_INFO("Decoding Interrupted");
            embd.resize(n_past);
            has_next_token = false;
            return result;
        }

        // Cold ingest only: snapshot at boundary / interval positions as we
        // reach them. Warm turns captured their frontier in loadPrompt already.
        if (prompt_checkpoint_pending &&
            std::binary_search(boundary_ckpts.begin(), boundary_ckpts.end(), n_past)) {
            captureStateCheckpoint();
        }
        else if (prefill_interval > 0 && n_past < (llama_pos) num_prompt_tokens &&
                 n_past % prefill_interval == 0) {
            captureStateCheckpoint();
        }
    }

    // Prompt end not captured: the next turn's frontier capture covers it.
    if (prompt_checkpoint_pending && n_past >= (llama_pos) num_prompt_tokens) {
        prompt_checkpoint_pending = false;
    }

    // No snapshots during generation: a stable append reuses the reply via
    // seq_rm; otherwise the next ingest reprocesses it once and lays a boundary
    // snapshot after it.

    const bool forced = bench_force_ids_enabled;
    const bool raw_probs_requested = bench_raw_probs > 0;
    result.raw_probs_requested = raw_probs_requested;
    result.forced = forced;

    llama_context* active_ctx = parent_ctx != nullptr ? parent_ctx->active_ctx() : nullptr;
    const llama_model* active_model =
        active_ctx != nullptr ? llama_get_model(active_ctx) : nullptr;
    const llama_vocab* vocab =
        active_model != nullptr ? llama_model_get_vocab(active_model) : nullptr;
    const int32_t n_vocab = vocab != nullptr ? llama_vocab_n_tokens(vocab) : 0;
    if (active_ctx == nullptr || active_model == nullptr || vocab == nullptr || n_vocab <= 0) {
        has_next_token = false;
        return result;
    }

    if (parent_ctx->params.n_predict == 0)
    {
        has_next_token = false;
        result.tok = llama_vocab_eos(vocab);
        return result;
    }

    if (ctx_sampling == nullptr) {
        has_next_token = false;
        return result;
    }

    startGenerationTiming();

    {
        // out of user input, sample next token
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);

        if (forced && bench_force_index >= bench_force_ids.size()) {
            has_next_token = false;
            return result;
        }

        if (raw_probs_requested) {
            benchRawTopProbs(active_ctx, vocab, bench_raw_probs, result.raw_probs);
        }

        llama_token new_token_id;
        if (forced) {
            const llama_token forced_token = bench_force_ids[bench_force_index];
            if (forced_token < 0 || (int64_t) forced_token >= (int64_t) n_vocab) {
                result.raw_probs.clear();
                has_next_token = false;
                return result;
            }
            ++bench_force_index;
            new_token_id = forced_token;
        } else {
            new_token_id = common_sampler_sample(ctx_sampling, active_ctx, -1);
        }

        const int32_t n_probs = parent_ctx->params.sampling.n_probs;
        if (n_probs > 0 && !forced) {
          llama_token_data_array * cur_p = common_sampler_get_candidates(ctx_sampling, true);
          if (cur_p != nullptr && cur_p->data != nullptr && cur_p->selected < cur_p->size) {
              for (size_t i = 0; i < std::min(cur_p->size, (size_t)n_probs); ++i)
              {
                  result.probs.push_back({cur_p->data[i].id, cur_p->data[i].p});
              }
          }
        }

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            has_next_token = false;
            stopped_eos = true;
            LOG_VERBOSE("EOS: %s", common_token_to_piece(parent_ctx->active_ctx(), new_token_id).c_str());
            return result;
        }

        if (parent_ctx->tts_wrapper != nullptr && parent_ctx->tts_wrapper->next_token_uses_guide_token && !parent_ctx->tts_wrapper->guide_tokens.empty() && !llama_vocab_is_control(vocab, new_token_id)) {
            new_token_id = parent_ctx->tts_wrapper->guide_tokens[0];
            parent_ctx->tts_wrapper->guide_tokens.erase(parent_ctx->tts_wrapper->guide_tokens.begin());
        }
        if (parent_ctx->tts_wrapper != nullptr) {
            parent_ctx->tts_wrapper->next_token_uses_guide_token = (new_token_id == 198);
        }
        result.tok = new_token_id;
        result.text = common_token_to_piece(parent_ctx->active_ctx(), new_token_id);

        common_sampler_accept(ctx_sampling, result.tok, true);
        if (tg) {
            num_tokens_predicted++;
            updateGenerationTiming();
        }
    }

    // add it to the context
    embd.push_back(result.tok);
    // decrement remaining sampling budget
    --n_remain;

    has_next_token = parent_ctx->params.n_predict == -1 || n_remain != 0;
    return result;
}

size_t llama_rn_context_completion::findStoppingStrings(const std::string &text, const size_t last_token_size,
                            const stop_type type)
{
    size_t stop_pos = std::string::npos;
    for (const std::string &word : parent_ctx->params.antiprompt)
    {
        size_t pos;
        if (type == STOP_FULL)
        {
            const size_t tmp = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
            pos = text.find(word, from_pos);
        }
        else
        {
            pos = find_partial_stop_string(word, text);
        }
        if (pos != std::string::npos &&
            (stop_pos == std::string::npos || pos < stop_pos))
        {
            if (type == STOP_FULL)
            {
                stopping_word = word;
                stopped_word = true;
                has_next_token = false;
            }
            stop_pos = pos;
        }
    }
    return stop_pos;
}

completion_token_output llama_rn_context_completion::doCompletion()
{
    completion_token_output token_with_probs = nextToken();

    if (parent_ctx->params.sampling.n_probs > 0 && token_with_probs.tok != -1) {
        generated_token_ids.push_back(token_with_probs.tok);
    }

    const std::string token_text = token_with_probs.tok == -1 ? "" : common_token_to_piece(parent_ctx->active_ctx(), token_with_probs.tok);
    generated_text += utf8_gate.feed(token_text);

    if (parent_ctx->isVocoderEnabled()) {
        tts_type type = parent_ctx->tts_wrapper->getTTSType(parent_ctx);
        if (parent_ctx->tts_wrapper->type == UNKNOWN) {
            parent_ctx->tts_wrapper->type = type;
        }
        if ((type == OUTETTS_V0_2 || type == OUTETTS_V0_3) && (token_with_probs.tok >= 151672 && token_with_probs.tok <= 155772)) {
            parent_ctx->tts_wrapper->audio_tokens.push_back(token_with_probs.tok);
        }
    }

    if (parent_ctx->params.sampling.n_probs > 0)
    {
        generated_token_probs.push_back(token_with_probs);
    }

    incomplete = utf8_gate.has_pending();

    if (incomplete && !has_next_token)
    {
        has_next_token = true;
        n_remain++;
    }

    if (!has_next_token && n_remain == 0)
    {
        stopped_limit = true;
    }

    LOG_VERBOSE("next token, token: %s, token_text: %s, has_next_token: %d, n_remain: %d, num_tokens_predicted: %d, stopped_eos: %d, stopped_word: %d, stopped_limit: %d, stopping_word: %s",
        token_with_probs.tok == -1 ? "" : common_token_to_piece(parent_ctx->active_ctx(), token_with_probs.tok).c_str(),
        token_with_probs.tok == -1 ? "" : tokens_to_output_formatted_string(parent_ctx->active_ctx(), token_with_probs.tok).c_str(),
        has_next_token,
        n_remain,
        num_tokens_predicted,
        stopped_eos,
        stopped_word,
        stopped_limit,
        stopping_word.c_str()
    );
    return token_with_probs;
}

completion_chat_output llama_rn_context_completion::parseChatOutput(bool is_partial) {
    common_chat_parser_params syntax;
    syntax.format = static_cast<common_chat_format>(current_chat_format);
    syntax.reasoning_format = current_reasoning_format;
    syntax.generation_prompt = current_generation_prompt;
    syntax.parse_tool_calls = true;

    // Load the PEG parser if available (required for COMMON_CHAT_FORMAT_PEG_* formats)
    if (!current_chat_parser.empty()) {
        syntax.parser.load(current_chat_parser);
    }

    common_chat_msg parsed_msg = common_chat_parse(prefill_text + generated_text, is_partial, syntax);

    completion_chat_output result;

    result.content = parsed_msg.content;
    result.reasoning_content = parsed_msg.reasoning_content;
    result.accumulated_text = prefill_text + generated_text;
    result.tool_calls = parsed_msg.tool_calls;

    return result;
}

std::vector<float> llama_rn_context_completion::embedding(common_params &embd_params)
{
    llama_memory_clear(llama_get_memory(parent_ctx->active_ctx()), true);

    rewind();
    llama_perf_context_reset(parent_ctx->active_ctx());
    if (!initSampling()) {
        throw std::runtime_error("Failed to initialize sampling");
    }
    beginCompletion();
    loadPrompt({}, /*allow_state_cache*/ false);
    doCompletion();
    endCompletion();

    static const int n_embd = llama_model_n_embd(llama_get_model(parent_ctx->active_ctx()));
    if (!embd_params.embedding)
    {
        LOG_WARNING("embedding disabled, embedding: %s", embd_params.embedding);
        return std::vector<float>(n_embd, 0.0f);
    }
    float *data;

    const enum llama_pooling_type pooling_type = llama_pooling_type(parent_ctx->active_ctx());
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        data = llama_get_embeddings(parent_ctx->active_ctx());
    } else {
        data = llama_get_embeddings_seq(parent_ctx->active_ctx(), 0);
    }

    if (!data) {
        return std::vector<float>(n_embd, 0.0f);
    }
    std::vector<float> embedding(data, data + n_embd), out(data, data + n_embd);
    common_embd_normalize(embedding.data(), out.data(), n_embd, embd_params.embd_normalize);
    return out;
}

std::vector<float> llama_rn_context_completion::rerank(const std::string &query, const std::vector<std::string> &documents)
{
    std::vector<float> scores;

    // Check if this model supports reranking (requires rank pooling type)
    const enum llama_pooling_type pooling_type = llama_pooling_type(parent_ctx->active_ctx());
    if (pooling_type != LLAMA_POOLING_TYPE_RANK) {
        throw std::runtime_error("reranking not supported, pooling_type: " + std::to_string(pooling_type));
    }

    if (!parent_ctx->params.embedding) {
        throw std::runtime_error("embedding disabled but required for reranking");
    }

    const llama_vocab * vocab = llama_model_get_vocab(parent_ctx->model);
    std::vector<llama_token> query_tokens = common_tokenize(vocab, query, false, true);

    scores.reserve(documents.size());

    for (size_t i = 0; i < documents.size(); ++i) {
        rewind();
        embd = {};

        const std::string & document = documents[i];

        std::vector<llama_token> doc_tokens = common_tokenize(vocab, document, false, true);

        std::vector<llama_token> rerank_tokens = format_rerank_tokens(vocab, query_tokens, doc_tokens);

        llama_memory_clear(llama_get_memory(parent_ctx->active_ctx()), false);

        // Process the rerank input
        try {
            parent_ctx->params.prompt = tokens_to_str(parent_ctx->active_ctx(), rerank_tokens.begin(), rerank_tokens.end());
            initSampling();
            loadPrompt({}, /*allow_state_cache*/ false); // No media paths for rerank
            beginCompletion();
            doCompletion();

            // Get the rerank score (single embedding value for rank pooling)
            float *data = llama_get_embeddings_seq(parent_ctx->active_ctx(), 0);
            if (data) {
                scores.push_back(data[0]); // For rank pooling, the score is the first (and only) dimension
            } else {
                scores.push_back(-1e6f); // Default low score if computation failed
            }
        } catch (const std::exception &e) {
            LOG_WARNING("rerank computation failed for document %zu: %s", i, e.what());
            scores.push_back(-1e6f);
        }
        endCompletion();

        // Clear KV cache again to prepare for next document or restore original state
        llama_memory_clear(llama_get_memory(parent_ctx->active_ctx()), false);
    }

    return scores;
}

std::string llama_rn_context_completion::bench(int pp, int tg, int pl, int nr) {
    if (is_predicting) {
        LOG_ERROR("cannot benchmark while predicting", "");
        return std::string("{}");
    }

    if (pp <= 0 || tg <= 0 || pl <= 0 || nr <= 0) {
        LOG_ERROR("invalid benchmark parameters pp=%d tg=%d pl=%d nr=%d", pp, tg, pl, nr);
        return std::string("{}");
    }

    is_predicting = true;

    auto * ctx = parent_ctx->active_ctx();
    auto * model = parent_ctx->model;
    auto * mem = llama_get_memory(ctx);

    const bool is_pp_shared = parent_ctx->params.is_pp_shared;
    const bool kv_unified   = parent_ctx->params.kv_unified;
    const int32_t n_batch   = parent_ctx->params.n_batch;
    const int32_t n_ubatch  = parent_ctx->params.n_ubatch;
    const int32_t flash_attn = static_cast<int32_t>(parent_ctx->params.flash_attn_type);
    const int32_t n_gpu_layers = parent_ctx->params.n_gpu_layers;
    const int32_t n_threads = llama_n_threads(ctx);
    const int32_t n_threads_batch = llama_n_threads_batch(ctx);
    const int32_t n_kv_max = llama_n_ctx(ctx);

    const int32_t n_ctx_req = is_pp_shared
        ? (kv_unified ? pp : pl * pp) + pl * tg
        : pl * (pp + tg);

    if (n_ctx_req > n_kv_max) {
        LOG_ERROR("benchmark requires n_ctx=%d but only %d available", n_ctx_req, n_kv_max);
        endCompletion();
        return std::string("{}");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;

    auto get_token_rand = [n_vocab]() -> llama_token {
        if (n_vocab <= 0) {
            return 0;
        }
        return std::rand() % n_vocab;
    };

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    auto decode_helper = [this, ctx](llama_batch & batch_ref, int32_t n_batch_ref, bool synchronize) -> bool {
        const int32_t total = batch_ref.n_tokens;
        for (int32_t i = 0; i < total; i += n_batch_ref) {
            const int32_t n_tokens_step = std::min(n_batch_ref, total - i);

            llama_batch batch_view = {
                n_tokens_step,
                batch_ref.token    + i,
                nullptr,
                batch_ref.pos      + i,
                batch_ref.n_seq_id + i,
                batch_ref.seq_id   + i,
                batch_ref.logits   + i,
            };

            const int ret = parent_ctx->decode(batch_view);
            if (ret != 0) {
                LOG_ERROR("llama_decode() failed during benchmark, n_batch=%d ret=%d", n_batch_ref, ret);
                return false;
            }

            if (synchronize) {
                llama_synchronize(ctx);
            }
        }

        return true;
    };

    // warm up like the CLI benchmark
    llama_batch_clear(&batch);
    const int warmup_tokens = std::min(16, n_kv_max);
    for (int i = 0; i < warmup_tokens; ++i) {
        llama_batch_add(&batch, get_token_rand(), i, {0}, i == warmup_tokens - 1);
    }
    if (!decode_helper(batch, n_batch, true)) {
        llama_batch_free(batch);
        endCompletion();
        return std::string("{}");
    }

    double acc_t_pp = 0.0;
    double acc_t_tg = 0.0;
    double acc_speed_pp = 0.0;
    double acc_speed_tg = 0.0;
    double acc_t_total = 0.0;
    double acc_speed_total = 0.0;

    int runs_completed = 0;

    for (int run = 0; run < nr && !is_interrupted; ++run) {
        bool run_failed = false;

        llama_batch_clear(&batch);

        const int prompt_sequences = is_pp_shared ? 1 : pl;
        for (int seq = 0; seq < prompt_sequences; ++seq) {
            for (int i = 0; i < pp; ++i) {
                llama_batch_add(&batch, get_token_rand(), i, {static_cast<llama_seq_id>(seq)}, i == pp - 1);
            }
        }

        llama_memory_clear(mem, false);

        const auto t_pp_start = lm_ggml_time_us();
        if (!decode_helper(batch, n_batch, false)) {
            run_failed = true;
            break;
        }

        llama_synchronize(ctx);
        const auto t_pp_end = lm_ggml_time_us();

        if (is_pp_shared && pl > 1) {
            for (int32_t seq = 1; seq < pl; ++seq) {
                llama_memory_seq_cp(mem, 0, seq, -1, -1);
            }

            if (!kv_unified) {
                llama_batch_clear(&batch);
                llama_batch_add(&batch, get_token_rand(), pp, {0}, true);
                if (!decode_helper(batch, n_batch, true)) {
                    run_failed = true;
                    break;
                }
                llama_memory_seq_rm(mem, 0, pp, -1);
            }
        }

        if (run_failed) {
            break;
        }

        const auto t_tg_start = lm_ggml_time_us();

        for (int i = 0; i < tg; ++i) {
            llama_batch_clear(&batch);

            for (int seq = 0; seq < pl; ++seq) {
                llama_batch_add(&batch, get_token_rand(), pp + i, {static_cast<llama_seq_id>(seq)}, true);
            }

            if (!decode_helper(batch, n_batch, true)) {
                run_failed = true;
                break;
            }
        }

        if (run_failed) {
            break;
        }

        const auto t_tg_end = lm_ggml_time_us();

        const double t_pp = (t_pp_end - t_pp_start) / 1e6;
        const double t_tg = (t_tg_end - t_tg_start) / 1e6;
        const double t_total = t_pp + t_tg;

        const double prompt_tokens = is_pp_shared ? static_cast<double>(pp) : static_cast<double>(pl * pp);
        const double generated_tokens = static_cast<double>(pl * tg);

        const double speed_pp = t_pp > 0.0 ? prompt_tokens / t_pp : 0.0;
        const double speed_tg = t_tg > 0.0 ? generated_tokens / t_tg : 0.0;
        const double speed_total = t_total > 0.0 ? (prompt_tokens + generated_tokens) / t_total : 0.0;

        acc_t_pp += t_pp;
        acc_t_tg += t_tg;
        acc_speed_pp += speed_pp;
        acc_speed_tg += speed_tg;
        acc_t_total += t_total;
        acc_speed_total += speed_total;

        ++runs_completed;
    }

    llama_memory_clear(mem, false);

    const double divisor = runs_completed > 0 ? static_cast<double>(runs_completed) : 1.0;

    json result_json = {
        {"n_kv_max", n_kv_max},
        {"n_batch", n_batch},
        {"n_ubatch", n_ubatch},
        {"flash_attn", flash_attn},
        {"is_pp_shared", is_pp_shared ? 1 : 0},
        {"n_gpu_layers", n_gpu_layers},
        {"n_threads", n_threads},
        {"n_threads_batch", n_threads_batch},
        {"pp", pp},
        {"tg", tg},
        {"pl", pl},
        {"n_kv", n_ctx_req},
        {"t_pp", acc_t_pp / divisor},
        {"speed_pp", acc_speed_pp / divisor},
        {"t_tg", acc_t_tg / divisor},
        {"speed_tg", acc_speed_tg / divisor},
        {"t", acc_t_total / divisor},
        {"speed", acc_speed_total / divisor}
    };

    llama_batch_free(batch);
    endCompletion();

    return result_json.dump();
}

void llama_rn_context_completion::processMedia(
    const std::string &prompt,
    const std::vector<std::string> &media_paths
) {
    if (!parent_ctx->isMultimodalEnabled()) {
        throw std::runtime_error("Multimodal is not enabled but image paths are provided");
    }

    // Delegate to the mtmd_wrapper method
    // For non-parallel mode, use the global bitmap_past_hashes from mtmd_wrapper
    probeStateCache();
    // Wire the state cache into the media path so images aren't re-encoded
    // every turn.
    auto recover = [this](const std::vector<llama_token> &target, size_t max_reuse,
                          size_t total_tokens, llama_pos &n_past_out) {
        return recoverStateCheckpoint(target, max_reuse, total_tokens, n_past_out);
    };
    auto capture = [this](const std::vector<llama_token> &seq, size_t n) {
        captureStateCheckpoint(seq, n);
    };
    auto invalidate = [this](size_t n) {
        eraseStateCheckpointsAfter(n);
    };
    parent_ctx->mtmd_wrapper->processMedia(
        parent_ctx->active_ctx(),
        prompt,
        media_paths,
        parent_ctx->n_ctx,
        parent_ctx->params.n_batch,
        n_past,
        embd,
        context_full,
        ctx_sampling,
        parent_ctx->mtmd_wrapper->bitmap_past_hashes,
        0,  // Use sequence ID 0 for non-parallel mode
        recover,
        capture,
        invalidate
    );
}

} // namespace rnllama
