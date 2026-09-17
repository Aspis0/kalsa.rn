#pragma once

#include "JSIHelpers.h"
#include "JSINativeHeaders.h"
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace rnllama_jsi {

    inline jsi::Object createTokenizeResult(jsi::Runtime& runtime, const rnllama::llama_rn_tokenize_result& result) {
        jsi::Object res(runtime);
        
        jsi::Array tokens = jsi::Array(runtime, result.tokens.size());
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            tokens.setValueAtIndex(runtime, i, (double)result.tokens[i]);
        }
        res.setProperty(runtime, "tokens", tokens);
        
        res.setProperty(runtime, "has_media", result.has_media);
        
        jsi::Array hashes = jsi::Array(runtime, result.bitmap_hashes.size());
        for (size_t i = 0; i < result.bitmap_hashes.size(); ++i) {
            hashes.setValueAtIndex(runtime, i, jsi::String::createFromUtf8(runtime, result.bitmap_hashes[i]));
        }
        res.setProperty(runtime, "bitmap_hashes", hashes);
        
        jsi::Array chunk_pos = jsi::Array(runtime, result.chunk_pos.size());
        for (size_t i = 0; i < result.chunk_pos.size(); ++i) {
            chunk_pos.setValueAtIndex(runtime, i, (double)result.chunk_pos[i]);
        }
        res.setProperty(runtime, "chunk_pos", chunk_pos);
        
        jsi::Array chunk_pos_media = jsi::Array(runtime, result.chunk_pos_media.size());
        for (size_t i = 0; i < result.chunk_pos_media.size(); ++i) {
            chunk_pos_media.setValueAtIndex(runtime, i, (double)result.chunk_pos_media[i]);
        }
        res.setProperty(runtime, "chunk_pos_media", chunk_pos_media);

        return res;
    }

    inline jsi::Object loadSession(jsi::Runtime& runtime, rnllama::llama_rn_context* ctx, const std::string& path) {
        if (!ctx || !ctx->completion) {
            throw std::runtime_error("Context or completion not initialized");
        }
        if (ctx->hasGovernor()) {
            throw std::runtime_error("Session load is not supported while governor mode is enabled");
        }
        if (ctx->slot_manager != nullptr) {
            // llama_state_load_file restores every sequence at once, which
            // would corrupt in-flight parallel slots
            throw std::runtime_error("Session load is not supported while parallel mode is enabled");
        }

        auto& embd = ctx->completion->embd;

        size_t n_token_count_out = 0;
        embd.resize(llama_n_ctx(ctx->active_ctx()));
        if (!llama_state_load_file(ctx->active_ctx(), path.c_str(), embd.data(), embd.size(), &n_token_count_out)) {
             throw std::runtime_error("Failed to load session");
        }
        // Keep LLAMA_TOKEN_NULL media placeholders: they hold the positions of
        // media evaluated into the restored memory
        embd.resize(n_token_count_out);

        // The restored memory may hold more positions than the token list
        // (legacy files saved from multimodal sequences or trimmed saves).
        // Reconcile so the next completion can resume; degrade to an empty
        // cache when the memory cannot be rolled back. M-RoPE media histories
        // legitimately hold fewer time positions than placeholder tokens -
        // but only when placeholders are actually present; for a text-only
        // token list a lagging frontier can only mean an inconsistent file.
        auto * kv = llama_get_memory(ctx->active_ctx());
        const llama_pos n_tokens = (llama_pos) embd.size();
        const llama_pos pos_max = llama_memory_seq_pos_max(kv, 0);
        const bool mrope_media = rnllama::model_uses_mrope(ctx->model) &&
            std::find(embd.begin(), embd.end(), LLAMA_TOKEN_NULL) != embd.end();
        bool resumable = pos_max + 1 == n_tokens ||
                         (mrope_media && pos_max >= 0 && pos_max + 1 < n_tokens);
        // Flags used by the SWA check below; also logged so a wiped restore
        // shows whether hybrid/SWA shaped the decision.
        const bool is_recurrent = llama_model_is_recurrent(ctx->model);
        const bool is_hybrid = llama_model_is_hybrid(ctx->model);
        const bool is_recurrent_or_hybrid = is_recurrent || is_hybrid;
        const int32_t n_swa = ctx->params.swa_full ? 0 : llama_model_n_swa(ctx->model);
        if (!resumable && pos_max + 1 > n_tokens) {
            // Split for diagnostics only: same short-circuit as before.
            const bool seq_rm_ok = llama_memory_seq_rm(kv, 0, n_tokens, -1);
            resumable = seq_rm_ok &&
                        llama_memory_seq_pos_max(kv, 0) + 1 == n_tokens;
            rnllama::log("WARNING", __func__, __LINE__,
                "KALSA_KVRESUME_SEQ_RM seq_rm_ok=%d n_tokens=%d pos_max=%d",
                (int) seq_rm_ok, (int) n_tokens, (int) pos_max);
            if (resumable) {
                // SWA caches prune cells behind the attention window; after
                // rolling back, the window ending at n_tokens must be intact.
                // Recurrent/hybrid models are exempt (their pos_min reflects
                // the recurrent tail; seq_rm itself enforces their safety).
                if (n_swa > 0 && !is_recurrent_or_hybrid) {
                    const llama_pos pos_min = llama_memory_seq_pos_min(kv, 0);
                    resumable = pos_min == 0 ||
                                (pos_min > 0 && pos_min < std::max<llama_pos>(0, n_tokens - n_swa));
                }
            }
        }
        rnllama::log("WARNING", __func__, __LINE__,
            "KALSA_KVRESUME n_tokens=%d pos_max=%d mrope_media=%d is_recurrent=%d is_hybrid=%d n_swa=%d resumable=%d",
            (int) n_tokens, (int) pos_max, (int) mrope_media,
            (int) is_recurrent, (int) is_hybrid, (int) n_swa, (int) resumable);
        // Hybrid/recurrent: seq_rm of a longer tail fails, and a lagging
        // frontier (pos_max+1 < n_tokens, no mrope) is the same inconsistent
        // file. Do not return tokens_loaded:0 — JS would delete the .kvs.
        // Clear the just-loaded live ctx so no caller sees the polluted KV,
        // then throw; JS keeps the file.
        if (!resumable && is_recurrent_or_hybrid && n_tokens > 0) {
            rnllama::log("WARNING", __func__, __LINE__,
                "KALSA_KVRESUME n_tokens=%d pos_max=%d mrope_media=%d is_recurrent=%d is_hybrid=%d n_swa=%d resumable=0 kv_inconsistent=1",
                (int) n_tokens, (int) pos_max, (int) mrope_media,
                (int) is_recurrent, (int) is_hybrid, (int) n_swa);
            llama_memory_clear(kv, false);
            embd.clear();
            ctx->completion->clearStateCheckpoints();
            throw std::runtime_error("kv_inconsistent");
        }
        if (!resumable) {
            llama_memory_seq_rm(kv, 0, 0, -1);
            embd.clear();
        }

        // Kalsa patch #2: adopt the restored memory as a state checkpoint.
        // On recurrent/hybrid models (Qwen3.5) loadPrompt's llama_memory_seq_rm
        // fails, so it falls back to recoverStateCheckpoint — and in a FRESH
        // process state_checkpoints is empty, so the just-restored memory is
        // discarded and the whole prompt is re-prefilled ("no usable state
        // checkpoint (recurrent/hybrid/SWA), doing full cache clear"; measured
        // n_past=0 on a Xiaomi 14 and in CI run 31293842336 despite ok:true).
        // Registering it at embd.size() lets findStateCheckpoint adopt it
        // when n_common equals this length (next prompt extends this history).
        // Do not snapshot message boundaries here: capture dumps the *current*
        // recurrent tail, so a shorter label on a 2441-token KV is a lie
        // (S23 extract restore then t2 save n_tokens=2360 pos_max=2973).
        // Prefix reuse after a diverge uses checkpoints taken while the
        // frontier actually sat at that length (prewarm / prior turns).
        if (!embd.empty()) {
            ctx->completion->probeStateCache();
            ctx->completion->captureStateCheckpoint(embd, embd.size());
        }

        // Media identity for placeholder positions; absent for text-only or
        // legacy files (media is then conservatively reprocessed)
        ctx->setMediaHashes(embd.empty() ? std::vector<std::string>{}
                                         : rnllama::read_state_meta(path));

        // Placeholders are not vocab ids - drop them from the prompt string
        std::vector<llama_token> text_tokens;
        text_tokens.reserve(embd.size());
        std::copy_if(embd.begin(), embd.end(), std::back_inserter(text_tokens),
                     [](llama_token t) { return t != LLAMA_TOKEN_NULL; });
        const std::string text = rnllama::tokens_to_str(ctx->active_ctx(), text_tokens.cbegin(), text_tokens.cend());

        jsi::Object result(runtime);
        result.setProperty(runtime, "tokens_loaded", (double)embd.size());
        result.setProperty(runtime, "prompt", jsi::String::createFromUtf8(runtime, text));
        return result;
    }

    inline int saveSession(rnllama::llama_rn_context* ctx, const std::string& path, int size) {
        if (!ctx || !ctx->completion) {
            throw std::runtime_error("Context or completion not initialized");
        }
        if (ctx->hasGovernor()) {
            throw std::runtime_error("Session save is not supported while governor mode is enabled");
        }
        if (ctx->slot_manager != nullptr) {
            // The single-completion embd does not describe the parallel slots'
            // sequences; a whole-context save would be inconsistent
            throw std::runtime_error("Session save is not supported while parallel mode is enabled");
        }

        // Keep LLAMA_TOKEN_NULL media placeholders: llama_state_save_file
        // serializes the whole memory, so the token list must cover the same
        // positions or the file cannot be resumed
        std::vector<llama_token> session_tokens = ctx->completion->embd;

        int default_size = session_tokens.size();
        int save_size = size > 0 && size <= default_size ? size : default_size;

        // Match load resumable: exact frontier, or mrope-shorter only.
        // Longer memory is refused even with mrope placeholders. Empty
        // (n_tokens=0, pos_max=-1) is consistent: -1+1==0.
        auto * save_kv = llama_get_memory(ctx->active_ctx());
        const llama_pos save_pos_max = llama_memory_seq_pos_max(save_kv, 0);
        const bool save_mrope_media = rnllama::model_uses_mrope(ctx->model) &&
            std::find(session_tokens.begin(), session_tokens.end(), LLAMA_TOKEN_NULL)
                != session_tokens.end();
        const bool save_consistent =
            save_pos_max + 1 == (llama_pos) save_size ||
            (save_mrope_media && save_pos_max >= 0 &&
             save_pos_max + 1 < (llama_pos) save_size);
        if (!save_consistent) {
            rnllama::log("WARNING", __func__, __LINE__,
                "KALSA_KVRESUME save_refused kv_inconsistent=1 n_tokens=%d pos_max=%d",
                (int) save_size, (int) save_pos_max);
            throw std::runtime_error("kv_inconsistent");
        }

        // Drop any previous sidecar before overwriting the state file so a
        // failure in between can never pair stale hashes with the new file
        rnllama::write_state_meta(path, {});

        if (!llama_state_save_file(ctx->active_ctx(), path.c_str(), session_tokens.data(), save_size)) {
             throw std::runtime_error("Failed to save session");
        }

        // Persist media identity only when the saved prefix actually holds
        // media and no media position was cut off; anything else cannot be
        // verified on reload
        const bool media_retained =
            std::find(session_tokens.begin(), session_tokens.begin() + save_size,
                      LLAMA_TOKEN_NULL) != session_tokens.begin() + save_size &&
            std::find(session_tokens.begin() + save_size, session_tokens.end(),
                      LLAMA_TOKEN_NULL) == session_tokens.end();
        rnllama::write_state_meta(path, media_retained ? ctx->getMediaHashes()
                                                       : std::vector<std::string>{});

        return save_size;
    }
}
