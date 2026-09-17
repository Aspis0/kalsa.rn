#pragma once

#include "llama-ext.h"

class llama_kv_cache;

bool llama_kv_context_has_non_host_v(const llama_context * ctx);
// Exact predicate used by governor auto-selection: staged V requires transposed, device-resident V.
bool llama_kv_context_can_stage_v(const llama_context * ctx);

bool llama_kv_commit_cache_range(llama_kv_cache * dst, const llama_kv_cache * src,
                                 llama_kv_commit_mode mode, lm_ggml_backend_sched_t src_sched,
                                 llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                                 size_t * copied_bytes, llama_kv_commit_stats * stats);

// Internal companion used by the governor to account for the existing copies.
bool llama_kv_commit_with_stats(llama_context * dst_ctx, llama_context * src_ctx,
                                llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                                size_t * copied_bytes, llama_kv_commit_mode mode = llama_kv_commit_mode::Naive,
                                llama_kv_commit_stats * stats = nullptr);
