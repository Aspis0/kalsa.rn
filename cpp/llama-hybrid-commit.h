#pragma once

#include "llama-ext.h"

#include <cstddef>

class llama_memory_hybrid;

bool llama_hybrid_memory_compatible(const llama_memory_hybrid & dst, const llama_memory_hybrid & src);
bool llama_hybrid_commit(llama_memory_hybrid & dst, const llama_memory_hybrid & src,
                         llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                         llama_kv_commit_mode mode, lm_ggml_backend_sched_t src_sched,
                         size_t * copied_bytes, llama_kv_commit_stats * stats);
