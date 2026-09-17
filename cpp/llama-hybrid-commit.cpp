#include "llama-hybrid-commit.h"

#include "llama-kv-commit.h"
#include "llama-memory-hybrid.h"

#include <cstddef>
#include <cstdint>
#include <vector>

static bool compatible_tensor(const lm_ggml_tensor * dst, const lm_ggml_tensor * src) {
    if ((dst == nullptr) != (src == nullptr)) {
        return false;
    }
    if (!dst) {
        return true;
    }
    for (int i = 0; i < LM_GGML_MAX_DIMS; ++i) {
        if (dst->ne[i] != src->ne[i]) {
            return false;
        }
    }
    return dst->type == src->type && lm_ggml_nbytes(dst) == lm_ggml_nbytes(src);
}

static bool compatible_states(const llama_memory_recurrent & dst, const llama_memory_recurrent & src) {
    if (dst.size != src.size || dst.n_rs_seq != src.n_rs_seq || dst.rs_idx.size() != src.rs_idx.size() ||
        dst.r_l.size() != src.r_l.size() || dst.s_l.size() != src.s_l.size()) {
        return false;
    }
    for (size_t i = 0; i < dst.r_l.size(); ++i) {
        if (!compatible_tensor(dst.r_l[i], src.r_l[i]) || !compatible_tensor(dst.s_l[i], src.s_l[i])) {
            return false;
        }
    }
    return true;
}

bool llama_hybrid_memory_compatible(const llama_memory_hybrid & dst, const llama_memory_hybrid & src) {
    return compatible_states(*dst.get_mem_recr(), *src.get_mem_recr());
}

static bool copy_tensor(const lm_ggml_tensor * src, lm_ggml_tensor * dst,
                        std::vector<uint8_t> & scratch, size_t * copied_bytes) {
    if (!src || !dst) {
        return src == dst;
    }
    const size_t size = lm_ggml_nbytes(src);
    scratch.resize(size);
    lm_ggml_backend_tensor_get(src, scratch.data(), 0, size);
    lm_ggml_backend_tensor_set(dst, scratch.data(), 0, size);
    if (copied_bytes) {
        *copied_bytes += size;
    }
    return true;
}

static bool copy_recurrent(llama_memory_recurrent & dst, const llama_memory_recurrent & src,
                           size_t * copied_bytes) {
    std::vector<uint8_t> scratch;
    // Mirrors llama_memory_recurrent::state_write_data/state_read_data:
    // src/llama-memory-recurrent.cpp:861-939 writes contiguous R/S rows;
    // :1041-1115 restores those rows with the same ggml tensor offsets.
    for (size_t i = 0; i < src.r_l.size(); ++i) {
        if (!copy_tensor(src.r_l[i], dst.r_l[i], scratch, copied_bytes) ||
            !copy_tensor(src.s_l[i], dst.s_l[i], scratch, copied_bytes)) {
            return false;
        }
    }
    dst.cells = src.cells;
    dst.head = src.head;
    dst.used = src.used;
    dst.rs_idx = src.rs_idx;
    dst.n = src.n;
    dst.rs_z = src.rs_z;
    return true;
}

bool llama_hybrid_commit(llama_memory_hybrid & dst, const llama_memory_hybrid & src,
                         llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                         llama_kv_commit_mode mode, lm_ggml_backend_sched_t src_sched,
                         size_t * copied_bytes, llama_kv_commit_stats * stats) {
    if (!llama_kv_commit_cache_range(dst.get_mem_attn(), src.get_mem_attn(), mode, src_sched,
                                     seq_id, p0, p1, copied_bytes, stats)) {
        return false;
    }
    return copy_recurrent(*dst.get_mem_recr(), *src.get_mem_recr(), copied_bytes);
}
