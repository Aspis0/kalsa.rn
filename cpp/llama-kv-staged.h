#pragma once

#include "llama-ext.h"
#include "llama-kv-cache.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

struct llama_kv_staged_key {
    uint32_t layer;
    uint32_t stream;
    uint32_t first;
    uint32_t count;

    bool operator<(const llama_kv_staged_key & other) const {
        return std::tie(layer, stream, first, count) < std::tie(other.layer, other.stream, other.first, other.count);
    }
};

struct llama_kv_staged_graph {
    lm_ggml_context_ptr ctx;
    lm_ggml_backend_buffer_ptr buffer;
    lm_ggml_backend_t backend = nullptr;
    lm_ggml_cgraph * graph = nullptr;
    lm_ggml_tensor * staging = nullptr;
};

struct llama_kv_staged_plan {
    std::map<llama_kv_staged_key, llama_kv_staged_graph> graphs;
};

struct llama_kv_staged_access {
    static size_t graph_count(const llama_kv_cache & src);

    static llama_kv_staged_graph * build_graph(
            llama_kv_staged_plan & plan,
            const llama_kv_cache & src,
            lm_ggml_backend_sched_t src_sched,
            uint32_t src_stream,
            uint32_t layer,
            uint32_t first,
            uint32_t count);

    static bool copy(
            llama_kv_cache & dst,
            const llama_kv_cache & src,
            lm_ggml_backend_sched_t src_sched,
            uint32_t src_stream,
            uint32_t dst_stream,
            const std::vector<std::pair<uint32_t, uint32_t>> & ranges,
            size_t * copied_bytes,
            llama_kv_commit_stats * stats);
};

bool llama_kv_copy_v_staged(
        llama_kv_cache & dst,
        const llama_kv_cache & src,
        lm_ggml_backend_sched_t src_sched,
        uint32_t src_stream,
        uint32_t dst_stream,
        const std::vector<std::pair<uint32_t, uint32_t>> & ranges,
        size_t * copied_bytes,
        llama_kv_commit_stats * stats);
