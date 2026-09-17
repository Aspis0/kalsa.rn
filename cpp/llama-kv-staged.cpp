#include "llama-kv-staged.h"

#include "llama-impl.h"

#include <cstddef>
#include <cstdint>
#include <vector>

size_t llama_kv_staged_access::graph_count(const llama_kv_cache & src) {
    return src.staged_plan ? src.staged_plan->graphs.size() : 0;
}

llama_kv_staged_graph * llama_kv_staged_access::build_graph(
        llama_kv_staged_plan & plan,
        const llama_kv_cache & src,
        lm_ggml_backend_sched_t src_sched,
        uint32_t src_stream,
        uint32_t layer,
        uint32_t first,
        uint32_t count) {
    llama_kv_staged_key key { layer, src_stream, first, count };
    if (auto it = plan.graphs.find(key); it != plan.graphs.end()) {
        return &it->second;
    }
    const auto * src_v = src.layers[layer].v_stream[src_stream];
    const uint32_t n_embd_v = src.hparams.n_embd_v_gqa(src.layers[layer].il);
    const size_t element_size = lm_ggml_type_size(src_v->type);
    const size_t kv_size = src.v_cells[src_stream].size();
    lm_ggml_backend_buffer_type_t buft = lm_ggml_backend_buffer_get_type(src_v->buffer);
    lm_ggml_backend_t backend = lm_ggml_backend_sched_get_tensor_backend(src_sched, const_cast<lm_ggml_tensor *>(src_v));
    if (!backend) {
        const lm_ggml_backend_dev_t device = lm_ggml_backend_buft_get_device(buft);
        for (int i = 0; i < lm_ggml_backend_sched_get_n_backends(src_sched); ++i) {
            lm_ggml_backend_t candidate = lm_ggml_backend_sched_get_backend(src_sched, i);
            if (candidate && device && lm_ggml_backend_get_device(candidate) == device) {
                backend = candidate;
                break;
            }
        }
    }
    if (!backend) {
        for (int i = 0; i < lm_ggml_backend_sched_get_n_backends(src_sched); ++i) {
            lm_ggml_backend_t candidate = lm_ggml_backend_sched_get_backend(src_sched, i);
            if (candidate && lm_ggml_backend_supports_buft(candidate, buft)) {
                backend = candidate;
                break;
            }
        }
    }
    if (!backend) {
        LLAMA_LOG_ERROR("%s: source backend is unavailable for staged V copy\n", __func__);
        return nullptr;
    }

    lm_ggml_init_params params = {
        /*.mem_size   =*/ 4 * lm_ggml_tensor_overhead() + lm_ggml_graph_overhead_custom(4, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    lm_ggml_context_ptr ctx { lm_ggml_init(params) };
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: failed to create staged V graph context\n", __func__);
        return nullptr;
    }

    lm_ggml_tensor * src_view = lm_ggml_view_2d(ctx.get(), const_cast<lm_ggml_tensor *>(src_v), count, n_embd_v,
                                          kv_size * element_size, first * element_size);
    lm_ggml_tensor * staging = lm_ggml_new_tensor_2d(ctx.get(), src_v->type, count, n_embd_v);
    lm_ggml_tensor * copy = lm_ggml_cpy(ctx.get(), src_view, staging);
    lm_ggml_cgraph * graph = lm_ggml_new_graph_custom(ctx.get(), 4, false);
    lm_ggml_build_forward_expand(graph, copy);

    lm_ggml_backend_buffer_ptr buffer { lm_ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft) };
    if (!buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate staged V buffer\n", __func__);
        return nullptr;
    }

    auto [it, inserted] = plan.graphs.emplace(key, llama_kv_staged_graph{});
    LM_GGML_ASSERT(inserted);
    it->second.ctx = std::move(ctx);
    it->second.buffer = std::move(buffer);
    it->second.backend = backend;
    it->second.graph = graph;
    it->second.staging = staging;
    return &it->second;
}

bool llama_kv_staged_access::copy(
        llama_kv_cache & dst,
        const llama_kv_cache & src,
        lm_ggml_backend_sched_t src_sched,
        uint32_t src_stream,
        uint32_t dst_stream,
        const std::vector<std::pair<uint32_t, uint32_t>> & ranges,
        size_t * copied_bytes,
        llama_kv_commit_stats * stats) {
    if (!src.v_trans) {
        return false;
    }
    if (!src.staged_plan) {
        src.staged_plan = std::make_unique<llama_kv_staged_plan>();
    }
    // Governor ranges advance between commits; retaining device staging buffers
    // would grow with the conversation. Clear each commit so map erasure drops
    // lm_ggml_backend_buffer_ptr and its lm_ggml_backend_buffer_free deleter.
    struct plan_cleanup {
        llama_kv_staged_plan & plan;
        ~plan_cleanup() { plan.graphs.clear(); }
    } cleanup { *src.staged_plan };

    for (uint32_t layer = 0; layer < src.layers.size(); ++layer) {
        const auto & src_layer = src.layers[layer];
        const auto & dst_layer = dst.layers[layer];
        if (src_layer.v_stream.size() <= src_stream || dst_layer.v_stream.size() <= dst_stream) {
            return false;
        }
        const auto * src_v = src_layer.v_stream[src_stream];
        auto * dst_v = dst_layer.v_stream[dst_stream];
        if (!src_v || !dst_v) {
            if (src_v != dst_v) {
                return false;
            }
            continue;
        }
        if (lm_ggml_blck_size(src_v->type) != 1) {
            LLAMA_LOG_ERROR("%s: staged V copy requires an element-addressable V type\n", __func__);
            return false;
        }

        const size_t element_size = lm_ggml_type_size(src_v->type);
        const size_t kv_size = src.v_cells[src_stream].size();
        const uint32_t n_embd_v = src.hparams.n_embd_v_gqa(src_layer.il);
        for (const auto & range : ranges) {
            const uint32_t first = range.first;
            const uint32_t count = range.second - range.first;
            if (count == 0) {
                continue;
            }
            const llama_kv_staged_key key { layer, src_stream, first, count };
            const bool graph_new = src.staged_plan->graphs.find(key) == src.staged_plan->graphs.end();
            auto * staged = build_graph(*src.staged_plan, src, src_sched, src_stream, layer, first, count);
            if (!staged || lm_ggml_backend_graph_compute(staged->backend, staged->graph) != LM_GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: staged V graph execution failed\n", __func__);
                return false;
            }
            if (stats) {
                if (graph_new) {
                    stats->staged_graph_builds++;
                }
                stats->transfers_staged++;
                stats->v_gets_staged++;
            }
            std::vector<uint8_t> scratch(lm_ggml_nbytes(staged->staging));
            lm_ggml_backend_tensor_get(staged->staging, scratch.data(), 0, scratch.size());
            for (uint32_t j = 0; j < n_embd_v; ++j) {
                const size_t offset = (first + j * kv_size) * element_size;
                lm_ggml_backend_tensor_set(dst_v, scratch.data() + j * count * element_size,
                                        offset, count * element_size);
                if (stats) {
                    stats->transfers_staged++;
                    stats->v_sets_staged++;
                }
                if (copied_bytes) {
                    *copied_bytes += count * element_size;
                }
            }
        }
    }
    return true;
}

bool llama_kv_copy_v_staged(
        llama_kv_cache & dst,
        const llama_kv_cache & src,
        lm_ggml_backend_sched_t src_sched,
        uint32_t src_stream,
        uint32_t dst_stream,
        const std::vector<std::pair<uint32_t, uint32_t>> & ranges,
        size_t * copied_bytes,
        llama_kv_commit_stats * stats) {
    return llama_kv_staged_access::copy(dst, src, src_sched, src_stream, dst_stream, ranges, copied_bytes, stats);
}
