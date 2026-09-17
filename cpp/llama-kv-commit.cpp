#include "llama-kv-commit.h"
#include "llama-hybrid-commit.h"
#include "llama-kv-cache.h"
#include "llama-kv-staged.h"
#include "llama-context.h"
#include "llama-memory-hybrid.h"
#include "llama-impl.h"
#include <cstdint>
#include <vector>
struct llama_kv_commit_access {
    using range_t = std::pair<uint32_t, uint32_t>;
    static bool compatible(const llama_kv_cache & dst, const llama_kv_cache & src) {
        if (dst.get_size() != src.get_size() || dst.cache_type_k != src.cache_type_k ||
            dst.cache_type_v != src.cache_type_v || dst.n_seq_max != src.n_seq_max ||
            dst.n_stream != src.n_stream || dst.n_pad != src.n_pad ||
            dst.n_swa != src.n_swa || dst.swa_type != src.swa_type || dst.v_trans != src.v_trans ||
            dst.hparams.n_layer_all != src.hparams.n_layer_all || dst.layers.size() != src.layers.size()) return false;
        for (size_t i = 0; i < dst.layers.size(); ++i) {
            const auto & d = dst.layers[i];
            const auto & s = src.layers[i];
            if (d.il != s.il || !d.k || !s.k || d.k->ne[0] != s.k->ne[0] || d.k->ne[1] != s.k->ne[1] ||
                (d.v == nullptr) != (s.v == nullptr) || (d.v && (d.v->ne[0] != s.v->ne[0] || d.v->ne[1] != s.v->ne[1]))) return false;
        }
        return true;
    }
    static bool same_host_tensor(const lm_ggml_tensor * dst, const lm_ggml_tensor * src) {
        if (!dst || !src) {
            return dst == src;
        }
        return dst->buffer && dst->buffer == src->buffer && dst->data == src->data &&
               lm_ggml_backend_buffer_is_host(dst->buffer);
    }
    static bool aliases(const llama_kv_cache & dst, const llama_kv_cache & src) {
        if (dst.layers.size() != src.layers.size()) { return false; }
        for (size_t i = 0; i < dst.layers.size(); ++i) {
            if (dst.layers[i].il != src.layers[i].il ||
                !same_host_tensor(dst.layers[i].k, src.layers[i].k) ||
                !same_host_tensor(dst.layers[i].v, src.layers[i].v)) {
                return false;
            }
        }
        return true;
    }
    static bool has_non_host_v(const llama_kv_cache & cache) {
        for (const auto & layer : cache.layers) {
            for (const auto * v : layer.v_stream) {
                if (v && v->buffer && !lm_ggml_backend_buffer_is_host(v->buffer)) {
                    return true;
                }
            }
        }
        return false;
    }
    static bool can_stage_v(const llama_kv_cache & cache) {
        return cache.v_trans && has_non_host_v(cache);
    }
    static bool shares_cells(const llama_kv_cache & dst, const llama_kv_cache & src) {
        return dst.v_cells_impl == src.v_cells_impl &&
               (dst.other == &src || src.other == &dst || &dst == &src);
    }
    static llama_kv_route_cost route_cost(const llama_kv_cache & cache) {
        llama_kv_route_cost cost{};
        for (const auto & layer : cache.layers) {
            if (!layer.v) {
                continue;
            }
            ++cost.v_layers;
            cost.v_naive_gets += cache.hparams.n_embd_v_gqa(layer.il);
            cost.v_staged_gets += cache.v_trans ? 1 : cache.hparams.n_embd_v_gqa(layer.il);
        }
        return cost;
    }
    static bool collect_ranges(const llama_kv_cache & src, llama_seq_id seq_id,
                               llama_pos p0, llama_pos p1,
                               std::vector<range_t> & ranges) {
        if (seq_id < 0 || static_cast<size_t>(seq_id) >= src.seq_to_stream.size() || p0 < 0 || p1 < p0) {
            return false;
        }
        const auto & cells = src.v_cells[src.seq_to_stream[seq_id]];
        uint32_t begin = cells.size();
        for (uint32_t i = 0; i < cells.size(); ++i) {
            const bool selected = !cells.is_empty(i) && cells.seq_has(i, seq_id) && cells.pos_in(i, p0, p1);
            if (selected) {
                if (begin == cells.size()) {
                    begin = i;
                }
            } else if (begin != cells.size()) {
                ranges.emplace_back(begin, i);
                begin = cells.size();
            }
        }
        if (begin != cells.size()) {
            ranges.emplace_back(begin, cells.size());
        }
        return true;
    }
    static bool copy_tensor(const lm_ggml_tensor * src, lm_ggml_tensor * dst,
                            size_t src_offset, size_t dst_offset, size_t size,
                            std::vector<uint8_t> & scratch, size_t * copied_bytes,
                            llama_kv_commit_stats * stats, bool is_v) {
        if (!src || !dst) {
            return src == dst;
        }
        scratch.resize(size);
        lm_ggml_backend_tensor_get(src, scratch.data(), src_offset, size);
        if (stats) {
            if (is_v) {
                ++stats->transfers_naive;
                ++stats->v_gets_naive;
            } else {
                ++stats->transfers_k;
            }
        }
        lm_ggml_backend_tensor_set(dst, scratch.data(), dst_offset, size);
        if (stats) {
            if (is_v) {
                ++stats->transfers_naive;
                ++stats->v_sets_naive;
            } else {
                ++stats->transfers_k;
            }
        }
        if (copied_bytes) *copied_bytes += size;
        return true;
    }
    static bool copy_data(const llama_kv_cache & dst, const llama_kv_cache & src,
                          uint32_t dst_stream, uint32_t src_stream,
                          const std::vector<range_t> & ranges, size_t * copied_bytes,
                          bool copy_v, llama_kv_commit_stats * stats) {
        std::vector<uint8_t> scratch;
        // Mirrored from llama_kv_cache::state_write_data,
        // src/llama-kv-cache.cpp:2156-2253: K rows use range.first * row_size;
        // transposed V uses (range.first + j * cells.size()) * element_size.
        for (size_t i = 0; i < src.layers.size(); ++i) {
            const auto & src_layer = src.layers[i];
            const auto & dst_layer = dst.layers[i];
            if (src_layer.il != dst_layer.il || src_layer.k_stream.size() <= src_stream ||
                dst_layer.k_stream.size() <= dst_stream) {
                return false;
            }
            const auto * src_k = src_layer.k_stream[src_stream];
            auto * dst_k = dst_layer.k_stream[dst_stream];
            const size_t k_size_row = lm_ggml_row_size(src_k->type, src.hparams.n_embd_k_gqa(src_layer.il));
            for (const auto & range : ranges) {
                const size_t range_size = range.second - range.first;
                if (!copy_tensor(src_k, dst_k, range.first * k_size_row, range.first * k_size_row,
                                 range_size * k_size_row, scratch, copied_bytes, stats, false)) {
                    return false;
                }
            }
        }
        if (!copy_v) {
            return true;
        }
        const auto & src_cells = src.v_cells[src_stream];
        if (!src.v_trans) {
            for (size_t i = 0; i < src.layers.size(); ++i) {
                const auto & src_layer = src.layers[i];
                const auto & dst_layer = dst.layers[i];
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
                const size_t v_size_row = lm_ggml_row_size(src_v->type, src.hparams.n_embd_v_gqa(src_layer.il));
                for (const auto & range : ranges) {
                    const size_t range_size = range.second - range.first;
                    if (!copy_tensor(src_v, dst_v, range.first * v_size_row, range.first * v_size_row,
                                     range_size * v_size_row, scratch, copied_bytes, stats, true)) {
                        return false;
                    }
                }
            }
        } else {
            const uint32_t kv_size = src_cells.size();
            for (size_t i = 0; i < src.layers.size(); ++i) {
                const auto & src_layer = src.layers[i];
                const auto & dst_layer = dst.layers[i];
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
                const size_t v_size_el = lm_ggml_type_size(src_v->type);
                const uint32_t n_embd_v_gqa = src.hparams.n_embd_v_gqa(src_layer.il);
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    for (const auto & range : ranges) {
                        const size_t range_size = range.second - range.first;
                        const size_t offset = (range.first + j * kv_size) * v_size_el;
                        if (!copy_tensor(src_v, dst_v, offset, offset, range_size * v_size_el,
                                         scratch, copied_bytes, stats, true)) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
    static bool mirror(llama_kv_cache & dst, const llama_kv_cache & src,
                       llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                       llama_kv_commit_mode mode, lm_ggml_backend_sched_t src_sched,
                       size_t * copied_bytes, llama_kv_commit_stats * stats) {
        std::vector<range_t> ranges;
        if (!collect_ranges(src, seq_id, p0, p1, ranges)) {
            return false;
        }
        if (ranges.empty()) { return true; }
        const uint32_t src_stream = src.seq_to_stream[seq_id];
        const uint32_t dst_stream = dst.seq_to_stream[seq_id];
        auto & dst_cells = dst.v_cells[dst_stream];
        const auto & src_cells = src.v_cells[src_stream];
        const bool shared_cells = llama_kv_commit_access::shares_cells(dst, src);
        if (!shared_cells) {
            for (const auto & range : ranges) {
                for (uint32_t i = range.first; i < range.second; ++i) {
                    if (src_cells.get_shift(i) != 0 ||
                        (!dst_cells.is_empty(i) &&
                         (!dst_cells.seq_has(i, seq_id) || dst_cells.pos_get(i) != src_cells.pos_get(i)))) {
                        LLAMA_LOG_ERROR("%s: destination cell collision or shifted source cell at %u\n", __func__, i);
                        return false;
                    }
                }
            }
        }
        const bool staged_v = mode == llama_kv_commit_mode::Staged && src.v_trans;
        if (!copy_data(dst, src, dst_stream, src_stream, ranges, copied_bytes, !staged_v, stats)) {
            return false;
        }
        if (staged_v && !llama_kv_copy_v_staged(dst, src, src_sched, src_stream, dst_stream,
                                                ranges, copied_bytes, stats)) {
            return false;
        }
        if (!shared_cells) {
            for (const auto & range : ranges) {
                for (uint32_t i = range.first; i < range.second; ++i) {
                    if (!dst_cells.is_empty(i)) {
                        dst_cells.rm(i);
                    }
                    dst_cells.set(i, src_cells.cp(i, 1));
                }
            }
            dst.v_heads[dst_stream] = 0;
            while (dst.v_heads[dst_stream] < dst_cells.size() && !dst_cells.is_empty(dst.v_heads[dst_stream])) {
                ++dst.v_heads[dst_stream];
            }
            if (dst.v_heads[dst_stream] == dst_cells.size()) {
                dst.v_heads[dst_stream] = 0;
            }
        }
        return true;
    }
};
static const llama_kv_cache * get_cache(const llama_context * ctx) {
    if (!ctx) return nullptr;
    auto * mem = llama_get_memory(ctx);
    if (auto * cache = dynamic_cast<const llama_kv_cache *>(mem)) return cache;
    if (auto * hybrid = dynamic_cast<const llama_memory_hybrid *>(mem)) return hybrid->get_mem_attn();
    return nullptr;
}
static llama_kv_cache * get_cache(llama_context * ctx) {
    if (!ctx) return nullptr;
    auto * mem = llama_get_memory(ctx);
    if (auto * cache = dynamic_cast<llama_kv_cache *>(mem)) return cache;
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) return hybrid->get_mem_attn();
    return nullptr;
}
bool llama_kv_context_has_non_host_v(const llama_context * ctx) {
    const auto * cache = get_cache(ctx);
    return cache && llama_kv_commit_access::has_non_host_v(*cache);
}
bool llama_kv_context_can_stage_v(const llama_context * ctx) {
    const auto * cache = get_cache(ctx);
    return cache && llama_kv_commit_access::can_stage_v(*cache);
}
static const llama_memory_hybrid * get_hybrid(const llama_context * ctx) { return ctx ? dynamic_cast<const llama_memory_hybrid *>(llama_get_memory(ctx)) : nullptr; }
static llama_memory_hybrid * get_hybrid(llama_context * ctx) { return ctx ? dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx)) : nullptr; }
llama_kv_route llama_kv_route_query(const llama_context * dst_ctx, const llama_context * src_ctx) {
    const auto * dst_hybrid = get_hybrid(dst_ctx);
    const auto * src_hybrid = get_hybrid(src_ctx);
    const auto * dst = get_cache(dst_ctx);
    const auto * src = get_cache(src_ctx);
    if ((dst_hybrid == nullptr) != (src_hybrid == nullptr) || !dst || !src || !llama_kv_commit_access::compatible(*dst, *src) ||
        (dst_hybrid && !llama_hybrid_memory_compatible(*dst_hybrid, *src_hybrid))) {
        if (!dst || !src) {
            LLAMA_LOG_ERROR("%s: unsupported or missing memory type for KV commit\n", __func__);
        }
        return llama_kv_route::Reject;
    }
    if (dst_hybrid) {
        return llama_kv_route::MirrorAndCopy;
    }
    if (llama_kv_commit_access::aliases(*dst, *src)) {
        if (!llama_kv_commit_access::shares_cells(*dst, *src)) {
            LLAMA_LOG_ERROR("%s: direct route requires shared KV cell metadata\n", __func__);
            return llama_kv_route::Reject;
        }
        return llama_kv_route::Direct;
    }
    return llama_kv_route::MirrorAndCopy;
}
bool llama_kv_route_query_cost(const llama_context * dst_ctx, const llama_context * src_ctx,
                               llama_kv_route_cost * cost) {
    if (cost) *cost = {};
    const auto route = llama_kv_route_query(dst_ctx, src_ctx);
    if (route == llama_kv_route::Reject) {
        return false;
    }
    if (route == llama_kv_route::Direct || !cost) {
        return true;
    }
    const auto * src = get_cache(src_ctx);
    if (!src) {
        return false;
    }
    *cost = llama_kv_commit_access::route_cost(*src);
    return true;
}
bool llama_kv_commit_cache_range(llama_kv_cache * dst, const llama_kv_cache * src,
                                 llama_kv_commit_mode mode, lm_ggml_backend_sched_t src_sched,
                                 llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                                 size_t * copied_bytes, llama_kv_commit_stats * stats) {
    if (!dst || !src || !llama_kv_commit_access::compatible(*dst, *src)) {
        return false;
    }
    return llama_kv_commit_access::mirror(*dst, *src, seq_id, p0, p1, mode, src_sched, copied_bytes, stats);
}
// A mid-commit failure may leave partial tensor data with stale destination metadata;
// the prevalidated copies are idempotent, so retrying the same range is safe.
// p1 must be explicit and greater than p0; unlike seq_rm, p1 == -1 is not an end sentinel.
// lm_ggml_backend_tensor_get/set return void, so mid-copy I/O failures are undetectable (known limitation).
bool llama_kv_commit_with_stats(llama_context * dst_ctx, llama_context * src_ctx,
                                llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                                size_t * copied_bytes, llama_kv_commit_mode mode,
                                llama_kv_commit_stats * stats) {
    if (copied_bytes) *copied_bytes = 0;
    if (stats) *stats = {};
    auto * dst = get_cache(dst_ctx);
    const auto * src = get_cache(src_ctx);
    if (!dst || !src) {
        LLAMA_LOG_ERROR("%s: commit requires compatible KV caches\n", __func__);
        return false;
    }
    const auto route = llama_kv_route_query(dst_ctx, src_ctx);
    if (route == llama_kv_route::Reject) {
        LLAMA_LOG_ERROR("%s: incompatible KV cache route\n", __func__);
        return false;
    }
    llama_synchronize(src_ctx);
    llama_synchronize(dst_ctx);
    if (route == llama_kv_route::Direct) {
        LLAMA_LOG_INFO("%s: direct route already aliases K/V tensors; no copy\n", __func__);
        return true;
    }
    if (auto * dst_hybrid = get_hybrid(dst_ctx)) {
        return llama_hybrid_commit(*dst_hybrid, *get_hybrid(src_ctx), seq_id, p0, p1,
                                  mode, src_ctx->get_sched(), copied_bytes, stats);
    }
    return llama_kv_commit_cache_range(dst, src, mode, src_ctx->get_sched(), seq_id, p0, p1, copied_bytes, stats);
}
bool llama_kv_commit_ex(llama_context * dst_ctx, llama_context * src_ctx,
                        llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                        llama_kv_commit_mode mode, llama_kv_commit_stats * stats) {
    size_t copied_bytes = 0;
    const bool ok = llama_kv_commit_with_stats(dst_ctx, src_ctx, seq_id, p0, p1,
                                               &copied_bytes, mode, stats);
    return ok;
}
bool llama_kv_commit(llama_context * dst_ctx, llama_context * src_ctx,
                     llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return llama_kv_commit_ex(dst_ctx, src_ctx, seq_id, p0, p1, llama_kv_commit_mode::Naive, nullptr);
}
