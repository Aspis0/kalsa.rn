// MUL_MAT over weights in a Q4_0 TILE32 buffer, correctness path: quantize the activation rows
// to q8_0 like repack.cpp forward_mul_mat, then run one gemv per 32-row weight tile, with the
// threads split over tiles. Each column re-reads the weights, so prefill (batch > 1) is far
// below the repacked GEMM; measuring that trade is e4, not this file. The gemv's per-tile fold
// (exact int32 dot, fp32 scale product, one fused multiply-add per K tile) keeps the output
// bit-identical to the CPU_REPACK kernels on a dotprod ARM host.

#include "tile32.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-q4_0-tile32.h"
#include "arch/arm/gemv-tile32-q4_0.h"

#include <mutex>
#include <set>
#include <utility>

// The one predicate: device supports_op, work_size and compute_forward must agree on exactly
// these shapes, or a declined op falls through to the native mul_mat and reads tiles as native
// rows (ggml_compute_forward_mul_mat aborts on that).
static bool ggml_cpu_tile32_op_supported(const struct ggml_tensor * op) {
    if (!ggml_cpu_has_dotprod()) {
        return false;
    }
    if (op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    // src1->buffer is NULL in the placement probes (weight_buft_supported builds unallocated
    // inputs); only src0's buffer carries the layout this dispatch exists for
    if (!src0 || !src1 || !src0->buffer) {
        return false;
    }
    if (ggml_backend_buft_weight_layout(src0->buffer->buft) != GGML_WEIGHT_LAYOUT_Q4_0_TILE32) {
        return false;
    }
    // a view of a TILE32 tensor has no tile-space address: ggml_backend_view_init computes
    // data = view_src->data + view_offs with a native byte offset (ggml-backend.cpp:2147), which
    // in the tiled layout lands mid-tile. A weight placed by ggml_backend_tensor_alloc keeps its
    // own tile base even when it shares the buffer with other weights, so view provenance is the
    // exact test, not data != buffer base.
    if (src0->view_src != NULL) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_0 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src0->ne[0] % GGML_Q4_0_TILE32_K != 0) {
        return false;
    }
    // correctness only: f32 activations with no outer planes in a host buffer (ne1 == 1 at
    // ubatch 1 counts as 2-D). src1->buffer is NULL in the placement probes
    // (weight_buft_supported builds unallocated inputs); at compute time the re-check declines
    // a non-host src1 into the native mul_mat abort.
    if (src1->type != GGML_TYPE_F32 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
        return false;
    }
    // the from_float loop at compute time reads every src1 row as ne10 consecutive floats
    if (src1->nb[0] != sizeof(float)) {
        return false;
    }
    // the gemv stores one output tile as consecutive floats per token row
    if (!ggml_is_contiguous(op)) {
        return false;
    }
    return true;
}

bool ggml_cpu_tile32_supports_op(const struct ggml_tensor * op) {
    return ggml_cpu_tile32_op_supported(op);
}

bool ggml_cpu_tile32_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    if (!ggml_cpu_tile32_op_supported(op)) {
        return false;
    }

    GGML_UNUSED(n_threads);
    const int64_t ne01 = op->src[0]->ne[1];
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne10 = op->src[1]->ne[0];
    const int64_t ne11 = op->src[1]->ne[1];

    // q8_0 activation rows, then a 32-float scratch for the partial tile (dst has no rows for
    // the padded ones, so the gemv cannot write them straight out)
    *size = ggml_row_size(GGML_TYPE_Q8_0, ne10) * ne11;
    if (ne01 % GGML_Q4_0_TILE32_ROWS != 0) {
        *size += GGML_PAD(GGML_Q4_0_TILE32_ROWS * sizeof(float), GGML_MEM_ALIGN);
    }
    return true;
}

bool ggml_cpu_tile32_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    if (!ggml_cpu_tile32_op_supported(op)) {
        // decline: the native-read guard in ggml_compute_forward aborts on the non-NATIVE layout,
        // so nothing prints here
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * dst  = op;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);

    // one proof line per weight tensor and batch width: decode logs once at ne11 == 1, each
    // prefill ubatch size once, never per token. Only thread 0 of the node enters this block,
    // so the mutex is off the hot path for the other threads; thread 0 always runs the node.
    // The key assumes model weights live for the process lifetime, so the data address
    // identifies the tensor (the test's short-lived buffers can recycle addresses and skip a
    // line - real weights cannot). The set stays tiny - cheap next to the gemv.
    if (params->ith == 0) {
        static std::mutex                                 log_mutex;
        static std::set<std::pair<const void *, int64_t>> logged;
        std::lock_guard<std::mutex> lock(log_mutex);
        if (logged.emplace(src0->data, ne11).second) {
            GGML_LOG_INFO("ggml-cpu: tile32 MUL_MAT tensor=%s ne=%lldx%lld ne11=%lld\n",
                          src0->name, (long long) ne00, (long long) ne01, (long long) ne11);
        }
    }

    const size_t row_q8  = ggml_row_size(GGML_TYPE_Q8_0, ne10);
    // the gemv's s_row strides consecutive weight-row outputs: nb0, not the token stride nb1
    const int64_t partial = ne01 % GGML_Q4_0_TILE32_ROWS;
    const size_t scratch_offset = GGML_PAD(row_q8 * ne11, GGML_MEM_ALIGN);
    const size_t tile_stride = ggml_q4_0_tile32_nbytes(ne00, GGML_Q4_0_TILE32_ROWS);

    GGML_ASSERT(params->wsize >= scratch_offset + (partial ? GGML_PAD(GGML_Q4_0_TILE32_ROWS * sizeof(float), GGML_MEM_ALIGN) : 0));

    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
        from_float((const float *) ((const char *) src1->data + i11 * nb11),
                   (char *) params->wdata + i11 * row_q8, ne10);
    }

    ggml_barrier(params->threadpool);

    const int64_t n_tiles = ne01 / GGML_Q4_0_TILE32_ROWS + (partial ? 1 : 0);
    float * scratch = partial ? (float *) ((char *) params->wdata + scratch_offset) : nullptr;

    for (int64_t t = ith; t < n_tiles; t += nth) {
        const uint8_t * tile = (const uint8_t *) src0->data + (size_t) t * tile_stride;
        const bool last_partial = partial && t == n_tiles - 1;

        for (int64_t i11 = 0; i11 < ne11; i11++) {
            const block_q8_0 * y = (const block_q8_0 *) ((const char *) params->wdata + i11 * row_q8);
            // token i11's outputs start at i11 * nb1; output (t*32 + r) sits nb0-strided inside
            float * rows = (float *) ((char *) dst->data + i11 * nb1) + t * GGML_Q4_0_TILE32_ROWS;

            if (last_partial) {
                // the buffer holds the padded rows, so the gemv runs on all 32; only the real
                // rows are stored
                ggml_gemv_tile32_q4_0_q8_0(ne00, scratch, 1, tile, y, GGML_Q4_0_TILE32_ROWS);
                memcpy(rows, scratch, partial * sizeof(float));
            } else {
                ggml_gemv_tile32_q4_0_q8_0(ne00, rows, 1, tile, y, GGML_Q4_0_TILE32_ROWS);
            }
        }
    }

    return true;
}
