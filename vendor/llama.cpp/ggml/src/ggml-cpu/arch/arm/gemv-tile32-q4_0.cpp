// Batch-1 gemv over Q4_0 weights in the ggml Q4_0 32x32 tile layout, one row of a q8_0
// activation: the CPU decode path that reads the weight copy the HTP reads (one-copy mode).
// Ported from kalsallama-cpu-gemv spike/onecopy-cpu-gemv 4e6c75252
// (arch/arm/gemv-tiled-q4_0.cpp); the tile geometry now comes from ggml-q4_0-tile32.h.
//
// A tile's 32 rows are contiguous and K is the strided axis, so a dotprod lane (4 consecutive
// K of one row) has to be rebuilt with two zip levels per K pair. What that buys: 32 rows share
// one activation broadcast per K group, and the per-row block scale is folded once per 32 rows
// instead of once per row.

#include "gemv-tile32-q4_0.h"

#include "ggml-q4_0-tile32.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "ggml-impl.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>

// Accumulate one 32-K block: 8 groups of 4 K, each built from the tile's K pair bytes 2g and
// 2g + 1. p[r] collects the block's int32 partial dot of 4 rows, exact because the nibbles are
// 0..15 and the activations int8.
static inline void tile32_dot_block(const uint8_t * GGML_RESTRICT tile,
                                    const block_q8_0 * GGML_RESTRICT y,
                                    int32x4_t * GGML_RESTRICT p) {
    const uint8x16_t m = vdupq_n_u8(0x0F);

    for (int g = 0; g < 8; ++g) {
        const uint8_t * pa = tile + (2 * g) * GGML_Q4_0_TILE32_ROWS;
        const uint8_t * pb = pa + GGML_Q4_0_TILE32_ROWS;
        // the 4 activation bytes of this K group, in every lane
        const int8x16_t av = vreinterpretq_s8_s32(vld1q_dup_s32((const int32_t *) (y->qs + 4 * g)));

        // byte zip: a 16-bit lane becomes the two K pair bytes of one row
        const uint8x16x2_t zl = vzipq_u8(vld1q_u8(pa), vld1q_u8(pb));            // rows 0..15
        const uint8x16x2_t zh = vzipq_u8(vld1q_u8(pa + 16), vld1q_u8(pb + 16));  // rows 16..31
        const uint8x16_t z[4] = { zl.val[0], zl.val[1], zh.val[0], zh.val[1] };

        for (int q = 0; q < 4; ++q) {
            const uint8x16_t lo = vandq_u8(z[q], m);     // K element 4g + 0 and 4g + 2
            const uint8x16_t hi = vshrq_n_u8(z[q], 4);   // K element 4g + 1 and 4g + 3
            const uint8x16x2_t w = vzipq_u8(lo, hi);     // lane bytes: 4g, 4g + 1, 4g + 2, 4g + 3
            p[2 * q]     = vdotq_s32(p[2 * q],     w.val[0], av); // rows 4q .. 4q + 3
            p[2 * q + 1] = vdotq_s32(p[2 * q + 1], w.val[1], av); // rows 4q + 4 .. 4q + 7
        }
    }
}

// Fold one 32-K block into the fp32 row accumulators: q4_0 nibbles are stored unsigned, so the
// -8 of every element becomes -8 * sum(activations) for the whole block, which all 32 rows share.
// The fold order (per K tile: exact int32 dot, fp32 scale product, one fused multiply-add) is
// what makes this kernel bit-identical to the CPU_REPACK gemm/gemv per output row.
static inline void tile32_scale_block(const int32x4_t * GGML_RESTRICT p,
                                      float32x4_t * GGML_RESTRICT acc,
                                      const uint8_t * GGML_RESTRICT tile,
                                      const block_q8_0 * GGML_RESTRICT y) {
    const int sum_a = vaddlvq_s8(vld1q_s8(y->qs)) + vaddlvq_s8(vld1q_s8(y->qs + 16));
    const int32x4_t off = vdupq_n_s32(8 * sum_a);
    const float32x4_t da = vcvt_f32_f16(vld1_dup_f16((const __fp16 *) &y->d));
    const ggml_half * dw = (const ggml_half *) (tile + GGML_Q4_0_TILE32_QUANT_BYTES);

    for (int r = 0; r < 8; ++r) {
        const float32x4_t s = vmulq_f32(vcvt_f32_f16(vld1_f16((const __fp16 *) (dw + 4 * r))), da);
        acc[r] = vfmaq_f32(acc[r], vcvtq_f32_s32(vsubq_s32(p[r], off)), s);
    }
}

void ggml_gemv_tile32_q4_0_q8_0(int n, float * s, size_t s_row,
                                const void * vw, const void * vy, int64_t nrows) {
    GGML_ASSERT(n > 0 && n % QK4_0 == 0);
    GGML_ASSERT(nrows > 0 && nrows % 32 == 0);

    const int n_kt = n / QK4_0;
    const block_q8_0 * y = (const block_q8_0 *) vy;
    const uint8_t * tiles = (const uint8_t *) vw;

    for (int64_t rt = 0; rt < nrows / 32; ++rt) {
        const uint8_t * tile = tiles + (size_t) rt * n_kt * GGML_Q4_0_TILE32_SIZE;
        float32x4_t acc[8];
        for (int r = 0; r < 8; ++r) {
            acc[r] = vdupq_n_f32(0.0f);
        }

        for (int kt = 0; kt < n_kt; ++kt) {
            int32x4_t p[8];
            for (int r = 0; r < 8; ++r) {
                p[r] = vdupq_n_s32(0);
            }
            tile32_dot_block(tile + (size_t) kt * GGML_Q4_0_TILE32_SIZE, y + kt, p);
            tile32_scale_block(p, acc, tile + (size_t) kt * GGML_Q4_0_TILE32_SIZE, y + kt);
        }

        // the spike stored each acc vector as 4 consecutive floats with a group stride; the ggml
        // mul_mat dst is row-major, so the 4 lanes scatter to their own rows instead
        for (int r = 0; r < 8; ++r) {
            float tmp[4];
            vst1q_f32(tmp, acc[r]);
            float * row0 = (float *) ((char *) s + (rt * 32 + 4 * r) * s_row * sizeof(float));
            row0[0]                = tmp[0];
            row0[s_row]            = tmp[1];
            row0[2 * (size_t) s_row] = tmp[2];
            row0[3 * (size_t) s_row] = tmp[3];
        }
    }
}

#else // the tiled layout has no stock-kernel fallback, so a non-dotprod build
      // must not silently produce numbers

void ggml_gemv_tile32_q4_0_q8_0(int n, float * s, size_t s_row,
                                const void * vw, const void * vy, int64_t nrows) {
    (void) n; (void) s; (void) s_row; (void) vw; (void) vy; (void) nrows;
    GGML_ABORT("gemv-tile32-q4_0 needs armv8.2-a+dotprod");
}

#endif // __aarch64__ && __ARM_FEATURE_DOTPROD
