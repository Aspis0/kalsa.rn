#ifndef GGML_Q4_0_TILE32_H
#define GGML_Q4_0_TILE32_H

// Q4_0 weight layout read by the Hexagon HTP matmul kernels (HTP_MM_WEIGHT_TILE_SIZE_Q4_0,
// htp/matmul-ops.h), kept here as a ggml contract so other backends can read the same bytes.
//
// Source rows are ggml Q4_0 rows (block_q4_0: one fp16 scale then 16 nibble bytes). A tile holds
// 32 consecutive rows of one matrix (row tiles are the major order) and 32 consecutive K elements
// (K tiles are the minor order), laid out as
//   [  0, 512)  16 K-pairs of 32 bytes: byte cp*32 + row carries element 2*cp in the low nibble
//               and element 2*cp + 1 in the high nibble
//   [512, 576)  the 32 fp16 row scales, in row order
// so tile (ct, kt) of a matrix starts at (ct * n_k_tiles + kt) * GGML_Q4_0_TILE32_SIZE. Rows and K
// are padded up to 32 with nibbles 8 (deviation 0) and scale 0, which contributes nothing to a dot
// product.
//
// ne0 must be a multiple of 32: a source row is an integral number of block_q4_0, the same
// requirement ggml_row_size enforces. ne1 is free and only pads. A tensor with ne2*ne3 matrices
// holds them back to back, each ggml_q4_0_tile32_nbytes(ne0, ne1) bytes.

#include "ggml-quants.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GGML_Q4_0_TILE32_ROWS        32
#define GGML_Q4_0_TILE32_K           32
#define GGML_Q4_0_TILE32_QUANT_BYTES 512
#define GGML_Q4_0_TILE32_SCALE_BYTES 64
#define GGML_Q4_0_TILE32_SIZE        (GGML_Q4_0_TILE32_QUANT_BYTES + GGML_Q4_0_TILE32_SCALE_BYTES)

static inline int64_t ggml_q4_0_tile32_round_up(int64_t n, int64_t m) {
    return (n + m - 1) / m * m;
}

static inline size_t ggml_q4_0_tile32_nbytes(int64_t ne0, int64_t ne1) {
    const int64_t n_row_tiles = ggml_q4_0_tile32_round_up(ne1, GGML_Q4_0_TILE32_ROWS) / GGML_Q4_0_TILE32_ROWS;
    const int64_t n_k_tiles   = ggml_q4_0_tile32_round_up(ne0, GGML_Q4_0_TILE32_K)    / GGML_Q4_0_TILE32_K;
    return (size_t) (n_row_tiles * n_k_tiles) * GGML_Q4_0_TILE32_SIZE;
}

// dst must hold n_slices * ggml_q4_0_tile32_nbytes(ne0, ne1) bytes. src points at the first slice
// of the [offset, offset + size) write range, not at slice 0 of the tensor: the ggml_backend_tensor_set
// paths either hand a whole tensor (offset 0) or a whole slice range, and the previous Hexagon
// implementation indexed src the same way.
static inline void ggml_q4_0_tile32_pack(uint8_t *              dst,
                                         const block_q4_0 *     src,
                                         int64_t                ne0,
                                         int64_t                ne1,
                                         int64_t                n_slices,
                                         size_t                 offset,
                                         size_t                 size) {
    assert(ne0 % GGML_Q4_0_TILE32_K == 0);

    const int64_t blocks_per_row = ne0 / QK4_0;
    const size_t  slice_size     = (size_t) ne1 * (size_t) blocks_per_row * sizeof(block_q4_0);
    const int     n_row_tiles    = (int) (ggml_q4_0_tile32_round_up(ne1, GGML_Q4_0_TILE32_ROWS) / GGML_Q4_0_TILE32_ROWS);
    const int     n_k_tiles      = (int) (ggml_q4_0_tile32_round_up(ne0, GGML_Q4_0_TILE32_K)    / GGML_Q4_0_TILE32_K);
    const size_t  matrix_size    = (size_t) n_row_tiles * (size_t) n_k_tiles * GGML_Q4_0_TILE32_SIZE;

    const int64_t start_slice = (int64_t) (offset / slice_size);
    int64_t       end_slice   = (int64_t) ((offset + size + slice_size - 1) / slice_size);
    if (end_slice > n_slices) {
        end_slice = n_slices;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_q4_0 * src_slice  = src + (slice_idx - start_slice) * (ne1 * blocks_per_row);
        uint8_t *          matrix_dst = dst + (size_t) slice_idx * matrix_size;

        for (int ct = 0; ct < n_row_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst +
                    ((size_t) ct * (size_t) n_k_tiles + (size_t) kt) * GGML_Q4_0_TILE32_SIZE;

                uint8_t tile_quants[GGML_Q4_0_TILE32_ROWS][GGML_Q4_0_TILE32_K];
                for (int row = 0; row < GGML_Q4_0_TILE32_ROWS; row++) {
                    const int64_t r = ct * GGML_Q4_0_TILE32_ROWS + row;
                    if (r < ne1 && kt < blocks_per_row) {
                        const block_q4_0 * b = &src_slice[r * blocks_per_row + kt];
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            tile_quants[row][j]             = (uint8_t) (b->qs[j] & 0x0F);
                            tile_quants[row][j + QK4_0 / 2] = (uint8_t) (b->qs[j] >> 4);
                        }
                    } else {
                        memset(tile_quants[row], 8, GGML_Q4_0_TILE32_K);
                    }
                }

                for (int cp = 0; cp < GGML_Q4_0_TILE32_K / 2; cp++) {
                    for (int row = 0; row < GGML_Q4_0_TILE32_ROWS; row++) {
                        tile_dst[cp * GGML_Q4_0_TILE32_ROWS + row] =
                            (uint8_t) ((tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp]);
                    }
                }

                ggml_half * scale_dst = (ggml_half *) (tile_dst + GGML_Q4_0_TILE32_QUANT_BYTES);
                for (int row = 0; row < GGML_Q4_0_TILE32_ROWS; row++) {
                    const int64_t r = ct * GGML_Q4_0_TILE32_ROWS + row;
                    scale_dst[row] = (r < ne1 && kt < blocks_per_row) ? src_slice[r * blocks_per_row + kt].d : 0;
                }
            }
        }
    }
}

// Inverse of the pack: tiles back into native block_q4_0 rows. dst points at the first native
// byte of the [offset, offset + size) read range, src at the tile tensor base (slice 0) - the
// mirror of the pack's pointing rules. TILE32 buffer types enforce whole-tensor or whole-slice
// ranges at their boundary (GGML_ABORT otherwise); the legacy Hexagon REPACK read paths also
// hand it row-aligned windows, so a range may start or end inside a slice: only the rows it
// covers are written, compactly from dst.
static inline void ggml_q4_0_tile32_unpack(block_q4_0 *    dst,
                                           const uint8_t * src,
                                           int64_t         ne0,
                                           int64_t         ne1,
                                           int64_t         n_slices,
                                           size_t          offset,
                                           size_t          size) {
    assert(ne0 % GGML_Q4_0_TILE32_K == 0);

    const int64_t blocks_per_row = ne0 / QK4_0;
    const size_t  row_size       = (size_t) blocks_per_row * sizeof(block_q4_0);
    const size_t  slice_size     = (size_t) ne1 * row_size;
    const int     n_row_tiles    = (int) (ggml_q4_0_tile32_round_up(ne1, GGML_Q4_0_TILE32_ROWS) / GGML_Q4_0_TILE32_ROWS);
    const int     n_k_tiles      = (int) (ggml_q4_0_tile32_round_up(ne0, GGML_Q4_0_TILE32_K)    / GGML_Q4_0_TILE32_K);
    const size_t  matrix_size    = (size_t) n_row_tiles * (size_t) n_k_tiles * GGML_Q4_0_TILE32_SIZE;

    const int64_t start_slice = (int64_t) (offset / slice_size);
    int64_t       end_slice   = (int64_t) ((offset + size + slice_size - 1) / slice_size);
    if (end_slice > n_slices) {
        end_slice = n_slices;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const int64_t slice_start = slice_idx * (int64_t) slice_size;
        const int64_t cur_start   = (int64_t) offset > slice_start ? (int64_t) offset : slice_start;
        const int64_t cur_end     = (int64_t) (offset + size) < slice_start + (int64_t) slice_size
                                        ? (int64_t) (offset + size) : slice_start + (int64_t) slice_size;

        const int64_t start_row = (cur_start - slice_start) / (int64_t) row_size;
        int64_t       end_row   = ((cur_end - slice_start) + (int64_t) row_size - 1) / (int64_t) row_size;
        if (end_row > ne1) {
            end_row = ne1;
        }

        block_q4_0 *    dst_slice  = dst + (cur_start - (int64_t) offset) / (int64_t) sizeof(block_q4_0);
        const uint8_t * matrix_src = src + (size_t) slice_idx * matrix_size;

        for (int ct = (int) (start_row / GGML_Q4_0_TILE32_ROWS);
             ct < (int) ((end_row + GGML_Q4_0_TILE32_ROWS - 1) / GGML_Q4_0_TILE32_ROWS); ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                const uint8_t * tile_src = matrix_src +
                    ((size_t) ct * (size_t) n_k_tiles + (size_t) kt) * GGML_Q4_0_TILE32_SIZE;
                const ggml_half * scale_src = (const ggml_half *) (tile_src + GGML_Q4_0_TILE32_QUANT_BYTES);

                for (int row = 0; row < GGML_Q4_0_TILE32_ROWS; row++) {
                    const int64_t r = ct * GGML_Q4_0_TILE32_ROWS + row;
                    if (r < start_row || r >= end_row) {
                        continue;
                    }
                    block_q4_0 * b = &dst_slice[(r - start_row) * blocks_per_row + kt];
                    b->d = scale_src[row];
                    // tile byte cp * 32 + row holds element 2*cp in the low nibble and element
                    // 2*cp + 1 in the high one; native qs[j] holds element j (low) and element
                    // j + 16 (high)
                    for (int j = 0; j < QK4_0 / 2; ++j) {
                        const uint8_t lo = tile_src[(j / 2) * GGML_Q4_0_TILE32_ROWS + row];
                        const uint8_t hi = tile_src[(j / 2 + QK4_0 / 4) * GGML_Q4_0_TILE32_ROWS + row];
                        const int     sh = (j % 2) * 4;
                        b->qs[j] = (uint8_t) (((lo >> sh) & 0x0F) | (((hi >> sh) & 0x0F) << 4));
                    }
                }
            }
        }
    }
}

// One tile row (32 weights of one K tile) against 32 f32 activations, accumulated in f32 and
// scaled once at the end. Kept free of activation quantization on purpose: a scalar reference for
// the layout, not for an activation format.
static inline float ggml_q4_0_tile32_dot_row(const uint8_t * tile, const float * x, int row) {
    const float d = ggml_fp16_to_fp32(((const ggml_half *) (tile + GGML_Q4_0_TILE32_QUANT_BYTES))[row]);

    float sum = 0.0f;
    for (int cp = 0; cp < GGML_Q4_0_TILE32_K / 2; cp++) {
        const uint8_t b = tile[cp * GGML_Q4_0_TILE32_ROWS + row];
        sum += (float) ((b & 0x0F) - 8) * x[2 * cp];
        sum += (float) ((b >> 4) - 8) * x[2 * cp + 1];
    }
    return sum * d;
}

#endif // GGML_Q4_0_TILE32_H
