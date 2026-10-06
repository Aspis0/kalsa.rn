#ifndef Q4_0_RAW_TILE_H
#define Q4_0_RAW_TILE_H

// Converts row-major GGUF block_q4_0 rows (as read by the CPU backend) into the
// 32x32 weight tiles consumed by the tiled_vec_dot_q4_0_32x{1,2} kernels. The output matches
// repack_q4_0_tiled in ggml-hexagon.cpp byte for byte: tile byte [cp * 32 + row]
// holds element 2*cp (low nibble) and 2*cp + 1 (high nibble) of that row's block,
// followed by the 32 fp16 row scales. Rows at or past n_rows are padding
// (quants 8, scale 0), so they contribute zero.
//
// Needs block_q4_0 and QK4_0 from ggml-common.h.

static inline void q4_0_raw_rows_to_tiles(uint8_t * restrict tiles,
                                          uint32_t tile_stride,
                                          const block_q4_0 * restrict rows,
                                          uint32_t n_k_tiles,
                                          uint32_t n_rows) {
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        uint8_t * restrict    q = tiles + kt * tile_stride;
        ggml_half * restrict  d = (ggml_half *) (q + 32 * QK4_0 / 2);

        for (uint32_t row = 0; row < 32; row++) {
            if (row >= n_rows) {
                for (uint32_t cp = 0; cp < QK4_0 / 2; cp++) {
                    q[cp * 32 + row] = 0x88;
                }
                d[row] = 0;
                continue;
            }

            // ggml order: element i is the low nibble of qs[i], element i + 16 the high one
            const block_q4_0 * b = &rows[row * n_k_tiles + kt];
            for (uint32_t cp = 0; cp < QK4_0 / 4; cp++) {
                const uint8_t lo = b->qs[2 * cp];
                const uint8_t hi = b->qs[2 * cp + 1];
                q[cp * 32 + row]                 = (lo & 0x0F) | (uint8_t) (hi << 4);
                q[(cp + QK4_0 / 4) * 32 + row]   = (lo >> 4)   | (hi & 0xF0);
            }
            d[row] = b->d;
        }
    }
}

#endif // Q4_0_RAW_TILE_H
