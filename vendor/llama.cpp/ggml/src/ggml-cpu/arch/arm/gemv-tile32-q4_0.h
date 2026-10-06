// Batch-1 gemv over Q4_0 weights in the ggml Q4_0 32x32 tile layout
// (ggml/src/ggml-q4_0-tile32.h): the same weight copy the HTP reads. Ported from
// kalsallama-cpu-gemv spike/onecopy-cpu-gemv 4e6c75252 (arch/arm/gemv-tiled-q4_0.cpp),
// renamed for the shared layout contract; mac8/mac16 variants were not ported.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// s[r] = dot(tile32 weight row r, q8_0 row) for nrows rows whose first row is the first row of
// a 32-row tile. n must be a multiple of QK4_0 and nrows a multiple of 32. s_row is the dst row
// stride in floats. Callers must guard on the same ARM conditions the definition is compiled for.
void ggml_gemv_tile32_q4_0_q8_0(int n, float * s, size_t s_row,
                                const void * vw, const void * vy, int64_t nrows);

#ifdef __cplusplus
}
#endif
