#pragma once

// The pure rules of the read-only TILE32 mat-vec checks: the CPU reference the load-time
// self-check and the lab verify both compare against, the pass rule, and the lab verify's
// budget parse. No OpenCL here, so the host test can drive all of it.

#include "ggml-q4_0-tile32.h"

#include <stdint.h>
#include <stdlib.h>

// y = W x for the M rows of one TILE32 matrix, read from its host tiles. Same accumulation
// order as the self-check always used, so moving it here changes no number.
static inline void ggml_opencl_tile32_check_reference(const uint8_t * tiles, const float * x,
                                                      int64_t K, int64_t M, float * y) {
    const int64_t nkt = K / GGML_Q4_0_TILE32_K;
    for (int64_t m = 0; m < M; m++) {
        float sum = 0.0f;
        for (int64_t kt = 0; kt < nkt; kt++) {
            sum += ggml_q4_0_tile32_dot_row(tiles + ((m / GGML_Q4_0_TILE32_ROWS) * nkt + kt) * GGML_Q4_0_TILE32_SIZE,
                                            x + kt * GGML_Q4_0_TILE32_K, m % GGML_Q4_0_TILE32_ROWS);
        }
        y[m] = sum;
    }
}

// |diff| <= 1e-5 of the output RMS. An all-zero weight makes both sides all zero, where a
// relative bound is undefined: accept that pair and keep the relative bound for the rest.
static inline bool ggml_opencl_tile32_check_pass(double maxd, double rms) {
    return rms > 0.0 ? maxd <= 1e-5 * rms : maxd == 0.0;
}

// GGML_OPENCL_TILE32_VERIFY: the number of GEMV calls the lab verify checks. 0 for an unset,
// empty, non-numeric or non-positive value.
static inline long ggml_opencl_tile32_check_budget(const char * s) {
    if (!s || !*s) {
        return 0;
    }
    char * end = nullptr;
    const long n = strtol(s, &end, 10);
    return *end == '\0' && n > 0 ? n : 0;
}
