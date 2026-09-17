#pragma once

// Internal Q2_K_8x4 / Q3_K_8x4 helpers. Not a public GGML API / ABI.
// Included by the M0 host test via a private ggml/src include path.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t       lm_ggml_cpu_repack_q2_k_8x4_nbytes(int64_t ne0, int64_t ne1, int64_t ne2);
size_t       lm_ggml_cpu_repack_q3_k_8x4_nbytes(int64_t ne0, int64_t ne1, int64_t ne2);
const char * lm_ggml_cpu_repack_extra_name(const struct lm_ggml_tensor * tensor);
int          lm_ggml_cpu_repack_pack_q2_k_8x4(struct lm_ggml_tensor * t, const void * data, size_t data_size);
int          lm_ggml_cpu_repack_pack_q3_k_8x4(struct lm_ggml_tensor * t, const void * data, size_t data_size);

// Scalar layout oracles: dst is [ne2][ne1][ne0] fp32. Same fp16->fp32 as dequantize_row_q*_K.
void lm_ggml_cpu_repack_dequant_q2_k_8x4(const void * packed, float * dst, int64_t ne0, int64_t ne1, int64_t ne2);
void lm_ggml_cpu_repack_dequant_q3_k_8x4(const void * packed, float * dst, int64_t ne0, int64_t ne1, int64_t ne2);

uint8_t lm_ggml_cpu_repack_q2_k_8x4_get_sm(const void * packed, int64_t ne0, int64_t ne1,
                                        int64_t expert, int64_t row, int64_t block, int sb);
uint8_t lm_ggml_cpu_repack_q2_k_8x4_get_q(const void * packed, int64_t ne0, int64_t ne1,
                                       int64_t expert, int64_t row, int64_t block, int sb, int lane);
uint8_t lm_ggml_cpu_repack_q3_k_8x4_get_scale_code(const void * packed, int64_t ne0, int64_t ne1,
                                                int64_t expert, int64_t row, int64_t block, int sb);
uint8_t lm_ggml_cpu_repack_q3_k_8x4_get_q(const void * packed, int64_t ne0, int64_t ne1,
                                       int64_t expert, int64_t row, int64_t block, int sb, int lane);

void lm_ggml_gemv_q2_K_8x4_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void lm_ggml_gemv_q3_K_8x4_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void lm_ggml_gemm_q2_K_8x4_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void lm_ggml_gemm_q3_K_8x4_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);

// True iff extra is the Q2_K_8x4 or Q3_K_8x4 CPU_REPACK trait. Used to refuse native fallthrough.
bool lm_ggml_cpu_repack_extra_is_q23k_8x4(const void * extra);
// True iff extra is any CPU_REPACK tiled trait (8x4/8x8/16x1 families).
// Pointer-equality against the file-static set; never dereference extra.
bool lm_ggml_cpu_repack_extra_is_tiled(const void * extra);

#ifdef __cplusplus
}
#endif
