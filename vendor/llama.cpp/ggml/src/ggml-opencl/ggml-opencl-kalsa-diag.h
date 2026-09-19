#pragma once

// KALSA diagnostic / experiment scaffolding relocated out of ggml-opencl.cpp.
//
// ggml-opencl.cpp #includes this header, so every existing call site in that
// file keeps compiling unchanged. The definitions live in ggml-opencl-kalsa-diag.cpp.
//
// Widening note: the 17 helpers below were `static` (internal linkage) inside
// ggml-opencl.cpp. They are now declared and defined WITHOUT `static` (external
// linkage) because their definitions live in a second translation unit
// (ggml-opencl-kalsa-diag.cpp) that ggml-opencl.cpp calls into. The 3 globals
// g_gemv_audit_dst_snaps / g_gemv_audit_n_post_enq / g_gemv_audit_n_post_bind are
// defined in ggml-opencl.cpp but referenced from the relocated helpers, so they
// were likewise widened from `static` to external linkage. The remaining moved
// globals (g_opencl_alloc_sum, g_gemv_audit_n_overlap_pairs,
// g_gemv_audit_overlap_logged) are used only within ggml-opencl-kalsa-diag.cpp and
// therefore keep internal (`static`) linkage.

// CL/cl.h pulls in CL/cl_version.h, which is include-guarded and fixes
// CL_TARGET_OPENCL_VERSION for the whole TU at first inclusion. A .cpp that
// includes this header must therefore define CL_TARGET_OPENCL_VERSION (and
// CL_USE_DEPRECATED_OPENCL_1_2_APIS) before doing so; see
// ggml-opencl-kalsa-diag.cpp. The macro is deliberately not set here, because
// ggml-opencl.cpp includes this header after making its own version selection.
#include <CL/cl.h>

#include <cstddef>
#include <cstdint>
#include <vector>

// Snapshot struct (previously defined in ggml-opencl.cpp).
// Qwen A4: dst snapshot at dispatch vs get_tensor. Checksums only (the
// audit's full `got` vector is freed when the audit returns).
struct ggml_opencl_gemv_dst_snap {
    cl_mem mem = nullptr;
    size_t off = 0;
    size_t size = 0;
    int    case_id = 0;
    uint64_t ck = 0;
    bool   consumed = false;
    float first4[4] = {0, 0, 0, 0};
    float last4[4]  = {0, 0, 0, 0};
};

// Relocated helpers (definitions in ggml-opencl-kalsa-diag.cpp).
void ggml_opencl_log_kernel_once(cl_kernel kernel);
bool ggml_opencl_env_value_enabled(const char * name);
bool ggml_opencl_oracle_enabled(void);
bool ggml_opencl_log_alloc_enabled(void);
void ggml_opencl_log_alloc(const char * kind, size_t bytes, cl_int err);
void ggml_opencl_log_alloc_note(const char * msg);
bool ggml_opencl_gemv_audit_enabled(void);
int ggml_opencl_env_int_clamped(const char * name, int def, int lo, int hi);
size_t ggml_opencl_gemv_audit_stash_cap_bytes(void);
uint64_t ggml_opencl_gemv_fnv1a64(const void * p, size_t n);
void ggml_opencl_gemv_ends4(const float * f, size_t n_elem, float first4[4], float last4[4]);
bool ggml_opencl_gemv_range_overlap(size_t a0, size_t a1, size_t b0, size_t b1);
void ggml_opencl_gemv_emit_overlap_map(void);
void ggml_opencl_gemv_register_dst_snap(
        int case_id, cl_mem mem, size_t off, size_t size,
        const float * got, size_t n_elem);
void ggml_opencl_gemv_check_dst_readback(
        cl_command_queue queue, cl_mem mem, size_t get_off, size_t get_size,
        const void * returned, size_t returned_size);
bool ggml_opencl_qk23_oracle_enabled(void);
bool ggml_opencl_gemv_audit_ratio_near(double r, double v);

// State defined in ggml-opencl.cpp but read by the relocated helpers.
extern std::vector<ggml_opencl_gemv_dst_snap> g_gemv_audit_dst_snaps;
extern int g_gemv_audit_n_post_enq;
extern int g_gemv_audit_n_post_bind;
