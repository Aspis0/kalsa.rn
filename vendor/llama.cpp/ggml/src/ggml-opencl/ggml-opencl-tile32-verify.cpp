// Match the OpenCL version setup of ggml-opencl.cpp (see ggml-opencl-kalsa-diag.cpp):
// cl_version.h is include-guarded and the first inclusion in a TU fixes the API level.
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// Lab-only run-time verify of one v4 GEMV. After the enqueue it reads back the activation the
// kernel was given and the result through both the dst sub-buffer and its parent, then compares
// the results with the CPU reference the load-time self-check uses. Reads only, after clFinish,
// so the op's result does not change.

#include "ggml-opencl-tile32-internal.h"

#include "ggml-impl.h"
#include "ggml-opencl-tile32-check.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

long g_budget = -1;   // GEMV calls still to check; -1 until the env is parsed
long g_calls  = 0;    // calls checked so far, numbered in the log

long budget_from_env(void) {
    const char * s = getenv("GGML_OPENCL_TILE32_VERIFY");
    const long n = ggml_opencl_tile32_check_budget(s);
    if (s && *s && n == 0) {
        GGML_LOG_WARN("ggml-opencl-tile32: GGML_OPENCL_TILE32_VERIFY=%s is not a positive count, verify off\n", s);
    }
    return n;
}

bool read_floats(const ggml_opencl_tile32_env * env, cl_mem buf, cl_ulong off, float * dst, size_t n) {
    return ggml_opencl_tile32_cl_ok(clEnqueueReadBuffer(env->queue, buf, CL_TRUE, off, n * sizeof(float), dst,
                                                        0, nullptr, nullptr), "verify read-back");
}

// Where a sub-buffer starts in its parent: the offset the kernel really got, not the one asked for.
bool sub_offset(cl_mem sub, size_t * off) {
    return ggml_opencl_tile32_cl_ok(clGetMemObjectInfo(sub, CL_MEM_OFFSET, sizeof(*off), off, nullptr),
                                    "verify sub-buffer offset");
}

bool image_width(cl_mem img, size_t * px) {
    return ggml_opencl_tile32_cl_ok(clGetImageInfo(img, CL_IMAGE_WIDTH, sizeof(*px), px, nullptr),
                                    "verify image width");
}

double max_abs_diff(const float * a, const float * b, int64_t n) {
    double maxd = 0.0;
    for (int64_t i = 0; i < n; i++) {
        maxd = std::max(maxd, fabs((double) a[i] - (double) b[i]));
    }
    return maxd;
}

} // namespace

bool ggml_opencl_tile32_verify_next_locked(void) {
    if (g_budget < 0) {
        g_budget = budget_from_env();
    }
    if (g_budget == 0) {
        return false;
    }
    g_budget--;
    return true;
}

void ggml_opencl_tile32_verify_gemv_locked(const ggml_opencl_tile32_env * env, const ggml_opencl_tile32_views * v,
                                           const ggml_tensor * w, const ggml_opencl_tile32_gemv_io * io) {
    const int64_t K = w->ne[0], M = w->ne[1];
    const long call = ++g_calls;

    std::vector<float> x(K), sub(M), par(M), ref(M);
    size_t xsub_off = 0, dsub_off = 0, img_px = 0;
    const bool read_ok =
        ggml_opencl_tile32_cl_ok(clFinish(env->queue), "verify finish") &&
        read_floats(env, io->act, io->act_off, x.data(), K) &&
        read_floats(env, io->dst_sub, 0, sub.data(), M) &&
        read_floats(env, io->dst, io->dst_off, par.data(), M) &&
        sub_offset(io->act_sub, &xsub_off) &&
        sub_offset(io->dst_sub, &dsub_off) &&
        image_width(io->act_img, &img_px);
    if (!read_ok) {
        GGML_LOG_INFO("ggml-opencl-tile32: verify call=%ld tensor=%s status=ERROR\n", call, w->name);
        return;
    }

    ggml_opencl_tile32_check_reference(v->host_tiles, x.data(), K, M, ref.data());
    double sq = 0.0;
    for (int64_t m = 0; m < M; m++) {
        sq += (double) ref[m] * ref[m];
    }
    const double rms = sqrt(sq / (double) M);
    // Both views must match the reference: the kernel wrote the sub-buffer, the graph reads the parent.
    const double maxd   = std::max(max_abs_diff(sub.data(), ref.data(), M), max_abs_diff(par.data(), ref.data(), M));
    const double maxsub = max_abs_diff(sub.data(), par.data(), M);
    const bool   pass   = ggml_opencl_tile32_check_pass(maxd, rms);

    GGML_LOG_INFO("ggml-opencl-tile32: verify call=%ld tensor=%s rows=%lld max|diff|=%.3e rms=%.4e "
                  "sub_vs_parent_max=%.3e act_off=%llu xsub_off=%zu img_px=%zu dst_off=%llu dsub_off=%zu status=%s\n",
                  call, w->name, (long long) M, maxd, rms, maxsub,
                  (unsigned long long) io->act_off, xsub_off, img_px,
                  (unsigned long long) io->dst_off, dsub_off, pass ? "PASS" : "FAIL");
}
