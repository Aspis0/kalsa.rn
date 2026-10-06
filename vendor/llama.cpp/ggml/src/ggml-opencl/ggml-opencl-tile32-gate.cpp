// Match the OpenCL version setup of ggml-opencl.cpp (see ggml-opencl-kalsa-diag.cpp):
// cl_version.h is include-guarded and the first inclusion in a TU fixes the API level.
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "ggml-opencl-tile32-internal.h"

#include "ggml-impl.h"
#include "ggml-q4_0-tile32.h"

#include <cmath>
#include <cstring>
#include <map>
#include <vector>

namespace {

enum gate_state { GATE_UNTESTED, GATE_OK, GATE_FAILED };

// One verdict per device (G7): a first device that misses the facts gate or the canary must
// not poison a later qualifying one. The program, kernel and imports are still one process
// singleton, bound to the first device whose gate passes; a second device's own facts are
// evaluated but gate_locked declines it while that binding stands.
std::map<cl_device_id, gate_state> g_gates;
cl_device_id     g_leg_device   = nullptr;
cl_command_queue g_leg_queue    = nullptr;
bool             g_leg_disabled = false;   // a runtime miss (import / views / self-check / program)

// Canary for the two image1d_buffer views the v4 kernel reads, same shape as the spike's:
// a 256 B buffer, a sub-buffer over it, one RGBA/UINT32 and one RGBA/HALF_FLOAT image
// covering those bytes.
bool views_canary(const ggml_opencl_tile32_env * env, const char ** reason) {
    if (env->image_max_buffer_size == 0) {
        *reason = "no image1d_buffer views (CL_DEVICE_IMAGE_MAX_BUFFER_SIZE 0)";
        return false;
    }
    cl_int err;
    bool ok = false;
    cl_mem b = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 256, nullptr, &err);
    if (b) {
        const cl_buffer_region region = { 0, 256 };
        cl_mem s = clCreateSubBuffer(b, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        if (s) {
            cl_image_desc d;
            memset(&d, 0, sizeof(d));
            d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            d.buffer     = s;
            cl_image_format fq = { CL_RGBA, CL_UNSIGNED_INT32 };
            d.image_width = 16;   // both canaries cover the same 256 B
            cl_mem iq = clCreateImage(env->context, CL_MEM_READ_ONLY, &fq, &d, nullptr, &err);
            cl_image_format fs = { CL_RGBA, CL_HALF_FLOAT };
            d.image_width = 32;
            cl_mem is = iq ? clCreateImage(env->context, CL_MEM_READ_ONLY, &fs, &d, nullptr, &err) : nullptr;
            ok = iq && is;
            if (is) { clReleaseMemObject(is); }
            if (iq) { clReleaseMemObject(iq); }
            clReleaseMemObject(s);
        }
        clReleaseMemObject(b);
    }
    if (!ok) {
        *reason = "RGBA/UINT32 or RGBA/HALF_FLOAT image1d_buffer views unavailable";
    }
    return ok;
}

// Deterministic pseudo-random activation normalized to unit L2 norm: unlike a basis
// vector it exercises every K-pair lane and K-tile of the views.
void unit_vector(int64_t k, std::vector<float> & x) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    double sq = 0.0;
    for (int64_t i = 0; i < k; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        x[i] = (float) ((s >> 11) * (1.0 / 9007199254740992.0));
        sq += (double) x[i] * x[i];
    }
    const float n = (float) (1.0 / sqrt(sq));
    for (int64_t i = 0; i < k; i++) {
        x[i] *= n;
    }
}

} // namespace

void ggml_opencl_tile32_disable_locked(const char * reason) {
    if (g_leg_disabled) {
        return;   // the first miss already named its clause
    }
    g_leg_disabled = true;
    GGML_LOG_INFO("ggml-opencl: TILE32 leg disabled: %s\n", reason);
}

bool ggml_opencl_tile32_gate_locked(const ggml_opencl_tile32_env * env) {
    if (g_leg_disabled) {
        return false;
    }
    gate_state & state = g_gates[env->device];
    if (state == GATE_UNTESTED) {
        // need_bytes = 0: no TILE32 buffer exists yet; the import re-asks the size clause with
        // the dma-buf block it is about to place in GPU memory
        const char * reason = nullptr;
        state = ggml_opencl_tile32_facts_ok(&env->facts, 0, &reason) && views_canary(env, &reason)
              ? GATE_OK : GATE_FAILED;
        if (state == GATE_FAILED) {
            GGML_LOG_INFO("ggml-opencl: TILE32 leg not available on %s: %s\n",
                          env->facts.device_name ? env->facts.device_name : "(unnamed device)", reason);
            return false;
        }
    }
    if (state != GATE_OK) {
        return false;
    }
    if (!g_leg_device) {
        g_leg_device = env->device;
        g_leg_queue  = env->queue;
        GGML_LOG_INFO("ggml-opencl: TILE32 leg enabled on %s (%s)\n",
                      env->facts.device_name, env->facts.device_version);
    }
    return g_leg_device == env->device;
}

cl_command_queue ggml_opencl_tile32_queue_locked(void) {
    return g_leg_queue;
}

bool ggml_opencl_tile32_ready(const ggml_opencl_tile32_env * env) {
    std::lock_guard<std::mutex> lock(ggml_opencl_tile32_mutex());
    return ggml_opencl_tile32_gate_locked(env);
}

// Read-only self-check (E4), once per tensor at admission: the v4 mat-vec on a unit
// vector through the very views the op will use, against the CPU tile reader on the host
// mapping of the same bytes. |diff| <= 1e-5 of output RMS (spike measured 6-9e-7).
bool ggml_opencl_tile32_self_check_locked(const ggml_opencl_tile32_env * env,
                                          const ggml_opencl_tile32_views * v,
                                          const ggml_tensor * w, cl_kernel kernel) {
    const int64_t K = w->ne[0], M = w->ne[1], nkt = K / GGML_Q4_0_TILE32_K;
    std::vector<float> x(K), ref(M), dev(M);
    unit_vector(K, x);

    const uint8_t * tiles = v->host_tiles;
    for (int64_t m = 0; m < M; m++) {
        float sum = 0.0f;
        for (int64_t kt = 0; kt < nkt; kt++) {
            sum += ggml_q4_0_tile32_dot_row(tiles + ((m / GGML_Q4_0_TILE32_ROWS) * nkt + kt) * GGML_Q4_0_TILE32_SIZE,
                                            x.data() + kt * GGML_Q4_0_TILE32_K, m % GGML_Q4_0_TILE32_ROWS);
        }
        ref[m] = sum;
    }

    cl_int err;
    cl_mem xb = clCreateBuffer(env->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               K * sizeof(float), x.data(), &err);
    if (!xb) {
        GGML_LOG_ERROR("ggml-opencl-tile32: self-check buffer failed: %d\n", err);
        return false;
    }
    cl_image_desc d;
    memset(&d, 0, sizeof(d));
    d.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    d.image_width = K / 4;
    d.buffer      = xb;
    cl_image_format f = { CL_RGBA, CL_FLOAT };
    cl_mem xi = clCreateImage(env->context, CL_MEM_READ_ONLY, &f, &d, nullptr, &err);
    cl_mem db = xi ? clCreateBuffer(env->context, CL_MEM_WRITE_ONLY, M * sizeof(float), nullptr, &err) : nullptr;
    bool ok = false;
    if (db) {
        cl_uint poff = v->poff;
        cl_int ki = (cl_int) K, mi = (cl_int) M;
        const size_t gws = ((M + 31) / 32) * 64, lws = 64;
        const bool args_ok =
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 0, sizeof(cl_mem),  &v->wq), "self-check arg wq")   &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 1, sizeof(cl_uint), &poff),  "self-check arg poff") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 2, sizeof(cl_mem),  &xi),    "self-check arg x")    &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 3, sizeof(cl_mem),  &v->ws), "self-check arg ws")   &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 4, sizeof(cl_mem),  &db),    "self-check arg dst")  &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 5, sizeof(cl_int),  &ki),    "self-check arg K")    &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 6, sizeof(cl_int),  &mi),    "self-check arg M");
        if (args_ok &&
            ggml_opencl_tile32_cl_ok(clEnqueueNDRangeKernel(env->queue, kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr),
                                     "self-check enqueue") &&
            ggml_opencl_tile32_cl_ok(clEnqueueReadBuffer(env->queue, db, CL_TRUE, 0, M * sizeof(float), dev.data(),
                                                         0, nullptr, nullptr), "self-check read-back")) {
            double sq = 0.0, maxd = 0.0;
            for (int64_t m = 0; m < M; m++) {
                sq += (double) dev[m] * dev[m];
                const double dd = fabs((double) dev[m] - (double) ref[m]);
                if (dd > maxd) {
                    maxd = dd;
                }
            }
            const double rms = sqrt(sq / (double) M);
            // an all-zero weight makes both sides all zero, where a relative bound is
            // undefined: accept that pair and keep the relative bound for the rest
            ok = rms > 0.0 ? maxd <= 1e-5 * rms : maxd == 0.0;
            GGML_LOG_INFO("ggml-opencl-tile32: self-check %s: %s max|diff| %.3e rms %.4f\n",
                          w->name, ok ? "PASS" : "FAIL", maxd, rms);
        }
        clReleaseMemObject(db);
    } else {
        GGML_LOG_ERROR("ggml-opencl-tile32: self-check setup failed: %d\n", err);
    }
    if (xi) { clReleaseMemObject(xi); }
    clReleaseMemObject(xb);
    return ok;
}

// Read-only self-check of one HOST Q6_K weight (F3), once per tensor at admission: the kernel
// the lm_head compute path launches (plan->kernel, the AoS GEMV or the plain
// kernel_mul_mv_q6_K_f32) on the first rows of the very bytes the import maps, against ggml's
// own Q6_K dequant plus a double-accumulated dot on the host pointer. A wrong import mapping,
// a stale mapping or a policy mismatch is O(rms) off; an emulation of the kernel's lane
// grouping differs from that reference by < 3e-7 * rms at K = 4096, so 1e-5 * rms keeps >
// 30x margin and still catches any wrong byte.
bool ggml_opencl_host_self_check_locked(const struct ggml_opencl_tile32_env * env,
                                        const struct ggml_opencl_tile32_host_view * v,
                                        const ggml_tensor * w,
                                        const struct ggml_opencl_host_q6k_plan * plan) {
    const int64_t K = w->ne[0], M = w->ne[1];
    const int64_t bs = ggml_blck_size(GGML_TYPE_Q6_K);
    const ggml_to_float_t to_float = ggml_get_type_traits(GGML_TYPE_Q6_K)->to_float;
    // the plain kernel reads a contiguous matrix with row stride ne00/bs blocks; anything else
    // was never admitted, and the check must not read a layout the kernel does not. Each miss
    // logs the precondition and its values: a silent false here is what hid the null kernel
    // handle in step 4b (Amendment I).
    if (!to_float) {
        GGML_LOG_WARN("ggml-opencl: HOST self-check for %s: no Q6_K to_float in this build\n", w->name);
        return false;
    }
    if (K <= 0 || K % bs != 0) {
        GGML_LOG_WARN("ggml-opencl: HOST self-check for %s: K %lld is not a positive multiple of the Q6_K block size %lld\n",
                      w->name, (long long) K, (long long) bs);
        return false;
    }
    if (w->ne[2] != 1 || w->ne[3] != 1) {
        GGML_LOG_WARN("ggml-opencl: HOST self-check for %s: ne2 %lld ne3 %lld, the plain kernel reads a 2-D matrix\n",
                      w->name, (long long) w->ne[2], (long long) w->ne[3]);
        return false;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q6_K, K);
    if (w->nb[1] != row_bytes) {
        GGML_LOG_WARN("ggml-opencl: HOST self-check for %s: row stride %zu != %zu bytes for K %lld\n",
                      w->name, (size_t) w->nb[1], row_bytes, (long long) K);
        return false;
    }

    // The plan carries the kernel and its grid: the check must read the same rows through the
    // same kernel the compute path launches, or it validates a kernel nothing runs.
    const int64_t R = M < 32 ? M : 32;   // a row slice of the same tensor, not a copy of it
    std::vector<float> x(K), row(K), ref(R), dev(R);
    unit_vector(K, x);
    for (int64_t r = 0; r < R; r++) {
        to_float((const uint8_t *) w->data + r * row_bytes, row.data(), K);
        double sum = 0.0;
        for (int64_t k = 0; k < K; k++) {
            sum += (double) x[k] * (double) row[k];
        }
        ref[r] = (float) sum;
    }

    cl_int err;
    cl_mem xb = clCreateBuffer(env->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               (size_t) K * sizeof(float), x.data(), &err);
    if (!xb) {
        GGML_LOG_ERROR("ggml-opencl: HOST self-check buffer failed: %d\n", err);
        return false;
    }
    cl_mem db = clCreateBuffer(env->context, CL_MEM_WRITE_ONLY, (size_t) R * sizeof(float), nullptr, &err);
    bool ok = false;
    if (db) {
        // the kernel the op will run, with the same row mapping at the R-row slice this check
        // reads and writes: the plan's own gws is the M-row grid (4000 work groups for the
        // 128000-row head), whose rows the R-sized ne01 argument already guards out
        cl_kernel kernel = plan->kernel;
        size_t gws[3], lws[3];
        ggml_opencl_host_q6k_plan_grid(plan, R, 1, gws, lws);
        cl_ulong off0 = v->off, off1 = 0, offd = 0;
        const int64_t k64 = K, r64 = R;
        cl_int arg_ne00 = (cl_int) k64, arg_ne01 = (cl_int) r64, arg_ne02 = 1, arg_ne10 = (cl_int) k64;
        cl_int arg_ne12 = 1, arg_ne0 = (cl_int) r64, arg_ne1 = 1, arg_r2 = 1, arg_r3 = 1;
        const bool args_ok =
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &v->buf),    "HOST self-check arg src0") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &off0),      "HOST self-check arg offset0") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &xb),        "HOST self-check arg src1") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &off1),      "HOST self-check arg offset1") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &db),        "HOST self-check arg dst") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offd),      "HOST self-check arg offsetd") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  6, sizeof(cl_int),   &arg_ne00),  "HOST self-check arg ne00") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  7, sizeof(cl_int),   &arg_ne01),  "HOST self-check arg ne01") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  8, sizeof(cl_int),   &arg_ne02),  "HOST self-check arg ne02") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel,  9, sizeof(cl_int),   &arg_ne10),  "HOST self-check arg ne10") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 10, sizeof(cl_int),   &arg_ne12),  "HOST self-check arg ne12") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 11, sizeof(cl_int),   &arg_ne0),   "HOST self-check arg ne0") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 12, sizeof(cl_int),   &arg_ne1),   "HOST self-check arg ne1") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 13, sizeof(cl_int),   &arg_r2),    "HOST self-check arg r2") &&
            ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 14, sizeof(cl_int),   &arg_r3),    "HOST self-check arg r3");
        if (args_ok &&
            ggml_opencl_tile32_cl_ok(clEnqueueNDRangeKernel(env->queue, kernel, 3, nullptr,
                                                            gws,
                                                            lws, 0, nullptr, nullptr),
                                     "HOST self-check enqueue") &&
            ggml_opencl_tile32_cl_ok(clEnqueueReadBuffer(env->queue, db, CL_TRUE, 0, (size_t) R * sizeof(float), dev.data(),
                                                         0, nullptr, nullptr), "HOST self-check read-back")) {
            double sq = 0.0, maxd = 0.0;
            for (int64_t r = 0; r < R; r++) {
                sq += (double) dev[r] * dev[r];
                const double dd = fabs((double) dev[r] - (double) ref[r]);
                if (dd > maxd) {
                    maxd = dd;
                }
            }
            const double rms = sqrt(sq / (double) R);
            ok = rms > 0.0 ? maxd <= 1e-5 * rms : maxd == 0.0;
            GGML_LOG_INFO("ggml-opencl: HOST self-check %s: %s max|diff| %.3e rms %.4f (%d of %d rows, %s kernel)\n",
                          w->name, ok ? "PASS" : "FAIL", maxd, rms, (int) R, (int) M,
                          plan->aos ? "aos" : "plain");
        }
        clReleaseMemObject(db);
    } else {
        GGML_LOG_ERROR("ggml-opencl: HOST self-check setup failed: %d\n", err);
    }
    clReleaseMemObject(xb);
    return ok;
}
