// Match the OpenCL version setup of ggml-opencl.cpp (see ggml-opencl-kalsa-diag.cpp):
// cl_version.h is include-guarded and the first inclusion in a TU fixes the API level.
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "ggml-opencl-tile32-internal.h"

#include "ggml-impl.h"
#include "ggml-q4_0-tile32.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

// Read lazily, not at static init: a missing file must only fail the TILE32 leg,
// never terminate a process that never touches it.
static const std::string & tile32_kernel_src(void) {
    static const std::string src = [] {
#ifdef GGML_OPENCL_EMBED_KERNELS
        return std::string {
            #include "mul_mv_q4_0_f32_tiled_v4.cl.h"
        };
#else
        std::ifstream f("mul_mv_q4_0_f32_tiled_v4.cl", std::ios::binary);
        return f ? std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>())
                 : std::string();
#endif
    }();
    return src;
}

namespace {

cl_program g_program = nullptr;
cl_kernel  g_kernel  = nullptr;

} // namespace

cl_kernel ggml_opencl_tile32_kernel_locked(const ggml_opencl_tile32_env * env) {
    if (g_kernel) {
        return g_kernel;
    }
    const char * src = tile32_kernel_src().c_str();
    if (!*src) {
        GGML_LOG_ERROR("ggml-opencl-tile32: kernel source not found\n");
        return nullptr;
    }
    if (!g_program) {
        // Amendment I declares a numerics change here (E2): env_fill now loads the kernel set
        // before the first self-check, so kernel_compile_opts is set and the v4 program is
        // built with the engine's general options (-cl-fast-relaxed-math, -cl-mad-enable, ...)
        // instead of the empty string it always saw before. Those options permit reassociation
        // and contraction in the reduction epilogue, so KLD numbers measured before this load
        // point are not comparable with numbers measured after it.
        g_program = env->build_program(env->build_program_opaque, src,
                                       env->kernel_compile_opts ? env->kernel_compile_opts : "");
        if (!g_program) {
            return nullptr;
        }
    }
    cl_int err;
    g_kernel = clCreateKernel(g_program, "kernel_mul_mat_q4_0_f32_tiled_v4", &err);
    if (!g_kernel) {
        GGML_LOG_ERROR("ggml-opencl-tile32: clCreateKernel failed: %d\n", err);
    }
    return g_kernel;
}

void ggml_cl_mul_mat_tile32(const struct ggml_opencl_tile32_env * env,
                            const struct ggml_tensor * src0,
                            const struct ggml_tensor * src1,
                            const struct ggml_tensor * dst,
                            cl_mem src1_buf, cl_ulong src1_off, cl_mem dst_buf, cl_ulong dst_off) {
    std::lock_guard<std::mutex> lock(ggml_opencl_tile32_mutex());
    const char * reason = nullptr;

    // G10: ggml_backend_sched_set_tensor_backend can pin this op without asking supports_op,
    // so the admitted shape is re-asserted here, right before the kernel.
    GGML_ASSERT(src0->buffer);
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0 && src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] > 0 && src0->ne[0] % GGML_Q4_0_TILE32_K == 0 && src0->ne[1] > 0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    // Amendment I fail-closed: the import map serves weights only, and the activation and the
    // result live in this split's own OpenCL buffers. A host buffer in either place means the
    // scheduler treated a device host buft as claimable (the step-4b -38) and skipped the copy;
    // aborting by name beats clCreateSubBuffer failing on an extra that is the host backend's.
    if (ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        GGML_ABORT("ggml-opencl-tile32: weight %s is in buffer %s with usage %d, not WEIGHTS",
                   src0->name, ggml_backend_buffer_name(src0->buffer),
                   (int) ggml_backend_buffer_get_usage(src0->buffer));
    }
    if (src1->buffer && ggml_backend_buffer_is_host(src1->buffer)) {
        GGML_ABORT("ggml-opencl-tile32: activation %s is in host buffer %s, the leg needs an OpenCL buffer",
                   src1->name, ggml_backend_buffer_name(src1->buffer));
    }
    if (dst->buffer && ggml_backend_buffer_is_host(dst->buffer)) {
        GGML_ABORT("ggml-opencl-tile32: result %s is in host buffer %s, the leg needs an OpenCL buffer",
                   dst->name, ggml_backend_buffer_name(dst->buffer));
    }

    // The graph split admitted this tensor through supports_op, so this is normally a cached
    // hit whose one self-check already passed; a leg-level miss on another tensor turns
    // admission off for scheduling but does not clear this cached success. Only a scheduler
    // override (set_tensor_backend) or a mid-run release reaches this call without a cached
    // admission, and a failure there is not recoverable from inside compute: abort.
    ggml_opencl_tile32_views * v = ggml_opencl_tile32_views_locked(env, src0->buffer, src0, &reason);
    if (!v) {
        GGML_ABORT("ggml-opencl-tile32: import/views failed: %s", reason);
    }
    GGML_ASSERT(v->checked);
    cl_kernel kernel = ggml_opencl_tile32_kernel_locked(env);
    if (!kernel) {
        GGML_ABORT("ggml-opencl-tile32: v4 program build failed");
    }

    const int64_t K = src0->ne[0], M = src0->ne[1];
    cl_int err;
    const cl_buffer_region xregion = { (size_t) src1_off, (size_t) K * sizeof(float) };
    const cl_buffer_region dregion = { (size_t) dst_off,  (size_t) M * sizeof(float) };
    cl_mem xsub = clCreateSubBuffer(src1_buf, 0, CL_BUFFER_CREATE_TYPE_REGION, &xregion, &err);
    cl_mem dsub = xsub ? clCreateSubBuffer(dst_buf, 0, CL_BUFFER_CREATE_TYPE_REGION, &dregion, &err) : nullptr;
    if (!dsub) {
        if (xsub) { clReleaseMemObject(xsub); }
        GGML_ABORT("ggml-opencl-tile32: activation/result sub-buffers failed: %d", err);
    }
    cl_image_desc d;
    memset(&d, 0, sizeof(d));
    d.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    d.image_width = K / 4;
    d.buffer      = xsub;
    cl_image_format f = { CL_RGBA, CL_FLOAT };
    cl_mem ximg = clCreateImage(env->context, CL_MEM_READ_ONLY, &f, &d, nullptr, &err);
    if (!ximg) {
        clReleaseMemObject(dsub);
        clReleaseMemObject(xsub);
        GGML_ABORT("ggml-opencl-tile32: activation image failed: %d", err);
    }

    cl_uint poff = v->poff;
    cl_int ki = (cl_int) K, mi = (cl_int) M;
    const bool args_ok =
        ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 0, sizeof(cl_mem),  &v->wq),   "v4 arg wq")   &&
        ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 1, sizeof(cl_uint), &poff),    "v4 arg poff") &&
        ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 2, sizeof(cl_mem),  &ximg),    "v4 arg ximg") &&
        ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 3, sizeof(cl_mem),  &v->ws),   "v4 arg ws")   &&
        ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 4, sizeof(cl_mem),  &dsub),    "v4 arg dst")  &&
        ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 5, sizeof(cl_int),  &ki),      "v4 arg K")    &&
        ggml_opencl_tile32_cl_ok(clSetKernelArg(kernel, 6, sizeof(cl_int),  &mi),      "v4 arg M");
    if (!args_ok) {
        clReleaseMemObject(ximg);
        clReleaseMemObject(dsub);
        clReleaseMemObject(xsub);
        GGML_ABORT("ggml-opencl-tile32: v4 argument setup failed on %s", src0->name);
    }

    const size_t gws = ((M + 31) / 32) * 64, lws = 64;
    if (!ggml_opencl_tile32_cl_ok(clEnqueueNDRangeKernel(env->queue, kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr),
                                  "v4 enqueue")) {
        GGML_ABORT("ggml-opencl-tile32: v4 launch failed on %s", src0->name);
    }
    clReleaseMemObject(ximg);
    clReleaseMemObject(dsub);
    clReleaseMemObject(xsub);
}
