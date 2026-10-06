// Match the OpenCL version setup of ggml-opencl.cpp (see ggml-opencl-kalsa-diag.cpp):
// cl_version.h is include-guarded and the first inclusion in a TU fixes the API level.
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "ggml-opencl-tile32-internal.h"

#include "ggml-opencl-kalsa-diag.h"
#include "ggml-impl.h"

#include <fstream>
#include <iterator>
#include <mutex>
#include <string>

// The HOST leg's Q6_K mat-vec: which kernel runs and the geometry that kernel needs.
// Two callers share the answer - the compute path in ggml-opencl.cpp and the leg's read-only
// self-check in ggml-opencl-tile32-gate.cpp - so the check always launches the kernel the op
// launches: a second copy of the decision, or of either geometry, can only drift into a
// self-check that validates a kernel nothing runs.
//
// The fast arm is kernel_mul_mv_q6_K_f32_aos (mul_mv_q6_k_f32_aos.cl): the AoS layout read in
// place, where the plain kernel's block-slot-per-lane mapping and its per-row activation
// re-reads hold the 215 MB tied lm_head at ~9.5 GB/s on the S23. Its program and kernel live
// in the record the backend context owns (env->host_q6k_aos), built on first ask through the
// leg's optional build path (a failure here must not poison the device's kernel verdict), and
// the arm is dropped for the rest of that context if its self-check fails, so a miscompile on
// an old Adreno compiler costs the speedup, not the leg. A process-global pair would be worse
// than a leak: a second cl_context would be handed the first context's kernel.
//
// GGML_OPENCL_HOST_Q6K_PLAIN=1 forces the plain kernel_mul_mv_q6_K_f32 so both arms can be
// A/B'd in one binary.

// Rows a work group of the AoS kernel covers: Q6K_AOS_N_SIMDGROUP (2) subgroups of
// Q6K_AOS_N_SIMDWIDTH (64) lanes, Q6K_AOS_N_DST (16) rows each. These three are the kernel's
// own macros and must move with them.
#define HOST_Q6K_AOS_ROWS_PER_WG  32
#define HOST_Q6K_AOS_LOCAL_SIZE  128

// Read lazily, not at static init: a missing file must only fail the AoS arm.
static const std::string & host_q6k_aos_src(void) {
    static const std::string src = [] {
#ifdef GGML_OPENCL_EMBED_KERNELS
        return std::string {
            #include "mul_mv_q6_k_f32_aos.cl.h"
        };
#else
        std::ifstream f("mul_mv_q6_k_f32_aos.cl", std::ios::binary);
        return f ? std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>())
                 : std::string();
#endif
    }();
    return src;
}

// The AoS kernel of this context's record, built on first ask. Every failure below leaves the
// record dropped, so the plain kernel carries the rest of this context's weights.
static cl_kernel host_q6k_aos_kernel_locked(const ggml_opencl_tile32_env * env) {
    struct ggml_opencl_host_q6k_aos * aos = env->host_q6k_aos;
    if (!aos || aos->kernel || aos->dropped) {
        return aos ? aos->kernel : nullptr;
    }
    const char * src = host_q6k_aos_src().c_str();
    if (!*src) {
        GGML_LOG_WARN("ggml-opencl: HOST Q6_K AoS kernel source not found - the plain kernel stays\n");
        aos->dropped = true;
        return nullptr;
    }
    const char * opts = env->kernel_compile_opts ? env->kernel_compile_opts : "";
    if (!aos->program) {
        // build_program_optional, not build_program: a leg arm that can be dropped must not set
        // kernel_build_failed or record a fail verdict, which would take every OpenCL op off
        // the graph (and off the next process) over a kernel the plain one replaces.
        if (!env->build_program_optional) {
            GGML_LOG_WARN("ggml-opencl: HOST Q6_K AoS kernel has no build hook - the plain kernel stays\n");
            aos->dropped = true;
            return nullptr;
        }
        aos->program = env->build_program_optional(env->build_program_opaque, src, opts);
        if (!aos->program) {
            GGML_LOG_WARN("ggml-opencl: HOST Q6_K AoS program build failed - the plain kernel stays\n");
            aos->dropped = true;
            return nullptr;
        }
    }
    cl_int err = CL_SUCCESS;
    aos->kernel = clCreateKernel(aos->program, "kernel_mul_mv_q6_K_f32_aos", &err);
    if (!aos->kernel) {
        GGML_LOG_WARN("ggml-opencl: HOST Q6_K AoS clCreateKernel failed: %d - the plain kernel stays\n", err);
        // The build path caches the binary before this call, so a program this driver has just
        // refused would be reloaded from the cache by the next process: forget that entry.
        if (env->forget_program_optional) {
            env->forget_program_optional(env->build_program_opaque, src, opts);
        }
        aos->dropped = true;
    }
    return aos->kernel;
}

// Value-parsed: "1" and any other non-empty value but "0"/"false" force the plain kernel.
static bool host_q6k_plain_forced(void) {
    static const bool forced = ggml_opencl_env_value_enabled("GGML_OPENCL_HOST_Q6K_PLAIN");
    return forced;
}

// The AoS arm's grid: one work group of HOST_Q6K_AOS_ROWS_PER_WG rows per row group, the
// column in local dimension 1 (the admitted shape has exactly one).
static void host_q6k_aos_grid(const int64_t ne01, const int64_t ne11, size_t * gws, size_t * lws) {
    gws[0] = (size_t) ((ne01 + HOST_Q6K_AOS_ROWS_PER_WG - 1)/HOST_Q6K_AOS_ROWS_PER_WG)*HOST_Q6K_AOS_LOCAL_SIZE;
    gws[1] = (size_t) ne11;
    gws[2] = 1;
    lws[0] = HOST_Q6K_AOS_LOCAL_SIZE;
    lws[1] = 1;
    lws[2] = 1;
}

// The plain kernel: 64 lanes, 2 rows per work group (one per subgroup), the column index in
// local dimension 1 - the launch the HOST leg has always used for the lm_head.
static void host_q6k_plain_grid(const int64_t ne01, const int64_t ne11, size_t * gws, size_t * lws) {
    gws[0] = (size_t) ((ne01 + 1)/2)*64;
    gws[1] = (size_t) (ne11*2);
    gws[2] = 1;
    lws[0] = 64;
    lws[1] = 2;
    lws[2] = 1;
}

bool ggml_opencl_host_q6k_plan_locked(const struct ggml_opencl_tile32_env * env, cl_kernel plain_kernel,
                                      const int64_t ne01, const int64_t ne11,
                                      struct ggml_opencl_host_q6k_plan * plan) {
    // Both kernels index the activation column through dimension 1 and run one plane: the
    // admitted mat-vec shape is one f32 column (ne11 == 1, ne12 == ne13 == 1).
    if (!plain_kernel) {
        return false;
    }
    if (cl_kernel aos = host_q6k_plain_forced() ? nullptr : host_q6k_aos_kernel_locked(env)) {
        plan->kernel = aos;
        plan->aos    = true;
        host_q6k_aos_grid(ne01, ne11, plan->gws, plan->lws);
        return true;
    }
    plan->kernel = plain_kernel;
    plan->aos    = false;
    host_q6k_plain_grid(ne01, ne11, plan->gws, plan->lws);
    return true;
}

void ggml_opencl_host_q6k_plan_grid(const struct ggml_opencl_host_q6k_plan * plan, const int64_t ne01,
                                    const int64_t ne11, size_t * gws, size_t * lws) {
    if (plan->aos) {
        host_q6k_aos_grid(ne01, ne11, gws, lws);
    } else {
        host_q6k_plain_grid(ne01, ne11, gws, lws);
    }
}

void ggml_opencl_host_q6k_aos_drop_locked(const struct ggml_opencl_tile32_env * env, const char * reason) {
    struct ggml_opencl_host_q6k_aos * aos = env->host_q6k_aos;
    if (!aos || aos->dropped) {
        return;
    }
    aos->dropped = true;
    GGML_LOG_WARN("ggml-opencl: HOST Q6K AoS kernel dropped (%s) - the plain kernel runs it\n", reason);
}

bool ggml_opencl_host_q6k_plan(const struct ggml_opencl_tile32_env * env, cl_kernel plain_kernel,
                               const int64_t ne01, const int64_t ne11,
                               struct ggml_opencl_host_q6k_plan * plan) {
    std::lock_guard<std::mutex> lock(ggml_opencl_tile32_mutex());
    return ggml_opencl_host_q6k_plan_locked(env, plain_kernel, ne01, ne11, plan);
}
