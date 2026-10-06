#pragma once

// The OpenCL TILE32 weight leg (Amendment E) and its HOST service (Amendment G, step 4b):
// claim the Hexagon TILE32 and HOST bufts when the R2 gate is green, lazily import each
// buffer with cl_qcom_dmabuf_host_ptr - both through one import map - build the verbatim v4
// mat-vec program on first use, address native-layout HOST weights in place, and release
// every import on one explicit call.
// ggml-opencl.cpp owns the hooks (supports_buft / supports_op / ggml_cl_mul_mat); the leg
// never touches the backend context struct - the caller hands over an env snapshot.
//
// Release contract for the binding: resolve the proc address
//   ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("OpenCL"),
//                                     "ggml_backend_opencl_release_imports")
// and call it before llama_model_free. The caller must have synchronized the OpenCL backend
// first (no compute in flight: the views are shared with the scheduler's buffers); the call
// additionally clFinish()es the leg's queue before releasing. Imports hold a dma-buf
// reference on the weights, so skipping the call leaks every TILE32 and HOST buffer for the
// rest of the process.
//
// Like ggml-opencl-kalsa-diag.h: a .cpp including this header must define
// CL_TARGET_OPENCL_VERSION (and CL_USE_DEPRECATED_OPENCL_1_2_APIS) before any CL/cl.h,
// because cl_version.h is include-guarded and the first TU to include it wins.
#include <CL/cl.h>

#include <stddef.h>
#include <stdint.h>

#include "ggml-opencl-tile32-facts.h"

struct ggml_tensor;

// The HOST leg's optional AoS Q6_K arm, one record per backend context: the leg fills it on
// first ask through the env below, and the context that owns it releases it in its free path.
// Both handles belong to that context's cl_context and device - a program and a kernel cannot
// cross contexts - so the record replaces the process-global pair the arm used to have.
// `dropped` is the arm's verdict for the context's remaining life: a build, create or
// self-check miss means the plain kernel runs the rest of its weights.
struct ggml_opencl_host_q6k_aos {
    cl_program program = nullptr;
    cl_kernel  kernel  = nullptr;
    bool       dropped = false;
};

// Everything the leg needs from the backend context, snapshotted by ggml-opencl.cpp.
// Borrowed pointers (names, opts) live as long as the context, i.e. the process.
struct ggml_opencl_tile32_env {
    cl_context       context;
    cl_command_queue queue;
    cl_device_id     device;
    struct ggml_opencl_tile32_facts facts;
    size_t image_max_buffer_size;   // CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, pixels
    size_t mem_base_align;          // CL_DEVICE_MEM_BASE_ADDR_ALIGN, bytes
    size_t image_base_align;        // CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, bytes
                                    // (the row-pitch query is in pixels, the base address is
                                    // not; both sub-buffer origins respect it)
    const char * kernel_compile_opts; // the engine's general build opts
    // The engine's standard program build path (on-disk binary cache, fail verdict);
    // opaque is the backend context pointer ggml-opencl.cpp filled this env from.
    cl_program (*build_program)(void * opaque, const char * src, const char * opts);
    // The same path without the verdict: for a program the leg may drop, so a build failure
    // costs the arm and neither kernel_build_failed nor the cache's fail record.
    cl_program (*build_program_optional)(void * opaque, const char * src, const char * opts);
    // Forget the binary-cache entry of an optional program whose kernel clCreateKernel refused:
    // a binary this driver just found useless must not be reloaded by the next process.
    void (*forget_program_optional)(void * opaque, const char * src, const char * opts);
    void * build_program_opaque;
    // The AoS arm of the context this env came from, or nullptr when the caller offers none
    // (the plan then names the plain kernel). Owned by the caller, released with the context.
    struct ggml_opencl_host_q6k_aos * host_q6k_aos;
};

// The E4 gate, evaluated once per device: device facts plus an image1d_buffer view canary
// (RGBA/UINT32 and RGBA/HALF_FLOAT over a sub-buffer). Passes get one INFO line, misses one
// INFO line with the reason. The buffer-size clause (max_alloc_size) is re-asked at import
// time, when the first real buffer exists. No import happens here. This is the buft-level
// answer (supports_buft); an op with a real weight goes through weight_ready below.
bool ggml_opencl_tile32_ready(const struct ggml_opencl_tile32_env * env);

// Admission for one TILE32 weight at placement time (G2): device gate, then the lazy import,
// the per-tensor views, the v4 program and the one read-only self-check, all cached per
// tensor. A zero-size placement probe (llama's weight_buft_supported allocates a 0-byte
// dummy buffer) has no data to read and is answered from the device facts alone. False on any
// miss, with the leg disabled for the process, so the scheduler hands the op to the CPU tile
// reader.
bool ggml_opencl_tile32_weight_ready(const struct ggml_opencl_tile32_env * env,
                                     const struct ggml_tensor * w);

// The decode mat-vec for a TILE32 src0 (ne11 == 1, admission enforced by supports_op).
// src0->extra is Hexagon's and stays untouched: weights are addressed by
// tensor->data - dma-buf base. src1_off / dst_off are extra->offset + view_offs in the
// OpenCL buffers holding the activation column and the result. src1 and dst are passed for
// the compute-time re-check of the admitted shape.
void ggml_cl_mul_mat_tile32(const struct ggml_opencl_tile32_env * env,
                            const struct ggml_tensor * src0,
                            const struct ggml_tensor * src1,
                            const struct ggml_tensor * dst,
                            cl_mem src1_buf, cl_ulong src1_off, cl_mem dst_buf, cl_ulong dst_off);

// The address of one native-layout tensor of a HOST buffer (Amendment G1), through the same
// lazy import and map as TILE32: *buf/*off are what the plain kernels pass as
// {data_device, offset}. The buffer must exist - callers answer the 0-byte placement probes
// from the gate alone. False on any miss with *reason set; a miss declines the one op and
// does not disable the leg, because a HOST import failure must not take the admitted TILE32
// tensors down. The compute path uses this after admission; the self-check runs in
// weight_ready, not here.
bool ggml_opencl_host_view(const struct ggml_opencl_tile32_env * env, const struct ggml_tensor * w,
                           cl_mem * buf, cl_ulong * off, const char ** reason);

// Admission for one HOST weight at placement time (G1/G2): the device gate, then the lazy
// import, the per-tensor address and - for a Q6_K lm_head weight - the one read-only
// self-check, all cached per tensor. plain_q6k_kernel is the engine's
// kernel_mul_mv_q6_K_f32, the leg's fallback arm: the self-check runs the kernel the compute
// path will launch (ggml_opencl_host_q6k_plan decides which), so a null handle declines a
// Q6_K weight only when neither arm is available. False on any miss with *reason set; a miss
// declines the one op and does not disable the leg.
bool ggml_opencl_host_weight_ready(const struct ggml_opencl_tile32_env * env,
                                   const struct ggml_tensor * w, cl_kernel plain_q6k_kernel,
                                   const char ** reason);

// The HOST leg's Q6_K mat-vec plan: the kernel the op runs and the geometry it needs for the
// row count it was asked for. The compute path and the read-only self-check share it, so the
// self-check cannot validate a kernel nothing launches; the check asks the same mapping for
// its own row slice through ggml_opencl_host_q6k_plan_grid (internal header).
struct ggml_opencl_host_q6k_plan {
    cl_kernel kernel = nullptr;
    bool      aos    = false;   // the AoS kernel is the fast arm; its self-check may drop it
    size_t    gws[3] = { 0, 0, 0 };
    size_t    lws[3] = { 0, 0, 0 };
};

// Plan the admitted HOST Q6_K mat-vec (2-D weight, one f32 column): the AoS kernel of this
// context's record when it built and has not been dropped, the plain kernel_mul_mv_q6_K_f32
// otherwise, and never the AoS one under GGML_OPENCL_HOST_Q6K_PLAIN=1. False when no kernel is
// available (the engine's plain kernel not loaded, or the AoS program build failed). Takes the
// leg mutex; the caller holds no other leg lock.
bool ggml_opencl_host_q6k_plan(const struct ggml_opencl_tile32_env * env, cl_kernel plain_kernel,
                               int64_t ne01, int64_t ne11, struct ggml_opencl_host_q6k_plan * plan);

// Drop every import and per-tensor view. See the release contract above.
void ggml_backend_opencl_release_imports(void);
