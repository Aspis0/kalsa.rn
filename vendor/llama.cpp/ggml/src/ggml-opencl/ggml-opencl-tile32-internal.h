#pragma once

// Internal surface of the OpenCL weight leg (Amendment E for TILE32, Amendment G step 4b for
// HOST). Four translation units divide it: ggml-opencl-tile32.cpp owns the lazy dma-buf
// imports, the per-tensor views and the release call, -gate.cpp the R2 verdict and the
// read-only self-check, -matvec.cpp the v4 program and the decode mat-vec, -host.cpp the
// op-admission predicate for HOST weights. ggml-opencl.cpp only calls the public header.
//
// The leg is a process singleton - the import map is keyed by buffer, the v4 program and
// kernel and the device verdict are global, and a second OpenCL device is refused by the gate
// - so a multi-device scheduler would need one leg each. The one exception is the HOST leg's
// AoS arm: its program, kernel and drop flag live in the record the backend context owns
// (ggml_opencl_host_q6k_aos), because a kernel belongs to one cl_context.
// Every _locked function requires the caller to hold ggml_opencl_tile32_mutex(), which
// guards all of the above.

#include "ggml-opencl-tile32.h"

#include "ggml-backend.h"

#include <CL/cl.h>

#include <mutex>

struct ggml_tensor;

std::mutex & ggml_opencl_tile32_mutex(void);

// Log the OpenCL code and report it as a bool: the exit every raw CL call in the leg uses.
bool ggml_opencl_tile32_cl_ok(cl_int err, const char * what);

// One tensor's views into the imported TILE32 block, built on first use.
struct ggml_opencl_tile32_views {
    cl_mem  sub  = nullptr;   // backing sub-buffer at the tensor's aligned base
    cl_mem  wq   = nullptr;   // quants, RGBA/UINT32, 1 px = 16 B
    cl_mem  ws   = nullptr;   // fp16 row scales, RGBA/HALF_FLOAT, 1 px = 8 B
    cl_uint poff = 0;         // pixel offset of the tensor region in wq
    // The same tiles in the host mapping of the import: the CPU tile reader's input in the
    // self-check, and the proof that the import maps the bytes the HTP tiled.
    const uint8_t * host_tiles = nullptr;
    // Reuse key: these views are cached under the tensor pointer, so a recycled pointer with
    // another weight must be told apart. toff is w->data - dma-buf base at build time, ne/nb
    // the shape then; any mismatch rebuilds instead of reading the old tensor's bytes.
    size_t  toff = 0;
    int64_t ne[GGML_MAX_DIMS] = {};
    size_t  nb[GGML_MAX_DIMS] = {};
    bool checked = false;     // the read-only self-check ran on this tensor
};

// One native-layout tensor's address in an imported HOST block (Amendment G1): the plain
// kernels read {buf, off} in place. buf is the shared import's cl_mem, not a reference of
// its own - nothing per-view to release. The reuse key matches ggml_opencl_tile32_views.
struct ggml_opencl_tile32_host_view {
    cl_mem  buf  = nullptr;
    cl_ulong off = 0;
    size_t  toff = 0;
    int64_t ne[GGML_MAX_DIMS] = {};
    size_t  nb[GGML_MAX_DIMS] = {};
    // F3: the once-per-tensor self-check verdict, cached like the TILE32 views' `checked`
    // and rebuilt under the same recycled-pointer rule.
    bool    checked  = false;
    bool    check_ok = false;
};

// Full admission for one tensor of an imported buffer: the lazy import, the per-tensor views,
// the v4 program and the one read-only self-check, all cached (an admitted tensor costs one
// map lookup). nullptr with *reason set on a miss; a miss also disables the leg, so the
// scheduler can hand the op to the CPU tile reader. The buffer-size clause of the gate is
// asked here, with the dma-buf block size, because no buffer exists when the one-time gate runs.
struct ggml_opencl_tile32_views * ggml_opencl_tile32_views_locked(
        const struct ggml_opencl_tile32_env * env, ggml_backend_buffer_t buffer,
        const struct ggml_tensor * w, const char ** reason);

// The E4 device verdict, evaluated once per device: facts plus the image1d_buffer view canary.
bool ggml_opencl_tile32_gate_locked(const struct ggml_opencl_tile32_env * env);

// The queue of the device the leg is bound to: the program, kernel and imports all live in
// that context, and the release path clFinish()es this queue before freeing any cl_mem.
// nullptr until a device passes the gate.
cl_command_queue ggml_opencl_tile32_queue_locked(void);

// Record a miss: the verdict stays false for the rest of the process and the first miss
// logs its reason. The caller still fails the operation in flight.
void ggml_opencl_tile32_disable_locked(const char * reason);

// Read-only self-check of one real tensor against the CPU tile reader, through v's views.
bool ggml_opencl_tile32_self_check_locked(const struct ggml_opencl_tile32_env * env,
        const struct ggml_opencl_tile32_views * v, const struct ggml_tensor * w, cl_kernel kernel);

// Read-only self-check of one HOST Q6_K weight (F3): the planned kernel on the first rows of
// the imported bytes against ggml's Q6_K dequant plus a double-accumulated dot on the same
// host pointer. The reference reads the tensor's own data, so a wrong or stale import mapping
// fails here instead of after placement. Kernel and row mapping come from `plan`, whose grid
// is the M-row one: the check asks ggml_opencl_host_q6k_plan_grid for its R-row slice. Requires
// the leg mutex.
bool ggml_opencl_host_self_check_locked(const struct ggml_opencl_tile32_env * env,
        const struct ggml_opencl_tile32_host_view * v, const struct ggml_tensor * w,
        const struct ggml_opencl_host_q6k_plan * plan);

// Plan the admitted HOST Q6_K mat-vec: the AoS kernel of this context's record when it built
// and has not been dropped, the plain kernel_mul_mv_q6_K_f32 otherwise, never the AoS one under
// GGML_OPENCL_HOST_Q6K_PLAIN=1. False when no kernel is available. The plan's gws/lws are the
// geometry for its ne01 rows; ggml_opencl_host_q6k_plan_grid re-asks that geometry for any row
// count under the same mapping. Requires the leg mutex; callers outside the leg use
// ggml_opencl_host_q6k_plan in ggml-opencl-tile32.h.
bool ggml_opencl_host_q6k_plan_locked(const struct ggml_opencl_tile32_env * env, cl_kernel plain_kernel,
        int64_t ne01, int64_t ne11, struct ggml_opencl_host_q6k_plan * plan);

// The plan's arm and row mapping for ne01 rows: the self-check launches its R-row slice with
// this, so its last work group is laid out exactly like the compute launch's and the kernel's
// ne01 argument guards the same tail. Pure: no lock, no state.
void ggml_opencl_host_q6k_plan_grid(const struct ggml_opencl_host_q6k_plan * plan, int64_t ne01,
        int64_t ne11, size_t * gws, size_t * lws);

// Drop this context's AoS arm after its self-check failed: every later tensor of the context
// plans the plain kernel instead of failing admission too. Requires the leg mutex.
void ggml_opencl_host_q6k_aos_drop_locked(const struct ggml_opencl_tile32_env * env, const char * reason);

// The v4 mat-vec program and kernel, built on first use.
cl_kernel ggml_opencl_tile32_kernel_locked(const struct ggml_opencl_tile32_env * env);
