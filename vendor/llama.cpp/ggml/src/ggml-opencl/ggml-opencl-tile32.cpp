// Match the OpenCL version setup of ggml-opencl.cpp (see ggml-opencl-kalsa-diag.cpp):
// cl_version.h is include-guarded and the first inclusion in a TU fixes the API level.
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "ggml-opencl-tile32-internal.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-q4_0-tile32.h"

#include <algorithm>
#include <cstring>
#include <map>

std::mutex & ggml_opencl_tile32_mutex(void) {
    static std::mutex mtx;
    return mtx;
}

bool ggml_opencl_tile32_cl_ok(cl_int err, const char * what) {
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml-opencl-tile32: %s failed: %d\n", what, err);
        return false;
    }
    return true;
}

namespace {

// Same layout as cl_mem_ext_host_ptr (cl_qcom_ext_host_ptr): the ion and dmabuf
// variants share it, allocation_type picks the route.
struct qcom_ext_host_ptr {
    cl_uint allocation_type;     // 0x411D cl_qcom_dmabuf_host_ptr
    cl_uint host_cache_policy;
    int     filedesc;
    void *  hostptr;
};

// cl_qcom_ext_host_ptr cache policies; the spike measured these on the S23 (probe.c). A
// cached rpcmem block may be imported io-coherent or write-back, an uncached one only
// uncached.
const cl_uint POLICY_IOCOHERENT = 0x40A9;
const cl_uint POLICY_WRITEBACK  = 0x40A5;
const cl_uint POLICY_UNCACHED   = 0x40A4;

struct tile32_import {
    cl_mem buf = nullptr;    // the dma-buf import of the whole block (TILE32 or HOST)
    int    fd  = -1;
    void * base = nullptr;
    size_t size = 0;
    std::map<const ggml_tensor *, ggml_opencl_tile32_views> tensors;
    std::map<const ggml_tensor *, ggml_opencl_tile32_host_view> hosts;
};

std::map<ggml_backend_buffer_t, tile32_import> g_imports;

void release_views(ggml_opencl_tile32_views & v) {
    if (v.ws)  { clReleaseMemObject(v.ws);  v.ws  = nullptr; }
    if (v.wq)  { clReleaseMemObject(v.wq);  v.wq  = nullptr; }
    if (v.sub) { clReleaseMemObject(v.sub); v.sub = nullptr; }
}

void release_import(tile32_import & imp) {
    for (auto & t : imp.tensors) {
        release_views(t.second);
    }
    imp.tensors.clear();
    // host views borrow imp.buf and hold no cl_mem of their own
    imp.hosts.clear();
    if (imp.buf) {
        clReleaseMemObject(imp.buf);
        imp.buf = nullptr;
    }
}

// The accessor Hexagon's registry resolves. G5: it also reports the rpcmem allocation's
// cache policy, so the import can match it instead of forcing io-coherent on every block.
using dmabuf_fn = bool (*)(ggml_backend_buffer_t, int *, void **, size_t *, bool *);

dmabuf_fn buffer_dmabuf_fn(ggml_backend_buffer_t buffer) {
    ggml_backend_reg_t reg = buffer->buft->device ? buffer->buft->device->reg : nullptr;
    return reg && reg->iface.get_proc_address
        ? (dmabuf_fn) reg->iface.get_proc_address(reg, "ggml_backend_buffer_dmabuf") : nullptr;
}

// Lazy per-buffer import (E1): the fd is per buffer, not per buft, so the whole block is
// imported once, on the first tensor that needs it. The block is probed on every call to
// detect a recycled buffer address whose old import outlived a model unload (the binding
// skipped the release call).
tile32_import * import_locked(const ggml_opencl_tile32_env * env, ggml_backend_buffer_t buffer, const char ** reason) {
    dmabuf_fn fn = buffer_dmabuf_fn(buffer);
    int fd = -1;
    void * base = nullptr;
    size_t size = 0;
    bool uncached = false;
    const bool have_block = fn && fn(buffer, &fd, &base, &size, &uncached);

    auto it = g_imports.find(buffer);
    if (it != g_imports.end()) {
        if (have_block && fd == it->second.fd && base == it->second.base && size == it->second.size) {
            return &it->second;
        }
#ifndef NDEBUG
        GGML_LOG_WARN("ggml-opencl-tile32: import for buffer %p is stale (fd %d -> %d) - "
                      "the binding did not release imports before the old buffer was freed\n",
                      (void *) buffer, it->second.fd, have_block ? fd : -1);
#endif
        release_import(it->second);
        g_imports.erase(it);
    }
    if (!have_block) {
        *reason = "the buffer's device has no dma-buf accessor for this buffer";
        return nullptr;
    }

    // the import asks for the dma-buf block, not the ggml allocation: that is the size
    // the driver has to place in GPU memory
    const char * why = nullptr;
    if (!ggml_opencl_tile32_facts_ok(&env->facts, size, &why)) {
        *reason = why;
        return nullptr;
    }

    tile32_import imp;
    imp.fd   = fd;
    imp.base = base;
    imp.size = size;

    // G5: the mapping policy has to match the rpcmem allocation. An uncached block has no
    // CPU-cached view, so io-coherent or write-back would map memory the CPU does not see the
    // same way; the uncached policy is the only correct one and has no fallback.
    // F3: a HOST block holds native-layout weights the loader wrote once (the flush is
    // opt-in and off by default), so a write-back import could serve the GPU stale bytes with
    // no other detector; the HOST route takes io-coherent or uncached and nothing else, while
    // TILE32 keeps its write-back fallback.
    const bool host_route = ggml_backend_buffer_is_host(buffer);
    const cl_uint policies[2] = { uncached ? POLICY_UNCACHED : POLICY_IOCOHERENT, POLICY_WRITEBACK };
    const char *  names[2]    = { uncached ? "uncached" : "io-coherent", "write-back" };
    const int     n_policies  = (uncached || host_route) ? 1 : 2;

    const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | (cl_mem_flags) (1 << 29) /* CL_MEM_EXT_HOST_PTR_QCOM */;
    cl_int err = CL_SUCCESS;
    const char * policy = names[0];
    for (int p = 0; p < n_policies && !imp.buf; p++) {
        qcom_ext_host_ptr hp = { 0x411D, policies[p], imp.fd, imp.base };
        policy = names[p];
        imp.buf = clCreateBuffer(env->context, flags, imp.size, &hp, &err);
    }
    if (!imp.buf) {
        if (host_route) {
            *reason = uncached ? "HOST import: the uncached mapping was refused"
                               : "HOST import: the io-coherent mapping was refused (no write-back fallback)";
        } else {
            *reason = uncached ? "uncached buffer: the dmabuf-host-ptr route refused it"
                               : "cl_qcom_dmabuf_host_ptr import failed";
        }
        GGML_LOG_INFO("ggml-opencl-tile32: %s: %d\n", *reason, err);
        return nullptr;
    }
    GGML_LOG_INFO("ggml-opencl-tile32: imported %s buffer of %zu bytes (fd %d, %s)\n",
                  ggml_backend_buffer_name(buffer), imp.size, imp.fd, policy);
    return &g_imports.insert({buffer, imp}).first->second;
}

ggml_opencl_tile32_views * views_locked(const ggml_opencl_tile32_env * env, tile32_import * imp,
                                        const ggml_tensor * w, const char ** reason) {
    const uint8_t * wbytes = (const uint8_t *) w->data;
    if (!wbytes || wbytes < (const uint8_t *) imp->base) {
        *reason = "tensor not inside the imported block";
        return nullptr;
    }
    const size_t toff   = wbytes - (const uint8_t *) imp->base;
    const size_t tbytes = ggml_q4_0_tile32_nbytes(w->ne[0], w->ne[1]);
    if (toff % 16 != 0 || toff > imp->size || tbytes > imp->size - toff) {
        *reason = "tensor region not 16-byte aligned or outside the imported block";
        return nullptr;
    }

    auto it = imp->tensors.find(w);
    if (it != imp->tensors.end()) {
        ggml_opencl_tile32_views & old = it->second;
        if (old.toff == toff &&
            memcmp(old.ne, w->ne, sizeof(old.ne)) == 0 &&
            memcmp(old.nb, w->nb, sizeof(old.nb)) == 0) {
            return &old;
        }
        // the tensor pointer was recycled for another weight: the views point at the old bytes
        release_views(old);
        imp->tensors.erase(it);
    }

    // G6: a sub-buffer that backs an image must satisfy both the buffer alignment and the
    // image base address alignment (the latter is a byte figure; the row-pitch query, which
    // this is not, is the pixel one). 16 keeps the pixel offset exact, and the mask below
    // needs the two-query maximum to be a power of two.
    const size_t align = std::max(std::max(env->mem_base_align, env->image_base_align), (size_t) 16);
    GGML_ASSERT((align & (align - 1)) == 0);
    const size_t base_off = toff & ~(align - 1);
    const cl_buffer_region region = { base_off, tbytes + (toff - base_off) };
    const size_t px = region.size / 16;
    if (px * 2 > env->image_max_buffer_size) {
        GGML_LOG_INFO("ggml-opencl-tile32: image needs %zu px, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE %zu\n",
                      px * 2, env->image_max_buffer_size);
        *reason = "tensor region above CL_DEVICE_IMAGE_MAX_BUFFER_SIZE";
        return nullptr;
    }

    ggml_opencl_tile32_views v;
    v.poff = (cl_uint) ((toff - base_off) / 16);
    v.host_tiles = wbytes;
    v.toff = toff;
    memcpy(v.ne, w->ne, sizeof(v.ne));
    memcpy(v.nb, w->nb, sizeof(v.nb));
    cl_int err;
    v.sub = clCreateSubBuffer(imp->buf, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    if (v.sub) {
        cl_image_desc d;
        memset(&d, 0, sizeof(d));
        d.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        d.image_width = px;
        d.buffer      = v.sub;
        cl_image_format fq = { CL_RGBA, CL_UNSIGNED_INT32 };
        v.wq = clCreateImage(env->context, CL_MEM_READ_ONLY, &fq, &d, nullptr, &err);
        if (v.wq) {
            cl_image_format fs = { CL_RGBA, CL_HALF_FLOAT };
            d.image_width = px * 2;
            v.ws = clCreateImage(env->context, CL_MEM_READ_ONLY, &fs, &d, nullptr, &err);
        }
    }
    if (!v.ws) {
        release_views(v);
        GGML_LOG_INFO("ggml-opencl-tile32: tensor views failed: %d\n", err);
        *reason = "per-tensor image views failed";
        return nullptr;
    }
    return &imp->tensors.insert({w, v}).first->second;
}

// The native-layout address of one HOST tensor in the imported block (Amendment G1): no
// sub-buffer and no images, the plain kernels read {imp->buf, data - base} directly. The
// reuse key matches views_locked so a recycled tensor pointer rebuilds instead of serving
// the old tensor's address.
ggml_opencl_tile32_host_view * host_views_locked(tile32_import * imp, const ggml_tensor * w, const char ** reason) {
    const uint8_t * wbytes = (const uint8_t *) w->data;
    if (!wbytes || wbytes < (const uint8_t *) imp->base) {
        *reason = "tensor not inside the imported block";
        return nullptr;
    }
    const size_t toff   = wbytes - (const uint8_t *) imp->base;
    const size_t tbytes = ggml_nbytes(w);
    // F11: the float4 kernels (kernel_mul_row and friends) load the weight as global float4,
    // so a misaligned tensor offset is a fault; the TILE32 clause is not optional here.
    if (toff % 16 != 0 || toff > imp->size || tbytes > imp->size - toff) {
        *reason = "tensor region not 16-byte aligned or outside the imported block";
        return nullptr;
    }

    auto it = imp->hosts.find(w);
    if (it != imp->hosts.end()) {
        ggml_opencl_tile32_host_view & old = it->second;
        if (old.toff == toff &&
            memcmp(old.ne, w->ne, sizeof(old.ne)) == 0 &&
            memcmp(old.nb, w->nb, sizeof(old.nb)) == 0) {
            return &old;
        }
        // the tensor pointer was recycled for another weight: the cached address is stale
        imp->hosts.erase(it);
    }

    ggml_opencl_tile32_host_view v;
    v.buf = imp->buf;
    v.off = toff;
    v.toff = toff;
    memcpy(v.ne, w->ne, sizeof(v.ne));
    memcpy(v.nb, w->nb, sizeof(v.nb));
    return &imp->hosts.insert({w, v}).first->second;
}

// F3: the Q6_K lm_head route's self-check, run once per tensor and cached in the host view,
// like the TILE32 views' `checked`. A miss is logged once here; the caller declines that op.
// Only Q6_K is checked: the f32 MUL/SSM_CONV weights read straight through with no packing
// step, so the coherent-only policy in import_locked is what guards those bytes.
// The check runs the kernel the compute path plans (ggml_opencl_host_q6k_plan): an AoS arm
// that fails it - a miscompile, say - is dropped for the process and the plain kernel is
// checked and kept, so the failure costs the speedup instead of the leg.
bool host_checked_locked(const ggml_opencl_tile32_env * env, ggml_opencl_tile32_host_view * v,
                         const ggml_tensor * w, cl_kernel plain_kernel) {
    if (v->checked) {
        return v->check_ok;
    }
    v->checked = true;
    if (w->type != GGML_TYPE_Q6_K) {
        v->check_ok = true;
        return true;
    }
    // Amendment I: the handles are null when the kernels were never loaded (they load lazily from
    // the OpenCL buffer allocator), so say that instead of ANDing it away as a failed check.
    if (!plain_kernel) {
        GGML_LOG_WARN("ggml-opencl: HOST self-check for %s not run: the Q6_K kernel is not loaded\n", w->name);
        v->check_ok = false;
        return false;
    }

    struct ggml_opencl_host_q6k_plan plan;
    if (!ggml_opencl_host_q6k_plan_locked(env, plain_kernel, w->ne[1], 1, &plan)) {
        GGML_LOG_WARN("ggml-opencl: HOST self-check for %s not run: no Q6_K kernel to check\n", w->name);
        v->check_ok = false;
        return false;
    }
    v->check_ok = ggml_opencl_host_self_check_locked(env, v, w, &plan);
    if (!v->check_ok && plan.aos) {
        // The AoS arm is the only kernel here that has not already carried this leg, so a miss
        // that is not an import problem is its own: drop it and check the plain kernel.
        ggml_opencl_host_q6k_aos_drop_locked(env, "self-check failed");
        if (ggml_opencl_host_q6k_plan_locked(env, plain_kernel, w->ne[1], 1, &plan)) {
            v->check_ok = ggml_opencl_host_self_check_locked(env, v, w, &plan);
        }
    }
    if (!v->check_ok) {
        GGML_LOG_WARN("ggml-opencl: HOST self-check failed for %s - the HOST leg declines it\n", w->name);
    }
    return v->check_ok;
}

} // namespace

ggml_opencl_tile32_views * ggml_opencl_tile32_views_locked(const ggml_opencl_tile32_env * env,
                                                           ggml_backend_buffer_t buffer,
                                                           const ggml_tensor * w, const char ** reason) {
    tile32_import * imp = import_locked(env, buffer, reason);
    ggml_opencl_tile32_views * v = imp ? views_locked(env, imp, w, reason) : nullptr;
    if (v && !v->checked) {
        cl_kernel kernel = ggml_opencl_tile32_kernel_locked(env);
        if (!kernel) {
            *reason = "v4 program build failed";
        } else if (ggml_opencl_tile32_self_check_locked(env, v, w, kernel)) {
            v->checked = true;
        } else {
            *reason = "self-check failed";
        }
    }
    if (!v || !v->checked) {
        ggml_opencl_tile32_disable_locked(*reason ? *reason : "TILE32 import failed");
        return nullptr;
    }
    return v;
}

bool ggml_opencl_tile32_weight_ready(const ggml_opencl_tile32_env * env, const ggml_tensor * w) {
    std::lock_guard<std::mutex> lock(ggml_opencl_tile32_mutex());
    if (!ggml_opencl_tile32_gate_locked(env)) {
        return false;
    }
    // llama's weight_buft_supported asks with a 0-byte dummy buffer and no data: those
    // placement probes have nothing to import, so the device facts are the whole answer.
    if (!w->buffer || w->buffer->size == 0 || !w->data) {
        return true;
    }
    const char * reason = nullptr;
    return ggml_opencl_tile32_views_locked(env, w->buffer, w, &reason) != nullptr;
}

bool ggml_opencl_host_view(const ggml_opencl_tile32_env * env, const ggml_tensor * w,
                           cl_mem * buf, cl_ulong * off, const char ** reason) {
    std::lock_guard<std::mutex> lock(ggml_opencl_tile32_mutex());
    if (!ggml_opencl_tile32_gate_locked(env)) {
        *reason = "the R2 gate is not green";
        return false;
    }
    tile32_import * imp = import_locked(env, w->buffer, reason);
    if (!imp) {
        return false;
    }
    ggml_opencl_tile32_host_view * v = host_views_locked(imp, w, reason);
    if (!v) {
        return false;
    }
    *buf = v->buf;
    *off = v->off;
    return true;
}

bool ggml_opencl_host_weight_ready(const ggml_opencl_tile32_env * env, const ggml_tensor * w,
                                   cl_kernel plain_q6k_kernel, const char ** reason) {
    std::lock_guard<std::mutex> lock(ggml_opencl_tile32_mutex());
    if (!ggml_opencl_tile32_gate_locked(env)) {
        *reason = "the R2 gate is not green";
        return false;
    }
    tile32_import * imp = import_locked(env, w->buffer, reason);
    if (!imp) {
        return false;
    }
    ggml_opencl_tile32_host_view * v = host_views_locked(imp, w, reason);
    if (!v) {
        return false;
    }
    if (!host_checked_locked(env, v, w, plain_q6k_kernel)) {
        *reason = "self-check failed";
        return false;
    }
    return true;
}

void ggml_backend_opencl_release_imports(void) {
    std::lock_guard<std::mutex> lock(ggml_opencl_tile32_mutex());
    // The views were handed to the leg's queue: a compute may still be in flight if the caller
    // did not synchronize the backend. Finish the queue before releasing any cl_mem.
    if (cl_command_queue queue = ggml_opencl_tile32_queue_locked()) {
        ggml_opencl_tile32_cl_ok(clFinish(queue), "release_imports clFinish");
    }
    size_t n_tensors = 0;
    for (auto & kv : g_imports) {
        n_tensors += kv.second.tensors.size();
        release_import(kv.second);
    }
    if (!g_imports.empty()) {
        GGML_LOG_INFO("ggml-opencl-tile32: released %zu imports (%zu tensor views)\n",
                      g_imports.size(), n_tensors);
    }
    g_imports.clear();
}
