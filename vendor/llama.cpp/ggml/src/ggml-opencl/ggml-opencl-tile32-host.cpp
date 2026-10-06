#include "ggml-opencl-tile32-host.h"

// Amendment G1, the admission half of the HOST weight leg. The E031.41 a7x lm-head guard in
// the native Q6_K path does not apply here on purpose: it exists because that compiler
// miscompiles the SoA flat K-quant kernels, which this route never touches - the lm_head
// runs on the plain kernel_mul_mv_q6_K_f32 over the imported block (Amendment G names this
// kernel explicitly, and the step-4b gate is "lm_head on the GPU").
//
// The leg replaces srcs only, never the dst, and it substitutes the weight slot: src0 for a
// MUL_MAT, src1 for an SSM_CONV (src0 is the conv input) and src1 for a MUL (the norm weight,
// ggml_mul(activation, weight) in llm_graph_context::build_norm). A tensor in a host buffer
// anywhere else would reach a kernel as a cast of the host backend's extra, so it is refused
// here (F4). For the same reason a HOST weight must be a plain contiguous tensor:
// its address is `data - dma-buf base`, which already carries view_offs, so a view would be
// read at the wrong offset and a non-contiguous one at the wrong stride (F5).

bool ggml_opencl_host_weights_buft(ggml_backend_buffer_type_t buft) {
    if (!buft || !ggml_backend_buft_is_host(buft)) {
        return false;
    }
    // the identity test, not the name: with one HOST object this refuses the weights buft too,
    // which is why Amendment I split the object in two
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev && buft == ggml_backend_dev_host_buffer_type(dev)) {
        return false;
    }
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg != nullptr &&
           ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_dmabuf") != nullptr;
}

bool ggml_opencl_host_weights_tensor(const struct ggml_tensor * t) {
    if (!t || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
        return false;
    }
    if (!ggml_opencl_host_weights_buft(ggml_backend_buffer_get_type(t->buffer))) {
        return false;
    }
    // llama's weight_buft_supported allocates a 0-byte dummy buffer: no data, usage still ANY
    return ggml_backend_buffer_get_size(t->buffer) == 0 ||
           ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// A tensor in a host buffer: the HOST leg imports host buffers, and its extra is the host
// backend's. The graph-local srcs of a placement probe have no buffer yet and pass.
static bool host_tensor(const struct ggml_tensor * t) {
    return t && t->buffer && ggml_backend_buffer_is_host(t->buffer);
}

bool ggml_opencl_host_foreign_tensor(const struct ggml_tensor * t) {
    return host_tensor(t) && !ggml_opencl_host_weights_tensor(t);
}

// The weight the leg will address must be a plain contiguous tensor, and the op must not put
// a host tensor in a slot the leg does not substitute.
static bool host_weight_ok(const struct ggml_tensor * op, const struct ggml_tensor * w) {
    if (w->view_src || !ggml_is_contiguous(w)) {
        return false;
    }
    return !host_tensor(op);
}

// The scan, in the order the original supports_op loop ran it: the first admitted HOST-W
// weight src decides, a tensor of the claimable buft that admission rejects refuses the op,
// and a buffered src that is neither on this device nor an admitted HOST-W weight refuses it
// too (a src in plain CPU memory would need a copy, which this leg never does).
enum ggml_opencl_host_leg ggml_opencl_host_leg_weight(ggml_backend_dev_t dev,
        const struct ggml_tensor * op, const struct ggml_tensor ** weight) {
    if (weight) {
        *weight = nullptr;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = op->src[i];
        if (!src) {
            continue;
        }
        if (!ggml_opencl_host_weights_tensor(src)) {
            // a tensor of the claimable weights HOST buft that is not an admitted HOST-W
            // weights tensor (a real buffer with COMPUTE or ANY usage) is not addressable,
            // and the generic switch has no host awareness: it would cast its extra as an
            // OpenCL one (Amendment I). The device host buft and the CPU buft are not
            // claimable, so the scheduler copies their tensors and the leg never sees them.
            if (src->buffer && ggml_opencl_host_weights_buft(ggml_backend_buffer_get_type(src->buffer))) {
                return GGML_OPENCL_HOST_LEG_REFUSE;
            }
            continue;
        }
        if (!ggml_opencl_host_op_ok(op, src)) {
            return GGML_OPENCL_HOST_LEG_REFUSE;
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const struct ggml_tensor * other = op->src[j];
            if (!other || other == src || !other->buffer) {
                continue;   // graph-local srcs are allocated into this split's own buffers
            }
            ggml_backend_buffer_type_t obuft = ggml_backend_buffer_get_type(other->buffer);
            if (ggml_backend_buft_get_device(obuft) == dev || ggml_opencl_host_weights_tensor(other)) {
                continue;
            }
            return GGML_OPENCL_HOST_LEG_REFUSE;
        }
        if (weight) {
            *weight = src;
        }
        return GGML_OPENCL_HOST_LEG_WEIGHT;
    }
    return GGML_OPENCL_HOST_LEG_OTHER;
}

// The placement half of the OpenCL device's offload_op (ggml-opencl.cpp's
// ggml_backend_opencl_device_offload_op calls this, and the device half - the import admission -
// is the caller's): the scan's WEIGHT verdict alone, so the scheduler's 1.off hand-off reaches
// OpenCL for exactly the ops this leg computes. The device admission is not repeated here because
// 1.off asks ggml_backend_supports_op on the same backend first, and that answer's WEIGHT branch
// is the admission itself; keeping the slot a pure ggml decision is what lets the host build test
// the wiring (tests/test-tile32-host-ops.cpp drives it through ggml_backend_dev_offload_op).
bool ggml_opencl_host_leg_offload(ggml_backend_dev_t dev, const struct ggml_tensor * op,
        const struct ggml_tensor ** weight) {
    return ggml_opencl_host_leg_weight(dev, op, weight) == GGML_OPENCL_HOST_LEG_WEIGHT;
}

bool ggml_opencl_host_op_ok(const struct ggml_tensor * op, const struct ggml_tensor * w) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            {
                if (w != op->src[0] || !host_weight_ok(op, w) || op->type != GGML_TYPE_F32) {
                    return false;
                }
                if (w->type != GGML_TYPE_Q6_K || w->ne[2] != 1 || w->ne[3] != 1) {
                    return false;
                }
                const ggml_tensor * x = op->src[1];
                return x && x->type == GGML_TYPE_F32 && !host_tensor(x) &&
                       x->ne[1] == 1 && x->ne[2] == 1 && x->ne[3] == 1 &&
                       x->nb[0] == sizeof(float) && ggml_is_contiguous(op);
            }
        case GGML_OP_MUL:
            {
                // The norm form build_norm emits (llama-graph.cpp; the LFM2 norms of
                // src/models/lfm2.cpp): the weight is src1, a broadcast row of the activation's
                // ne[0] - {n_embd, 1, 1, 1} against {n_embd, n_tokens, ...} for RMS norm. A weight
                // on the other side would be the activation the leg cannot substitute, and a wider
                // one is not the row the kernel broadcasts.
                if (w != op->src[1] || !host_weight_ok(op, w)) {
                    return false;
                }
                const ggml_tensor * x = op->src[0];
                if (!x || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1 || w->ne[0] != x->ne[0]) {
                    return false;
                }
                return x->type == w->type && x->type == op->type && x->type == GGML_TYPE_F32;
            }
        case GGML_OP_SSM_CONV:
            {
                // the SSM_CONV weight is src1 by construction (src0 is its input), so the conv
                // state can never be the HOST tensor the leg is asked about
                if (w != op->src[1] || !host_weight_ok(op, w)) {
                    return false;
                }
                // the upstream ssm_conv kernels are unverified for more than one sequence
                // (their per-sequence indexing is under review), so only n_s == 1 is admitted
                if (op->src[0]->ne[2] != 1) {
                    return false;
                }
                return op->src[0]->type == op->src[1]->type &&
                       op->src[0]->type == op->type &&
                       op->src[0]->type == GGML_TYPE_F32;
            }
        default:
            return false;
    }
}

// No fused path substitutes its dst or its intermediate outputs, so those must never sit in a
// host buffer. The srcs follow `substitutes`: the norm-family fused kernels read an admitted
// HOST-W weight through the import (ggml_opencl_fused_src) and may keep one, while every other
// fused kernel casts the extras as OpenCL ones - a host tensor's extra is its own backend's,
// and the cast hands a host pointer to clSetKernelArg as a cl_mem, silent garbage with no CL
// error (the one-copy G arm read the f32 norms that way: KLD 9.85). Those fusions fail closed
// here, and even a substituting one refuses a host tensor the leg did not admit. `nodes` is
// the fused range, the graph's node array at node_idx, so this TU needs no cgraph internals.
bool ggml_opencl_fuse_operands_ok(const struct ggml_tensor * const * nodes, int n_ops, bool substitutes) {
    for (int i = 0; i < n_ops; i++) {
        const struct ggml_tensor * node = nodes[i];
        if (node->buffer && ggml_backend_buffer_is_host(node->buffer)) {
            return false;
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const struct ggml_tensor * src = node->src[j];
            if (!src || !src->buffer || !ggml_backend_buffer_is_host(src->buffer)) {
                continue;
            }
            if (!(substitutes && ggml_opencl_host_weights_tensor(src))) {
                return false;
            }
        }
    }
    return true;
}
