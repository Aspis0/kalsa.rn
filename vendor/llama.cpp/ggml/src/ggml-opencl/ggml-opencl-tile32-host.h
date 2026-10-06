#pragma once

// The admission predicates of the HOST weight leg (Amendment G1, narrowed to one buffer type
// per role in Amendment I): which buffer types OpenCL may claim for weights, which tensors it
// may address through the dma-buf import map, and which ops it may compute over them. Pure
// ggml decisions, deliberately free of OpenCL types so the host build (GGML_OPENCL=OFF) can
// test them; the caller in ggml-opencl.cpp finds the weight, checks that every buffered src is
// addressable, and asks the import separately.
#include "ggml.h"
#include "ggml-backend.h"

struct ggml_tensor;

// True for a host buffer type the HOST leg may claim for weights. Amendment I: a device has
// two host buffer types; the one ggml_backend_dev_host_buffer_type returns is what llama hands
// to the CPU for its compute and output buffers, and claiming it would stop the scheduler from
// copying CPU-produced activations into OpenCL buffers. Only the other one - the weights buft -
// is claimable, and its registry must expose the generic dma-buf accessor.
bool ggml_opencl_host_weights_buft(ggml_backend_buffer_type_t buft);

// True for a tensor the HOST leg may address through the import map: its buffer is host, uses
// a weights buft, and carries WEIGHTS. This is the one predicate both admission (supports_op)
// and compute (the import site) call, so they cannot disagree; a zero-sized buffer is llama's
// placement probe, where usage is still ANY and the device facts are the whole answer. The
// exemption is shared: a probe has no data, so the import fails closed on it if one arrives.
bool ggml_opencl_host_weights_tensor(const struct ggml_tensor * t);

// True for a tensor in a host buffer that is not an admitted HOST-W weights tensor: its extra
// is the host backend's, and the OpenCL compute sites must abort by name before any cast of
// it. The scheduler copies a tensor in a buft OpenCL does not claim, so reaching compute with
// one of these is a placement override (G10).
bool ggml_opencl_host_foreign_tensor(const struct ggml_tensor * t);

// The HOST weight leg's src scan: what ggml_opencl_host_leg_weight found.
enum ggml_opencl_host_leg {
    GGML_OPENCL_HOST_LEG_OTHER,   // no admitted HOST-W weight src: not this leg's op
    GGML_OPENCL_HOST_LEG_REFUSE,  // a claimable HOST buft tensor admission rejects: refuse the op
    GGML_OPENCL_HOST_LEG_WEIGHT,  // the op is admitted on the weight returned in *weight
};

// The scan supports_op and offload_op share, so the two cannot disagree about which ops the
// HOST weight leg takes. GGML_OPENCL_HOST_LEG_WEIGHT needs the device half of the admission
// (import + self-check) before the op is answered; GGML_OPENCL_HOST_LEG_REFUSE means a src
// carries a host tensor the leg cannot address at all, and the caller must refuse the op
// instead of falling through to its generic switch. `weight` may be NULL when the caller
// only asks whether the leg wants the op.
enum ggml_opencl_host_leg ggml_opencl_host_leg_weight(ggml_backend_dev_t dev,
        const struct ggml_tensor * op, const struct ggml_tensor ** weight);

// The placement half of the OpenCL device's offload_op: true only for the scan's WEIGHT verdict,
// the ops the scheduler's 1.off hand-off may send to this device. ggml-opencl.cpp's
// ggml_backend_opencl_device_offload_op calls this and adds the device half (the import
// admission), which 1.off has already asked through supports_op on the same backend, so the slot
// itself stays a pure ggml decision the host build can wire and test. `weight` takes the admitted
// HOST-W tensor, as in the scan; it may be NULL.
bool ggml_opencl_host_leg_offload(ggml_backend_dev_t dev, const struct ggml_tensor * op,
        const struct ggml_tensor ** weight);

// True only for the three op forms whose kernels read a plain {cl_mem, offset} extra:
// the MUL of a norm pair (the weight in src1, a broadcast row of the activation's ne[0]),
// the single-sequence f32 SSM_CONV, and the flat Q6_K mat-vec.
// Everything else is refused on HOST, never converted: Q4_0/Q4_K/Q5_K exist only as SoA or
// Adreno-repacked forms in this build, and quantized GET_ROWS has no kernel. The MUL_MAT form
// is a mat-vec only (ne11 == 1), like the TILE32 leg: no GPU prefill. `w` is the HOST tensor
// the caller found among the op's srcs; the weight must be plain and contiguous, and no other
// slot the compute path cannot substitute (a MUL_MAT src1, any dst) may be a host tensor.
bool ggml_opencl_host_op_ok(const struct ggml_tensor * op, const struct ggml_tensor * w);

// True when the fused range starting at nodes[0] and spanning n_ops nodes may run as one
// kernel. No fused path substitutes its dst or its intermediate outputs, so any of those in a
// host buffer refuses the fusion. The srcs follow `substitutes`: a substituting fusion (the
// norm families) reads an admitted HOST-W weight through the import (ggml_opencl_fused_src)
// and may keep one, while every other fused kernel casts the extras as OpenCL ones, and a
// weights HOST tensor's extra is the host backend's - the cast hands a host pointer to
// clSetKernelArg as a cl_mem, which is silent garbage with no CL error (the one-copy G arm
// read the f32 norms that way: KLD 9.85). Those fusions fail closed here.
bool ggml_opencl_fuse_operands_ok(const struct ggml_tensor * const * nodes, int n_ops,
                                  bool substitutes);
