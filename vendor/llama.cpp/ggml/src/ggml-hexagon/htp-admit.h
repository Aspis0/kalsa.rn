#pragma once

// Per-phase admission of ops to the Hexagon backend, for a model whose weights sit in the shared -HOST
// buffer that the CPU also reads (one weight copy). Pure predicates on ggml tensors: no session, no
// buffers, so the host test can compile this header alone (tests/test-hexagon-admit.cpp).
//
// Two questions, answered separately because the scheduler asks them at different points:
//   offload: should a weight in host memory send this op to the HTP?  (ggml_backend_sched, weight branch)
//   gate:    may the HTP take this op at all?  The scheduler also assigns ops by the graph callback's
//            layer pins and by expanding from HTP neighbours; both ask only supports_op, so a decode-shaped
//            op must be refused here to stay on the CPU.
//
// The one admission struct is ggml_hexagon_admission (ggml-hexagon.h): the same values reach the backend
// through the proc-address setter and through the env-seeded defaults.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>

static inline bool htp_admit_valid(const ggml_hexagon_admission & a) {
    return a.offload_min >= 1 && a.offload_ops <= GGML_HEXAGON_ADMIT_OPS_ALL && a.gate_min >= 0;
}

static inline bool htp_admit_same(const ggml_hexagon_admission & a, const ggml_hexagon_admission & b) {
    return a.offload_min == b.offload_min && a.offload_ops == b.offload_ops && a.gate_min == b.gate_min;
}

// The admission as a single 64-bit word, so a device context can publish it through one std::atomic and
// the scheduler threads read it without tears while a lane change rewrites it. The token thresholds are
// token counts and saturate at 2^30 - 1 (no graph has that many tokens); the ops mask takes 3 bits.
constexpr int      htp_admit_field_bits = 30;
constexpr uint64_t htp_admit_field_max  = (1ull << htp_admit_field_bits) - 1;

static inline uint64_t htp_admit_pack(const ggml_hexagon_admission & a) {
    const auto sat = [](int64_t v) {
        return v < 0 ? 0 : (v > int64_t(htp_admit_field_max) ? htp_admit_field_max : uint64_t(v));
    };
    return sat(a.offload_min) | (sat(a.gate_min) << htp_admit_field_bits) |
           (uint64_t(a.offload_ops & GGML_HEXAGON_ADMIT_OPS_ALL) << (2 * htp_admit_field_bits));
}

static inline ggml_hexagon_admission htp_admit_unpack(uint64_t v) {
    const ggml_hexagon_admission a = {
        int64_t(v & htp_admit_field_max),
        uint32_t((v >> (2 * htp_admit_field_bits)) & GGML_HEXAGON_ADMIT_OPS_ALL),
        int64_t((v >> htp_admit_field_bits) & htp_admit_field_max),
    };
    return a;
}

// Tokens of the batch an op works on, as far as the op's own shape defines them. Activations are
// [features, tokens] except the per-head ROPE and FLASH_ATTN_EXT tensors [head_dim, heads, tokens]; a
// per-head norm has tokens in ne[2] only when the batch is larger than one (at decode it reads as
// `heads`, which is harmless: only a node already on the HTP can pull it there).
static inline int64_t htp_admit_tokens_shape(const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_ROPE:
        case GGML_OP_FLASH_ATTN_EXT:
            return op->ne[2];
        case GGML_OP_SET_ROWS:
            return op->src[0]->ne[1];
        case GGML_OP_MUL_MAT:  // dst [rows, tokens, batch...]
        case GGML_OP_SSM_CONV: // dst [d_inner, tokens, sequences]
            return op->ne[1];
        default:
            return op->ne[2] > 1 ? op->ne[2] : op->ne[1];
    }
}

// Whether the op's own shape can be trusted for its batch dim. Views reshuffle the dims and element-wise
// compute inherits them from src0, so both keep the walk going; GGML_OP_NONE is a leaf, read as-is, and
// the ops in the switch define their batch dim themselves.
static inline bool htp_admit_tokens_inherited(const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_MUL_MAT: // dst [rows, tokens, batch...]
        case GGML_OP_GET_ROWS:
        case GGML_OP_SSM_CONV:
        case GGML_OP_ROPE:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_SET_ROWS:
            return false;
        default: // views and element-wise compute: the batch dim comes from elsewhere
            return true;
    }
}

// Batch tokens of an op. A MUL cannot count them from its own shape: the per-head activations of an
// attention are [head_dim, heads, tokens], which reads as [head_dim, heads] once tokens == 1, so a
// weighted per-head norm MUL at decode passed a threshold of 32 as if it were a 32-token batch. Count it
// on the activation's producer chain instead, at the nearest op whose token dim is defined by itself.
// The walk is bounded (a producer chain is a few ops long): past the limit, a deep or cyclic chain falls
// back to the op's own dims.
constexpr int htp_admit_walk_max = 8;

static inline int64_t htp_admit_tokens(const struct ggml_tensor * op) {
    if (op->op == GGML_OP_MUL) {
        const struct ggml_tensor * t = op->src[0];
        for (int hops = 0; t && htp_admit_tokens_inherited(t) && (t->src[0] || t->view_src); hops++) {
            if (hops == htp_admit_walk_max) {
                t = nullptr;
                break;
            }
            t = t->src[0] ? t->src[0] : t->view_src;
        }
        if (t) {
            return htp_admit_tokens_shape(t);
        }
    }
    return htp_admit_tokens_shape(op);
}

static inline bool htp_admit_offload(const ggml_hexagon_admission & a, const struct ggml_tensor * op,
                                     bool mm_flat, bool src1_is_weight) {
    if (htp_admit_tokens(op) < a.offload_min) {
        return false;
    }
    switch (op->op) {
        case GGML_OP_MUL_MAT:  return mm_flat && (a.offload_ops & GGML_HEXAGON_ADMIT_MUL_MAT);
        case GGML_OP_SSM_CONV: return (a.offload_ops & GGML_HEXAGON_ADMIT_SSM_CONV) != 0;
        case GGML_OP_MUL:      return src1_is_weight && (a.offload_ops & GGML_HEXAGON_ADMIT_MUL_WEIGHT);
        default:               return false;
    }
}

// htp_only: an operand lives in a Hexagon buffer the CPU cannot read. Such an op must stay supported,
// otherwise the scheduler aborts ("pre-allocated tensor in a buffer that cannot run the operation").
static inline bool htp_admit_gate(const ggml_hexagon_admission & a, const struct ggml_tensor * op, bool htp_only) {
    if (a.gate_min <= 0 || htp_only) {
        return true;
    }
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true; // no compute, they follow their source
        default:
            return htp_admit_tokens(op) >= a.gate_min;
    }
}

// src1 is a model weight (a view resolves to its source, as ggml_hexagon_op_htp_only does).
static inline bool htp_admit_src1_is_weight(const struct ggml_tensor * op) {
    const struct ggml_tensor * w = op->src[1];
    if (w && w->view_src) {
        w = w->view_src;
    }
    return w && w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// Whole-string integer in [min, max]; `out` is written only on success. Empty, leading whitespace, a '+'
// sign, trailing garbage, overflow and out-of-range all fail, so an env typo can never turn into a
// silently different value.
static inline bool htp_admit_parse_int(const char * str, int64_t min, int64_t max, int64_t * out) {
    if (str == nullptr || str[0] == '+' || isspace((unsigned char) str[0])) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const long long value = std::strtoll(str, &end, 10);
    if (errno == ERANGE || end == str || *end != '\0' || value < min || value > max) {
        return false;
    }
    *out = value;
    return true;
}
