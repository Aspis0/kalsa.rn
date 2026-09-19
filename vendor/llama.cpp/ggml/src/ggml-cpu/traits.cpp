#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "repack-q23k.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
            // Q2_K_8x4 / Q3_K_8x4 storage is tiled. Returning false here used to fall
            // through ggml_compute_forward into the native mul_mat path, which reads
            // those bytes as block_q2_K / block_q3_K -> silent wrong output.
            if (ggml_cpu_repack_extra_is_q23k_8x4(tensor_traits)) {
                GGML_ABORT("q23k_8x4 extra declined op=%s; refusing native fallthrough on tiled layout",
                           ggml_op_name(op->op));
            }
        }
    }
    if (op->src[0] && ggml_cpu_repack_extra_is_q23k_8x4(op->src[0]->extra)) {
        GGML_ABORT("q23k_8x4 tiled src0 reached native compute (op=%s); extra path did not handle it",
                   ggml_op_name(op->op));
    }
    return false;
}

bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                return true;
            }
        }
    }
    return false;
}
