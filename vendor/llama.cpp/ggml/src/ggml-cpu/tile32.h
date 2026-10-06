// CPU dispatch for MUL_MAT over weights stored in a Q4_0 TILE32 buffer type: the CPU reads the
// device's tiled weight copy directly instead of keeping its own repacked copy (one-copy mode).
// ggml-cpu.cpp (device supports_op) and traits.cpp (compute_forward / work_size) route
// layout-keyed branches here; there is no entry in ggml_backend_cpu_get_extra_buffer_types
// because a TILE32 buffer is never CPU-allocated.

#pragma once

#include "ggml-cpu-impl.h"

bool ggml_cpu_tile32_supports_op(const struct ggml_tensor * op);
bool ggml_cpu_tile32_work_size(int n_threads, const struct ggml_tensor * op, size_t * size);
bool ggml_cpu_tile32_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);
