#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(void);

GGML_BACKEND_API bool ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

// Per-phase admission of ops to the HTP, for a model whose weights sit in the shared -HOST buffer.
// Reached at run time through ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_set_admission");
// the env vars GGML_OP_OFFLOAD_MIN_BATCH, GGML_HEXAGON_ADMIT_OPS and GGML_HEXAGON_ADMIT_GATE only seed it.
enum ggml_hexagon_admit_op {
    GGML_HEXAGON_ADMIT_MUL_MAT    = 1 << 0,
    GGML_HEXAGON_ADMIT_SSM_CONV   = 1 << 1,
    GGML_HEXAGON_ADMIT_MUL_WEIGHT = 1 << 2, // MUL whose src1 is a model weight (norm scale)
    GGML_HEXAGON_ADMIT_OPS_ALL    = 7,
};

struct ggml_hexagon_admission {
    int64_t  offload_min; // tokens an op needs before a host weight offloads it, >= 1
    uint32_t offload_ops; // ggml_hexagon_admit_op bits a host weight may offload, 0..OPS_ALL, 0 = offload nothing
    int64_t  gate_min;    // tokens any CPU-runnable op needs to be supported, 0 = gate off
};

// Returns -1 when the values are invalid (kept as they were), 0 when unchanged, 1 when changed. Must not
// run concurrently with a reserve, split or decode on ANY context of the same device - a single decode
// thread calls it between phases. The scheduler reads the device state during a decode.
typedef int (*ggml_backend_hexagon_set_admission_t)(ggml_backend_t backend, const struct ggml_hexagon_admission * admission);

// Returns how many times the device's admission has changed so far (its generation). Every context running
// on the device compares it against the generation it reserved its scheduler with and re-reserves when it
// moved, because any of them - not only the caller of set_admission - may have been the one to change it.
typedef uint64_t (*ggml_backend_hexagon_admission_gen_t)(ggml_backend_t backend);

// One-copy shared weights: when enabled, the device offers its TILE32 and HOST buffer types to the llama
// weight placement probe, so one weights copy serves the HTP and the CPU legs. The env vars
// GGML_HEXAGON_HOSTBUF, GGML_HEXAGON_HOSTBUF_REPACK and GGML_HEXAGON_MBUF seed the same state at init.
struct ggml_hexagon_shared_weights {
    bool   enabled;          // offer the device's TILE32/HOST buffer types to the weight placement probe
    size_t max_buffer_bytes; // get_max_size of every Hexagon buft. Enabling with 0 keeps the caps
                             // (first enable also snapshots the prior pair), with > 0 overwrites both.
                             // Disabling with 0 restores that snapshot if held; with > 0 writes both
                             // caps and drops the snapshot.
};

// Returns 0 when it took effect, nonzero when it changed nothing. Takes effect only while no TILE32
// or HOST buffer of the device is alive: those buffers are counted from their alloc to their free
// (the llama weight placement probe allocates 0-byte buffers that never reach the backend, so
// probes do not count), and after a refused or freed shared load the count is 0 again - the process
// can fall back to a two-copy load. Caller contract: before any model load or context creation - a
// call concurrent with a model load is a caller error the count does not catch, and a ggml_gallocr
// created before the call keeps the old buffer cap.
// Reached at run time through ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_set_shared_weights");
typedef int (*ggml_backend_hexagon_set_shared_weights_t)(const struct ggml_hexagon_shared_weights * shared_weights);

#ifdef  __cplusplus
}
#endif
