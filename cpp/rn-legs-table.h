#pragma once

#include "llama-ext.h"

#include <string>

namespace rnllama {

// The device facts the legs table matches on (FABLE (c)). A fact the reader
// could not read is the empty string, false for dotprod: "unreadable" and
// "absent" are the same outcome under every match rule, and both keep the
// device on today's loads (R2).
struct rn_hw_facts {
    std::string soc_model;     // ro.soc.model, e.g. "SM8550"
    std::string hexagon_arch;  // registered HTP device description, "Hexagon v73"
    std::string gpu_name;      // CL_DEVICE_NAME of the GPUOpenCL backend's device
    std::string gpu_version;   // its CL_DEVICE_VERSION, e.g. "OpenCL 3.0 Adreno(TM) 740"
    std::string gpu_driver;    // its CL_DRIVER_VERSION, ends in "... E031.41.03.62"
    bool dotprod = false;      // AT_HWCAP ASIMDDP: the tile reader's sdot
};

// What the load should build. one_copy = false means today's loads: the
// two-model governor where its lane resolves, the plain load elsewhere.
// one_copy = true always implies all three legs - the NPU decode reads the
// tied Q6_K output through the OpenCL HOST leg (owner decision J(1)), so a
// device without a validated GPU leg must never load one-copy (R2), not even
// NPU+CPU.
struct rn_leg_set {
    bool one_copy = false;
    bool npu = false;
    bool gpu = false;
    bool cpu = false;
    // The decode-hop tuning of the matched row (owner decisions 2026-10-07
    // and 2026-10-09): rule v3 as measured on that device - S23 round 4
    // (step5-vsleg4, 2026-10-08) - so a matched device hops with HEAT_RANK
    // without the app sending anything. Per-device constants live in the
    // row, never in global defaults; all zero / NPU_FIRST when no row
    // matched = the raw rule, off. These are DEFAULTS, not overrides: the
    // one-copy load applies them through merge_leg_row_defaults
    // (rn-governor-params.h), where a key the app JSON explicitly sent wins.
    float decode_headroom_tau_s = 0.0f;
    float decode_heat_weight = 0.0f;
    float decode_heat_per_token_npu = 0.0f;
    float decode_heat_per_token_gpu = 0.0f;
    float decode_heat_per_token_cpu = 0.0f;
    uint32_t decode_hop_tokens = 0;
    llama_governor_leg_weighting decode_leg_weighting =
        llama_governor_leg_weighting::NPU_FIRST;
    float decode_load_step_npu_c = 0.0f;
    float decode_load_step_gpu_c = 0.0f;
    float decode_load_step_cpu_c = 0.0f;
    float decode_guard_headroom_c = 0.0f;
};

// The capability table: one pure, host-tested match against the validated
// rows. Exactly one row today (FABLE (c)); a row is added only with a lab
// report on that hardware - the table never infers from a family.
rn_leg_set rn_legs_for(const rn_hw_facts & facts);

// The device-side fact readers: system properties, the registered HTP device
// description, the OpenCL strings of the device the GPUOpenCL backend
// selected (empty, so never a row match, when that selection is pinned by
// GGML_OPENCL_PLATFORM / GGML_OPENCL_DEVICE) and AT_HWCAP. Android only - a
// host or non-Android build reads empty facts, which never match a row.
rn_hw_facts rn_read_hw_facts();

} // namespace rnllama
