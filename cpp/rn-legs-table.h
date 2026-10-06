#pragma once

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
