#pragma once

#include <limits>
#include <string>
#include <vector>

namespace rnllama {

// One usable thermal zone: the sysfs temp file to re-read at sample() time
// plus the leg trip it was admitted against (milli-C).
struct rn_thermal_zone {
    std::string temp_path;
    long passive_trip_mc = 0;
};

/** The decode-hop thermal-leg reader: finds this device's CPU, NPU and GPU
 *  thermal zones under a sysfs root (production "/sys/class/thermal", the
 *  tests pass a temp dir) and reports each leg's headroom to its first
 *  passive trip - the margin the engine's decode-hop rule hops on
 *  (llama_governor_set_decode_headroom). Qualcomm zone naming, verified on
 *  the S23 (SM8550): type prefix "cpu" ("cpu-0-0", "cpuss-0") is the CPU
 *  leg, "nsp" ("nspss-0") the NPU leg, "gpu" ("gpuss-0".."gpuss-7") the GPU
 *  decode leg. Zones are discovered by readdir and matched against
 *  thermal_zone<digits>, so the numbering gaps Linux leaves when it frees a
 *  zone id are walked past; a device that denies the directory (Jelly / MTK)
 *  leaves all legs empty and the hop on its alternation fallback. A zone
 *  joins a leg only when its temp and a positive passive trip read at
 *  construction (THERMAL_TEMP_INVALID is not a trip); the counts expose the
 *  discovered topology for the one-time device log. Never throws. */
class rn_thermal_legs {
public:
    struct headroom {
        float cpu_headroom_c = std::numeric_limits<float>::quiet_NaN();
        float npu_headroom_c = std::numeric_limits<float>::quiet_NaN();
        // The GPU decode leg; NaN on a device without gpu zones (the hop's
        // availability input, not an error).
        float gpu_headroom_c = std::numeric_limits<float>::quiet_NaN();
    };

    explicit rn_thermal_legs(const std::string & sysfs_root);

    // Headroom in C per leg, min over the leg's zones; NaN = no usable zone.
    // A zone whose temp stopped answering since construction makes its whole
    // leg unknown - the min over the survivors would hide exactly the
    // hottest block. Never throws.
    headroom sample() const;

    int cpu_zone_count() const { return static_cast<int>(cpu_zones_.size()); }
    int npu_zone_count() const { return static_cast<int>(npu_zones_.size()); }
    int gpu_zone_count() const { return static_cast<int>(gpu_zones_.size()); }

private:
    std::vector<rn_thermal_zone> cpu_zones_;
    std::vector<rn_thermal_zone> npu_zones_;
    std::vector<rn_thermal_zone> gpu_zones_;
};

} // namespace rnllama
