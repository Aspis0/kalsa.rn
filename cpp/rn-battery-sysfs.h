#pragma once

#include <cstdint>
#include <fstream>
#include <string>

namespace rnllama {

// The production sysfs root the probe reads (the tests pass a temp dir).
inline constexpr char k_battery_sysfs_root[] = "/sys/class/power_supply";

// One battery sysfs probe (Stage 0, OPEN 4): is the kernel battery node
// readable in-process, and what did temp and status answer? Pure data; the
// caller decides what to log. Header-only: the HTP skel fingerprint covers
// every cmake list a .cpp would have to be registered in, and this stage
// must not move it (scripts/htp-skels-common.sh HTP_SRC_PATHS).
struct rn_battery_sysfs {
    bool temp_ok = false;
    int32_t temp_tenths_c = -1;
    bool status_ok = false;
    int8_t status = 0;
};

/** Reads <root>/battery/temp (integer, tenths of C on Qualcomm Android) and
 *  <root>/battery/status ("Charging" 1, "Discharging" 2, "Not charging" 3,
 *  "Full" 4, anything else 0). A missing or non-numeric temp reads
 *  temp_ok=false, temp_tenths_c=-1; a missing status reads status_ok=false,
 *  status=0. Never throws. */
inline rn_battery_sysfs rn_battery_sysfs_read(const std::string & root) {
    rn_battery_sysfs probe;
    const auto read_temp = [&probe](const std::string & path) {
        std::ifstream in(path);
        long value = 0;
        if (in.is_open() && (in >> value)) {
            probe.temp_tenths_c = static_cast<int32_t>(value);
            probe.temp_ok = true;
        }
    };
    const auto read_status = [&probe](const std::string & path) {
        std::ifstream in(path);
        std::string text;
        if (in.is_open() && std::getline(in, text)) {
            probe.status = text == "Charging"       ? 1
                : text == "Discharging"             ? 2
                : text == "Not charging"            ? 3
                : text == "Full"                    ? 4
                                                    : 0;
            probe.status_ok = true;
        }
    };
    read_temp(root + "/battery/temp");
    if (!probe.temp_ok) {
        probe.temp_tenths_c = -1;
    }
    read_status(root + "/battery/status");
    return probe;
}

} // namespace rnllama
