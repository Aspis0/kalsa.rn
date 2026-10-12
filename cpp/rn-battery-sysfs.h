#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
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

// The temp file holds one integer line. "301junk" is a malformed read, not
// 301: after the optional sign and digits only whitespace may follow, and
// the value must fit int32 (strtol saturates without failing on its own).
inline bool parse_battery_temp_line(const std::string & line, int32_t * out) {
    const std::size_t begin = line.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return false;
    }
    const std::size_t end = line.find_last_not_of(" \t\r\n");
    for (std::size_t i = begin; i <= end; ++i) {
        const char c = line[i];
        const bool digit = c >= '0' && c <= '9';
        const bool sign = (c == '-' || c == '+') && i == begin && end > begin;
        if (!digit && !sign) {
            return false;
        }
    }
    errno = 0;
    char * stop = nullptr;
    const long value = std::strtol(line.c_str() + begin, &stop, 10);
    if (errno == ERANGE || stop != line.c_str() + end + 1 ||
        value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    *out = static_cast<int32_t>(value);
    return true;
}

/** Reads <root>/battery/temp (integer, tenths of C on Qualcomm Android) and
 *  <root>/battery/status ("Charging" 1, "Discharging" 2, "Not charging" 3,
 *  "Full" 4, anything else 0). A sysfs node holds one value, so a temp with
 *  trailing junk on its line or a second non-empty line is a malformed read:
 *  temp_ok=false, temp_tenths_c=-1 (a missing status reads status_ok=false,
 *  status=0). Never throws. */
inline rn_battery_sysfs rn_battery_sysfs_read(const std::string & root) {
    rn_battery_sysfs probe;
    const auto read_temp = [&probe](const std::string & path) {
        std::ifstream in(path);
        std::string line;
        if (!in.is_open() || !std::getline(in, line)) {
            return;
        }
        // A sysfs node holds one value: any non-blank line past the first
        // ("301\n\njunk" included) means this is not the kernel node we take
        // it for, even when the first line parses.
        std::string extra;
        while (std::getline(in, extra)) {
            if (extra.find_first_not_of(" \t\r\n") != std::string::npos) {
                return;
            }
        }
        if (parse_battery_temp_line(line, &probe.temp_tenths_c)) {
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
