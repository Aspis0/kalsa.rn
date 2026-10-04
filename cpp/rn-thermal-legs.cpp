#include "rn-thermal-legs.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace rnllama {

namespace {

bool read_long(const std::string & path, long * out) {
    std::ifstream in(path);
    long value = 0;
    if (!in.is_open() || !(in >> value)) {
        return false;
    }
    *out = value;
    return true;
}

// Lowest trip_point_<i>_temp whose _type is "passive"; false when the zone
// has none. Trips are contiguous from 0, so the first missing temp ends the
// scan. Active and hot trips are ignored even when they sit lower: the
// passive line is the throttle point the decode hop must respect.
bool lowest_passive_trip_mc(const std::string & dir, long * out) {
    bool found = false;
    for (int i = 0;; ++i) {
        char name[48];
        std::snprintf(name, sizeof(name), "/trip_point_%d_temp", i);
        long temp_mc = 0;
        if (!read_long(dir + name, &temp_mc)) {
            break;
        }
        std::snprintf(name, sizeof(name), "/trip_point_%d_type", i);
        std::ifstream type_file(dir + name);
        std::string type;
        if (!type_file.is_open() || !std::getline(type_file, type) ||
            type.rfind("passive", 0) != 0) {
            continue;
        }
        if (!found || temp_mc < *out) {
            *out = temp_mc;
            found = true;
        }
    }
    return found;
}

} // namespace

rn_thermal_legs::rn_thermal_legs(const std::string & sysfs_root) {
    for (int index = 0;; ++index) {
        char name[32];
        std::snprintf(name, sizeof(name), "/thermal_zone%d", index);
        const std::string dir = sysfs_root + name;
        std::ifstream type_file(dir + "/type");
        std::string type;
        // Zones are numbered contiguously, so the first gap ends the scan;
        // a denied root (Jelly) breaks here at index 0 with both legs empty.
        if (!type_file.is_open() || !std::getline(type_file, type)) {
            break;
        }
        const bool is_cpu = type.rfind("cpu", 0) == 0;
        const bool is_npu = type.rfind("nsp", 0) == 0;
        if (!is_cpu && !is_npu) {
            continue;
        }
        long trip_mc = 0;
        long temp_mc = 0;
        if (!lowest_passive_trip_mc(dir, &trip_mc) || !read_long(dir + "/temp", &temp_mc)) {
            continue;
        }
        auto & leg = is_cpu ? cpu_zones_ : npu_zones_;
        leg.push_back({dir + "/temp", trip_mc});
    }
}

rn_thermal_legs::headroom rn_thermal_legs::sample() const {
    const auto leg_headroom = [](const std::vector<rn_thermal_zone> & zones) {
        float best = std::numeric_limits<float>::quiet_NaN();
        for (const auto & zone : zones) {
            long temp_mc = 0;
            if (!read_long(zone.temp_path, &temp_mc)) {
                continue;
            }
            const float headroom_c = static_cast<float>(zone.passive_trip_mc - temp_mc) / 1000.0f;
            best = std::isfinite(best) ? std::min(best, headroom_c) : headroom_c;
        }
        return best;
    };
    return {leg_headroom(cpu_zones_), leg_headroom(npu_zones_)};
}

} // namespace rnllama
