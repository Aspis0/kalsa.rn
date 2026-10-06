#include "rn-thermal-legs.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <dirent.h>
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

// thermal_zone<digits>, at least one digit - anything else in the class dir
// (power, devices, ...) is not a zone.
bool is_thermal_zone_dir(const std::string & name) {
    const std::string prefix = "thermal_zone";
    if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size()) {
        return false;
    }
    return std::all_of(name.begin() + static_cast<std::string::difference_type>(prefix.size()),
                       name.end(), [](char c) { return c >= '0' && c <= '9'; });
}

// Lowest trip_point_<i>_temp whose _type is "passive"; false when the zone
// has none. Trips are contiguous from 0, so the first missing temp ends the
// scan. Active and hot trips are ignored even when they sit lower: the
// passive line is the throttle point the decode hop must respect. A passive
// entry at or under 0 is Linux's THERMAL_TEMP_INVALID (-274000), not a trip.
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
            type.rfind("passive", 0) != 0 || temp_mc <= 0) {
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
    // readdir scan matched against thermal_zone<digits>: Linux frees zone ids
    // on unregister, so the numbering has gaps and cannot be walked by index.
    // A denied or missing root (Jelly/MTK) opens nothing and leaves both legs
    // empty.
    DIR * root = ::opendir(sysfs_root.c_str());
    if (root == nullptr) {
        return;
    }
    while (const dirent * entry = ::readdir(root)) {
        const std::string name = entry->d_name;
        if (!is_thermal_zone_dir(name)) {
            continue;
        }
        const std::string dir = sysfs_root + "/" + name;
        std::ifstream type_file(dir + "/type");
        std::string type;
        if (!type_file.is_open() || !std::getline(type_file, type)) {
            continue;
        }
        const bool is_cpu = type.rfind("cpu", 0) == 0;
        const bool is_npu = type.rfind("nsp", 0) == 0;
        const bool is_gpu = type.rfind("gpu", 0) == 0;
        if (!is_cpu && !is_npu && !is_gpu) {
            continue;
        }
        long trip_mc = 0;
        long temp_mc = 0;
        if (!lowest_passive_trip_mc(dir, &trip_mc) || !read_long(dir + "/temp", &temp_mc)) {
            continue;
        }
        auto & leg = is_cpu ? cpu_zones_ : (is_npu ? npu_zones_ : gpu_zones_);
        leg.push_back({dir + "/temp", trip_mc});
    }
    ::closedir(root);
}

rn_thermal_legs::headroom rn_thermal_legs::sample() const {
    const auto leg_headroom = [](const std::vector<rn_thermal_zone> & zones) {
        float best = std::numeric_limits<float>::quiet_NaN();
        for (const auto & zone : zones) {
            long temp_mc = 0;
            if (!read_long(zone.temp_path, &temp_mc)) {
                // One dark zone makes the whole leg unknown: the min over the
                // survivors would hide exactly the hottest block.
                return std::numeric_limits<float>::quiet_NaN();
            }
            const float headroom_c = static_cast<float>(zone.passive_trip_mc - temp_mc) / 1000.0f;
            best = std::isfinite(best) ? std::min(best, headroom_c) : headroom_c;
        }
        return best;
    };
    return {leg_headroom(cpu_zones_), leg_headroom(npu_zones_), leg_headroom(gpu_zones_)};
}

} // namespace rnllama
