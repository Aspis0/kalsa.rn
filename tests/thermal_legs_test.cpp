// Decode-hop thermal-leg reader tests (host-only: no model, no ggml).
//
// Builds fake sysfs trees in a temp dir and pins the contract of
// cpp/rn-thermal-legs.cpp: Qualcomm prefix routing (cpu / nsp, gpu ignored),
// passive-trip-only admission (a lower active trip never counts), zone
// membership only when temp + passive trip read, NaN for a leg with no
// usable zone, and a sample that never throws when a zone goes dark.

#include "rn-thermal-legs.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace rnllama;

// Test result tracking (same shape as governor_params_test.cpp)
struct TestResults {
    int total_tests = 0;
    int passed_tests = 0;

    void run_test(const std::string& name, bool result) {
        total_tests++;
        std::cout << "TEST: " << name << " ... ";
        if (result) {
            std::cout << "PASSED" << std::endl;
            passed_tests++;
        } else {
            std::cout << "FAILED" << std::endl;
        }
    }

    void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed: " << passed_tests << std::endl;
        std::cout << "Failed: " << (total_tests - passed_tests) << std::endl;
    }
};

namespace {

std::filesystem::path make_temp_root(const std::string & name) {
    auto root = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}

void write_file(const std::filesystem::path & path, const std::string & content) {
    std::ofstream out(path);
    out << content << "\n";
}

void make_zone(const std::filesystem::path & root, int index, const std::string & type,
               const std::vector<std::pair<std::string, std::string>> & files) {
    const auto dir = root / ("thermal_zone" + std::to_string(index));
    std::filesystem::create_directories(dir);
    write_file(dir / "type", type);
    for (const auto & file : files) {
        write_file(dir / file.first, file.second);
    }
}

// The S23-shaped tree: two usable CPU zones (cpu-1-0 with a lower active
// trip, cpuss-0), one usable NPU zone whose active trip sits BELOW its
// passive one (the trap), a CPU zone with only active trips and an NPU zone
// without a temp (both must be dropped), and a GPU zone (neither leg).
std::filesystem::path make_s23_tree() {
    const auto root = make_temp_root("kalsa_thermal_legs_s23");
    make_zone(root, 0, "cpu-1-0", {{"temp", "50000"},
                                   {"trip_point_0_type", "passive"}, {"trip_point_0_temp", "108000"},
                                   {"trip_point_1_type", "active"},  {"trip_point_1_temp", "85000"}});
    make_zone(root, 1, "cpuss-0", {{"temp", "60000"},
                                   {"trip_point_0_type", "passive"}, {"trip_point_0_temp", "115000"}});
    make_zone(root, 2, "nspss-0", {{"temp", "88000"},
                                   {"trip_point_0_type", "active"},  {"trip_point_0_temp", "90000"},
                                   {"trip_point_1_type", "passive"}, {"trip_point_1_temp", "95000"}});
    make_zone(root, 3, "cpu-0-2", {{"temp", "55000"},
                                   {"trip_point_0_type", "active"},  {"trip_point_0_temp", "85000"}});
    make_zone(root, 4, "gpuss-0", {{"temp", "70000"},
                                   {"trip_point_0_type", "passive"}, {"trip_point_0_temp", "110000"}});
    make_zone(root, 5, "nspss-1", {{"trip_point_0_type", "passive"}, {"trip_point_0_temp", "95000"}});
    return root;
}

bool close_to(float value, float expected) {
    return std::isfinite(value) && std::fabs(value - expected) < 0.001f;
}

bool is_unknown(float value) {
    return !std::isfinite(value);
}

bool test_s23_tree_topology_and_headroom() {
    const auto root = make_s23_tree();
    const rn_thermal_legs legs(root.string());
    if (legs.cpu_zone_count() != 2 || legs.npu_zone_count() != 1) {
        std::cerr << "zone counts cpu=" << legs.cpu_zone_count()
                  << " npu=" << legs.npu_zone_count() << std::endl;
        return false;
    }
    // CPU: min(108000-50000, 115000-60000) = 55.0. NPU: 95000-88000 = 7.0 -
    // the lower ACTIVE trip (90000) would answer 2.0 and fails this.
    const auto reading = legs.sample();
    if (!close_to(reading.cpu_headroom_c, 55.0f) || !close_to(reading.npu_headroom_c, 7.0f)) {
        std::cerr << "headroom cpu=" << reading.cpu_headroom_c
                  << " npu=" << reading.npu_headroom_c << std::endl;
        return false;
    }
    std::filesystem::remove_all(root);
    return true;
}

// Jelly-like device: the thermal class dir exists but holds no zones (MTK
// denies them) - both legs unknown, no throw.
bool test_jelly_like_empty_root() {
    const auto root = make_temp_root("kalsa_thermal_legs_jelly");
    const rn_thermal_legs legs(root.string());
    const bool ok = legs.cpu_zone_count() == 0 && legs.npu_zone_count() == 0 &&
        is_unknown(legs.sample().cpu_headroom_c) && is_unknown(legs.sample().npu_headroom_c);
    std::filesystem::remove_all(root);
    return ok;
}

// Zones whose temps stop answering after construction make their leg
// unknown at sample() time; the other leg keeps its reading.
bool test_zone_going_dark_at_sample() {
    const auto root = make_s23_tree();
    const rn_thermal_legs legs(root.string());
    std::error_code ec;
    std::filesystem::remove(root / "thermal_zone0" / "temp", ec);
    std::filesystem::remove(root / "thermal_zone1" / "temp", ec);
    if (ec) { return false; }
    const auto reading = legs.sample();
    const bool ok = is_unknown(reading.cpu_headroom_c) && close_to(reading.npu_headroom_c, 7.0f);
    std::filesystem::remove_all(root);
    return ok;
}

// One dark zone out of two in a leg darkens the WHOLE leg: the min over the
// survivors would report exactly the zone that went dark - usually the
// hottest - as absent, flattering the leg.
bool test_one_dark_zone_darkens_the_leg() {
    const auto root = make_s23_tree();
    const rn_thermal_legs legs(root.string());
    std::error_code ec;
    std::filesystem::remove(root / "thermal_zone0" / "temp", ec);
    if (ec) { return false; }
    const auto reading = legs.sample();
    const bool ok = is_unknown(reading.cpu_headroom_c) && close_to(reading.npu_headroom_c, 7.0f);
    std::filesystem::remove_all(root);
    return ok;
}

// Linux frees zone ids on unregister, so the class dir holds gaps: with only
// thermal_zone0 and thermal_zone2 on disk, the readdir scan still finds both
// (an index walk would stop at the missing zone1).
bool test_zone_numbering_gap() {
    const auto root = make_temp_root("kalsa_thermal_legs_gap");
    make_zone(root, 0, "cpu-1-0", {{"temp", "50000"},
                                   {"trip_point_0_type", "passive"}, {"trip_point_0_temp", "108000"}});
    make_zone(root, 2, "nspss-0", {{"temp", "88000"},
                                   {"trip_point_0_type", "passive"}, {"trip_point_0_temp", "95000"}});
    const rn_thermal_legs legs(root.string());
    const auto reading = legs.sample();
    const bool ok = legs.cpu_zone_count() == 1 && legs.npu_zone_count() == 1 &&
        close_to(reading.cpu_headroom_c, 58.0f) && close_to(reading.npu_headroom_c, 7.0f);
    std::filesystem::remove_all(root);
    return ok;
}

// A passive trip at Linux's THERMAL_TEMP_INVALID (-274000) is not a trip: a
// zone whose only passive line is invalid joins no leg instead of reporting
// a negative headroom.
bool test_invalid_passive_trip_ignored() {
    const auto root = make_temp_root("kalsa_thermal_legs_invalid_trip");
    make_zone(root, 0, "cpu-1-0", {{"temp", "50000"},
                                   {"trip_point_0_type", "passive"}, {"trip_point_0_temp", "-274000"}});
    make_zone(root, 1, "cpuss-0", {{"temp", "60000"},
                                   {"trip_point_0_type", "passive"}, {"trip_point_0_temp", "115000"}});
    const rn_thermal_legs legs(root.string());
    const auto reading = legs.sample();
    const bool ok = legs.cpu_zone_count() == 1 && close_to(reading.cpu_headroom_c, 55.0f);
    std::filesystem::remove_all(root);
    return ok;
}

} // namespace

int main() {
    TestResults results;
    results.run_test("S23 tree: counts and passive-only headroom", test_s23_tree_topology_and_headroom());
    results.run_test("empty root reads as unknown legs", test_jelly_like_empty_root());
    results.run_test("zone going dark at sample drops out", test_zone_going_dark_at_sample());
    results.run_test("one dark zone darkens its whole leg", test_one_dark_zone_darkens_the_leg());
    results.run_test("numbering gap: zone2 found past a missing zone1", test_zone_numbering_gap());
    results.run_test("invalid passive trip (-274000) ignored", test_invalid_passive_trip_ignored());
    results.print_summary();
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
