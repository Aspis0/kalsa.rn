// Battery sysfs probe tests (host-only: no model, no ggml).
//
// Builds fake power_supply trees in a temp dir and pins the contract of
// cpp/rn-battery-sysfs.cpp: temp in tenths of C, the status text ladder,
// garbage temp / unknown status string / missing node failure values, and a
// read that never throws when the root is absent.

#include "rn-battery-sysfs.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace rnllama;

// Test result tracking (same shape as thermal_legs_test.cpp)
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
    std::filesystem::create_directories(root / "battery");
    return root;
}

void write_file(const std::filesystem::path & path, const std::string & content) {
    std::ofstream out(path);
    out << content << "\n";
}

// The Qualcomm-shaped node: integer tenths of C and the text status ladder.
bool test_both_files_present() {
    const auto root = make_temp_root("kalsa_battery_sysfs_present");
    write_file(root / "battery" / "temp", "301");       // 30.1 C
    write_file(root / "battery" / "status", "Discharging");
    const rn_battery_sysfs probe = rn_battery_sysfs_read(root.string());
    const bool ok = probe.temp_ok && probe.temp_tenths_c == 301 && probe.status_ok &&
        probe.status == 2;
    std::filesystem::remove_all(root);
    return ok;
}

// Every rung of the status ladder, and an unknown string reading 0 with
// status_ok still true (the file answered).
bool test_status_ladder() {
    const std::vector<std::pair<std::string, int8_t>> ladder = {
        {"Charging", 1}, {"Discharging", 2}, {"Not charging", 3}, {"Full", 4},
        {"Weird", 0},
    };
    for (const auto & rung : ladder) {
        const auto root = make_temp_root("kalsa_battery_sysfs_ladder");
        write_file(root / "battery" / "temp", "280");
        write_file(root / "battery" / "status", rung.first);
        const rn_battery_sysfs probe = rn_battery_sysfs_read(root.string());
        if (!probe.status_ok || probe.status != rung.second) {
            std::cerr << "status '" << rung.first << "' read " << static_cast<int>(probe.status)
                      << " ok=" << probe.status_ok << std::endl;
            std::filesystem::remove_all(root);
            return false;
        }
        std::filesystem::remove_all(root);
    }
    return true;
}

// A non-numeric temp reads temp_ok=false, temp_tenths_c=-1; a present status
// still reads.
bool test_garbage_temp() {
    const auto root = make_temp_root("kalsa_battery_sysfs_garbage");
    write_file(root / "battery" / "temp", "not-a-number");
    write_file(root / "battery" / "status", "Charging");
    const rn_battery_sysfs probe = rn_battery_sysfs_read(root.string());
    const bool ok = !probe.temp_ok && probe.temp_tenths_c == -1 && probe.status_ok &&
        probe.status == 1;
    std::filesystem::remove_all(root);
    return ok;
}

// Trailing junk after the integer is a malformed read ("301junk" must not
// answer 301), while whitespace around it is the kernel's own formatting. A
// sysfs node holds one value, so a second non-empty line rejects too; a bare
// newline after the value ("301\n", the kernel's own format) accepts.
bool test_temp_trailing_junk() {
    const std::vector<std::pair<std::string, bool>> cases = {
        {"301junk", false}, {"301 trailing", false}, {"-", false},
        {"301\nextra", false}, {"301\n", true}, {" 301 \t", true}, {"-5", true},
    };
    for (const auto & temp_case : cases) {
        const auto root = make_temp_root("kalsa_battery_sysfs_junk");
        write_file(root / "battery" / "temp", temp_case.first);
        const rn_battery_sysfs probe = rn_battery_sysfs_read(root.string());
        // -5 is a real (if cold) read; the expected value is the line's
        // integer whenever the line is well-formed.
        const int32_t expected = temp_case.second
            ? static_cast<int32_t>(std::strtol(temp_case.first.c_str(), nullptr, 10))
            : -1;
        if (probe.temp_ok != temp_case.second || probe.temp_tenths_c != expected) {
            std::cerr << "temp '" << temp_case.first << "' ok=" << probe.temp_ok
                      << " value=" << probe.temp_tenths_c << std::endl;
            std::filesystem::remove_all(root);
            return false;
        }
        std::filesystem::remove_all(root);
    }
    return true;
}

// A missing status file reads status_ok=false, status=0; a present temp
// still reads.
bool test_missing_status() {
    const auto root = make_temp_root("kalsa_battery_sysfs_no_status");
    write_file(root / "battery" / "temp", "280");
    const rn_battery_sysfs probe = rn_battery_sysfs_read(root.string());
    const bool ok = probe.temp_ok && probe.temp_tenths_c == 280 && !probe.status_ok &&
        probe.status == 0;
    std::filesystem::remove_all(root);
    return ok;
}

// A root that does not exist reads all-unknown and never throws.
bool test_root_absent() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "kalsa_battery_sysfs_absent";
    std::filesystem::remove_all(root);
    const rn_battery_sysfs probe = rn_battery_sysfs_read(root.string());
    const bool ok = !probe.temp_ok && probe.temp_tenths_c == -1 && !probe.status_ok &&
        probe.status == 0;
    return ok;
}

} // namespace

int main() {
    TestResults results;
    results.run_test("both files present: temp + status ladder read", test_both_files_present());
    results.run_test("status text ladder, unknown string reads 0", test_status_ladder());
    results.run_test("garbage temp reads temp_ok=false", test_garbage_temp());
    results.run_test("temp trailing junk rejected, whitespace fine", test_temp_trailing_junk());
    results.run_test("missing status reads status_ok=false", test_missing_status());
    results.run_test("absent root reads all-unknown, no throw", test_root_absent());
    results.print_summary();
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
