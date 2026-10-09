// Mid-turn platform thermal status tests (host-only: no model, no ggml, no
// Android).
//
// Pins cpp/rn-platform-thermal.cpp: the AThermalStatus ladder mapped 1:1 onto
// the engine's platform_thermal_status, and the three decisions the governor
// leans on when forwarding a native read - at most one read per second of
// wall time, only on a change, and never let "absent" overwrite a status the
// app already handed over.

#include "rn-platform-thermal.h"

#include <cstdint>
#include <iostream>
#include <string>

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

constexpr int64_t t0 = 1000000;  // some steady-clock value, well past the 0 start

// What the app hands over once per completion: PowerManager's MODERATE.
constexpr int32_t k_app_status = 2;

// AThermalStatus is the engine's ladder: ATHERMAL_STATUS_NONE(0)..
// ATHERMAL_STATUS_SHUTDOWN(6) pass through untouched, SEVERE (3) included -
// the status the S23 reported mid-turn and the governor missed.
bool test_ladder_maps_one_to_one() {
    for (int32_t status = 0; status <= k_platform_thermal_shutdown; ++status) {
        if (rn_map_platform_thermal_status(status) != status) {
            return false;
        }
    }
    return true;
}

// Everything the platform never sent is "absent", including the NDK's own
// ATHERMAL_STATUS_ERROR and whatever a future platform adds above SHUTDOWN.
bool test_out_of_scale_is_absent() {
    if (rn_map_platform_thermal_status(-1) != k_platform_thermal_absent) { return false; }
    if (rn_map_platform_thermal_status(-100) != k_platform_thermal_absent) { return false; }
    if (rn_map_platform_thermal_status(k_platform_thermal_shutdown + 1) != k_platform_thermal_absent) {
        return false;
    }
    return rn_map_platform_thermal_status(100) == k_platform_thermal_absent;
}

// Off Android (iOS, host) the same TU answers "absent", which is what keeps
// every governor path on the app's own status there.
bool test_off_android_reads_absent() {
#if defined(__ANDROID__)
    // tests/CMakeLists.txt never targets Android; the real reader is proved
    // in-app, so this case only carries an off-Android meaning.
    return true;
#else
    const rn_platform_thermal thermal;
    return thermal.status() == k_platform_thermal_absent;
#endif
}

bool test_second_read_inside_the_second_is_dropped() {
    int64_t last_read = 0;
    if (!rn_platform_thermal_should_send(t0, last_read, 3, -1)) { return false; }
    if (last_read != t0) { return false; }
    // Same tick: the engine must not hear one move twice.
    if (rn_platform_thermal_should_send(t0, last_read, 3, 3)) { return false; }
    if (last_read != t0) { return false; }
    // And still nothing a microsecond short of the second.
    return !rn_platform_thermal_should_send(t0 + k_platform_thermal_interval_us - 1, last_read, 3, 3);
}

bool test_read_due_after_a_second_and_only_on_change() {
    int64_t last_read = 0;
    const int64_t due = t0 + k_platform_thermal_interval_us;
    if (!rn_platform_thermal_should_send(t0, last_read, 2, -1)) { return false; }
    if (last_read != t0) { return false; }
    // Unchanged: no send, but the read still cost its second.
    if (rn_platform_thermal_should_send(due, last_read, 2, 2)) { return false; }
    if (last_read != due) { return false; }
    if (!rn_platform_thermal_should_send(due + k_platform_thermal_interval_us, last_read, 3, 2)) {
        return false;
    }
    return last_read == due + k_platform_thermal_interval_us;
}

bool test_absent_never_overwrites_a_valid_status() {
    int64_t last_read = 0;
    if (!rn_platform_thermal_should_send(t0, last_read, k_app_status, -1)) { return false; }
    // An unasking platform one hour later still leaves the app's status.
    const int64_t hour = t0 + 3600 * k_platform_thermal_interval_us;
    if (rn_platform_thermal_should_send(hour, last_read, k_platform_thermal_absent,
                                        k_app_status)) {
        return false;
    }
    if (last_read != hour) { return false; }  // the read was taken anyway
    // The engine hears the recovery the moment the platform answers again.
    return rn_platform_thermal_should_send(hour + k_platform_thermal_interval_us, last_read, 0,
                                           k_app_status);
}

bool test_absent_on_absent_is_not_a_change() {
    int64_t last_read = 0;
    if (rn_platform_thermal_should_send(t0, last_read, k_platform_thermal_absent,
                                        k_platform_thermal_absent)) {
        return false;
    }
    return !rn_platform_thermal_should_send(t0 + k_platform_thermal_interval_us, last_read,
                                            k_platform_thermal_absent,
                                            k_platform_thermal_absent);
}

} // namespace

int main() {
    TestResults results;
    results.run_test("AThermal ladder maps 1:1 onto the engine scale", test_ladder_maps_one_to_one());
    results.run_test("out-of-scale and ATHERMAL_STATUS_ERROR are absent",
                     test_out_of_scale_is_absent());
    results.run_test("off-Android build reads absent", test_off_android_reads_absent());
    results.run_test("second read inside the second is dropped",
                     test_second_read_inside_the_second_is_dropped());
    results.run_test("read due after a second, and only on change",
                     test_read_due_after_a_second_and_only_on_change());
    results.run_test("absent never overwrites a valid status",
                     test_absent_never_overwrites_a_valid_status());
    results.run_test("absent on absent is not a change", test_absent_on_absent_is_not_a_change());
    results.print_summary();
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
