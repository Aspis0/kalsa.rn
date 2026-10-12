// Governor observation ring tests (host-only: no model, no ggml).
//
// Pins the contract of cpp/rn-governor-samples.cpp: ring order, the overflow
// flag past 256, reset clearing a completion's record (trips survive), the
// first-step latch firing once, last_step advancing, and set_trips holding
// until replaced.

#include "rn-governor-samples.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace rnllama;

// Test result tracking (same shape as battery_sysfs_test.cpp)
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

rn_window_sample sample_with_tokens(uint32_t tokens) {
    rn_window_sample sample;
    sample.tokens = tokens;
    return sample;
}

// Samples come out oldest first, and count matches what was recorded.
bool test_ring_order() {
    rn_governor_samples record;
    for (uint32_t k = 0; k < 5; ++k) {
        record.record(sample_with_tokens(k * 32));
    }
    const std::vector<rn_window_sample> samples = record.samples();
    bool ok = record.count() == 5 && samples.size() == 5 && !record.overflow();
    for (uint32_t k = 0; ok && k < 5; ++k) {
        ok = samples[k].tokens == k * 32;
    }
    return ok;
}

// Past 256 the ring keeps the first entries, stops recording, and sets the
// overflow flag (route_chunks precedent).
bool test_overflow_flag() {
    rn_governor_samples record;
    const uint32_t extra = 3;
    for (uint32_t k = 0; k < rn_governor_samples::k_ring_capacity + extra; ++k) {
        record.record(sample_with_tokens(k));
    }
    const std::vector<rn_window_sample> samples = record.samples();
    bool ok = record.count() == rn_governor_samples::k_ring_capacity &&
        record.overflow() && samples.size() == rn_governor_samples::k_ring_capacity;
    for (uint32_t k = 0; ok && k < rn_governor_samples::k_ring_capacity; ++k) {
        ok = samples[k].tokens == k;
    }
    return ok;
}

// reset empties the ring and reopens the first-step latch, stamps included;
// the trips survive (per governor, not per completion).
bool test_reset_clears() {
    rn_governor_samples record;
    rn_leg_trips trips;
    trips.cpu_min_c = 70.0f;
    record.set_trips(trips);
    record.first_step(100, sample_with_tokens(0), {});
    record.record(sample_with_tokens(32));
    record.record(sample_with_tokens(64));
    record.reset();
    const bool cleared = record.count() == 0 && !record.overflow() &&
        record.samples().empty() && record.first_step_ms() == 0 &&
        record.last_step_ms() == 0 && !record.battery().temp_ok;
    // The latch is open again: a second completion's first step lands.
    record.first_step(900, sample_with_tokens(0), {});
    const bool ok = cleared && record.first_step_ms() == 900 &&
        record.trips().cpu_min_c == 70.0f;
    return ok;
}

// The first-step latch keeps only the first call's snapshot and battery.
bool test_first_step_latch_once() {
    rn_governor_samples record;
    rn_battery_sysfs first_probe;
    first_probe.temp_ok = true;
    first_probe.temp_tenths_c = 301;
    record.first_step(1000, sample_with_tokens(0), first_probe);
    rn_battery_sysfs ignored_probe;
    ignored_probe.status_ok = true;
    record.first_step(2000, sample_with_tokens(32), ignored_probe);
    const bool ok = record.first_step_ms() == 1000 && record.first().tokens == 0 &&
        record.battery().temp_ok && record.battery().temp_tenths_c == 301 &&
        !record.battery().status_ok;
    return ok;
}

// last_step advances on every call, the first included (first == last after
// one step).
bool test_last_step_advances() {
    rn_governor_samples record;
    record.first_step(1000, sample_with_tokens(0), {});
    record.last_step(1000);
    const bool first_step_stamp = record.last_step_ms() == 1000;
    record.last_step(1500);
    record.last_step(1750);
    const bool ok = first_step_stamp && record.last_step_ms() == 1750;
    return ok;
}

// set_trips holds what was set; an empty record reports NaN trips (the
// unknown leg coding).
bool test_trips_set_once() {
    rn_governor_samples record;
    const bool nan_before = !std::isfinite(record.trips().gpu_min_c);
    rn_leg_trips trips;
    trips.cpu_min_c = 95.0f;
    trips.cpu_max_c = 115.0f;
    trips.npu_min_c = 80.0f;
    trips.npu_max_c = 95.0f;
    trips.gpu_min_c = 105.0f;
    trips.gpu_max_c = 110.0f;
    record.set_trips(trips);
    bool ok = nan_before;
    const rn_leg_trips & read = record.trips();
    ok = ok && read.cpu_min_c == 95.0f && read.cpu_max_c == 115.0f &&
        read.npu_min_c == 80.0f && read.npu_max_c == 95.0f &&
        read.gpu_min_c == 105.0f && read.gpu_max_c == 110.0f;
    record.record(sample_with_tokens(0));
    ok = ok && read.gpu_min_c == 105.0f;
    return ok;
}

} // namespace

int main() {
    TestResults results;
    results.run_test("ring order: samples oldest first", test_ring_order());
    results.run_test("overflow flag past 256, first entries kept", test_overflow_flag());
    results.run_test("reset clears the record, trips survive", test_reset_clears());
    results.run_test("first-step latch fires once", test_first_step_latch_once());
    results.run_test("last_step advances, first included", test_last_step_advances());
    results.run_test("trips set once and held", test_trips_set_once());
    results.print_summary();
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
