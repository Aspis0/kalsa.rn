// Pure platform thermal policy tests; no Android, model, or ggml runtime.

#include "rn-platform-thermal.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

using namespace rnllama;

namespace {

struct TestResults {
    int total = 0;
    int passed = 0;

    void run(const std::string & name, bool result) {
        ++total;
        std::cout << "TEST: " << name << " ... " << (result ? "PASSED" : "FAILED") << '\n';
        passed += result;
    }
};

constexpr int64_t t0 = 1000000;

bool test_interval_gate_advances_only_when_due() {
    int64_t last = t0;
    if (rn_platform_thermal_should_read(t0 + k_platform_thermal_interval_us - 1, last,
                                        k_platform_thermal_interval_us)) return false;
    if (last != t0) return false;
    if (!rn_platform_thermal_should_read(t0 + k_platform_thermal_interval_us, last,
                                         k_platform_thermal_interval_us)) return false;
    return last == t0 + k_platform_thermal_interval_us;
}

bool test_headroom_interval_is_its_own() {
    int64_t last = t0;
    return !rn_platform_thermal_should_read(t0 + k_platform_thermal_interval_us, last,
                                            k_platform_thermal_headroom_interval_us) &&
           rn_platform_thermal_should_read(t0 + k_platform_thermal_headroom_interval_us, last,
                                           k_platform_thermal_headroom_interval_us);
}

bool test_headroom_finite_read_is_forwarded_and_remembered() {
    rn_platform_thermal_headroom_state state;
    return rn_platform_thermal_headroom_forward(t0, 0.5f, state) == 0.5f &&
           state.value == 0.5f && state.read_us == t0;
}

bool test_headroom_nan_or_negative_read_carries_the_last_finite_value() {
    rn_platform_thermal_headroom_state state;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    rn_platform_thermal_headroom_forward(t0, 0.8f, state);
    if (rn_platform_thermal_headroom_forward(t0 + 1000000, nan, state) != 0.8f) return false;
    if (rn_platform_thermal_headroom_forward(t0 + 2000000, -1.0f, state) != 0.8f) return false;
    // A carried value does not refresh the window: it still expires 10 s after t0.
    return state.read_us == t0;
}

bool test_headroom_is_nan_after_the_carry_window() {
    rn_platform_thermal_headroom_state state;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    rn_platform_thermal_headroom_forward(t0, 0.8f, state);
    const float expired = rn_platform_thermal_headroom_forward(
        t0 + k_platform_thermal_carry_us + 1, nan, state);
    return std::isnan(expired) &&
           std::isnan(rn_platform_thermal_headroom_carried(t0 + k_platform_thermal_carry_us + 1, state));
}

bool test_headroom_never_read_is_nan() {
    const rn_platform_thermal_headroom_state state;
    return std::isnan(rn_platform_thermal_headroom_carried(t0, state));
}

bool test_escalation_requires_five_seconds_of_reads() {
    rn_platform_thermal_send_state state;
    auto decision = rn_platform_thermal_should_send(t0, 2, 1, state);
    if (decision.send || !decision.pending_started) return false;
    state = decision.next_state;
    for (int second = 1; second < 5; ++second) {
        decision = rn_platform_thermal_should_send(t0 + second * 1000000, 2, 1, state);
        if (decision.send || decision.pending_started) return false;
        state = decision.next_state;
    }
    decision = rn_platform_thermal_should_send(t0 + k_platform_thermal_escalation_debounce_us,
                                                2, 1, state);
    return decision.send;
}

bool test_escalation_keeps_consecutive_reads_at_or_above_pending_level() {
    rn_platform_thermal_send_state state;
    auto decision = rn_platform_thermal_should_send(t0, 2, 1, state);
    state = decision.next_state;
    for (int second = 1; second <= 5; ++second) {
        decision = rn_platform_thermal_should_send(t0 + second * 1000000, 3, 1, state);
        if (second < 5 && decision.send) return false;
        state = decision.next_state;
    }
    return decision.send;
}

bool test_escalation_restarts_after_a_read_gap() {
    auto decision = rn_platform_thermal_should_send(t0, 2, 1, {});
    auto state = decision.next_state;
    decision = rn_platform_thermal_should_send(t0 + 4 * 1000000, 2, 1, state);
    if (decision.send || !decision.pending_started ||
        decision.next_state.pending_since_us != t0 + 4 * 1000000) return false;
    state = decision.next_state;
    for (int second = 1; second < 5; ++second) {
        decision = rn_platform_thermal_should_send(t0 + (4 + second) * 1000000, 2, 1, state);
        if (decision.send) return false;
        state = decision.next_state;
    }
    decision = rn_platform_thermal_should_send(t0 + 9 * 1000000, 2, 1, state);
    return decision.send;
}

bool test_absent_read_keeps_pending_escalation() {
    rn_platform_thermal_send_state state;
    auto decision = rn_platform_thermal_should_send(t0, 3, 1, state);
    if (decision.send || !decision.pending_started) return false;
    state = decision.next_state;
    // Absent read within the cadence: the pending escalation survives it.
    decision = rn_platform_thermal_should_send(t0 + k_platform_thermal_interval_us,
                                                k_platform_thermal_absent, 1, state);
    if (decision.send || decision.pending_started ||
        decision.next_state.pending_since_us != t0 ||
        decision.next_state.last_seen_us != t0) return false;
    state = decision.next_state;
    for (int second = 2; second < 5; ++second) {
        decision = rn_platform_thermal_should_send(t0 + second * 1000000, 3, 1, state);
        if (decision.send) return false;
        state = decision.next_state;
    }
    decision = rn_platform_thermal_should_send(t0 + k_platform_thermal_escalation_debounce_us,
                                                3, 1, state);
    return decision.send;
}

bool test_deescalation_is_immediate() {
    const auto decision = rn_platform_thermal_should_send(t0, 2, 4, {});
    return decision.send && decision.next_state.pending_status == k_platform_thermal_absent;
}

bool test_absent_never_overwrites() {
    const auto decision = rn_platform_thermal_should_send(t0, k_platform_thermal_absent, 3, {});
    return !decision.send && decision.next_state.pending_status == k_platform_thermal_absent;
}

bool test_unchanged_status_does_not_resend() {
    const auto decision = rn_platform_thermal_should_send(t0, 3, 3, {});
    return !decision.send && !decision.pending_started;
}

bool test_ladder_mapping() {
    for (int32_t status = 0; status <= k_platform_thermal_shutdown; ++status) {
        if (rn_map_platform_thermal_status(status) != status) return false;
    }
    return rn_map_platform_thermal_status(-2) == k_platform_thermal_absent &&
           rn_map_platform_thermal_status(7) == k_platform_thermal_absent;
}

bool test_off_android_reader_is_absent() {
#if defined(__ANDROID__)
    return true;
#else
    return rn_platform_thermal().status() == k_platform_thermal_absent;
#endif
}

} // namespace

int main() {
    TestResults results;
    results.run("read interval advances only when due", test_interval_gate_advances_only_when_due());
    results.run("headroom has its own read interval", test_headroom_interval_is_its_own());
    results.run("finite headroom is forwarded and remembered", test_headroom_finite_read_is_forwarded_and_remembered());
    results.run("NaN or negative headroom carries the last finite value", test_headroom_nan_or_negative_read_carries_the_last_finite_value());
    results.run("headroom is NaN after the carry window", test_headroom_is_nan_after_the_carry_window());
    results.run("headroom never read is NaN", test_headroom_never_read_is_nan());
    results.run("escalation requires five seconds of reads", test_escalation_requires_five_seconds_of_reads());
    results.run("higher reads sustain the pending escalation", test_escalation_keeps_consecutive_reads_at_or_above_pending_level());
    results.run("escalation restarts after a read gap", test_escalation_restarts_after_a_read_gap());
    results.run("absent read keeps a pending escalation", test_absent_read_keeps_pending_escalation());
    results.run("de-escalation is immediate", test_deescalation_is_immediate());
    results.run("absent never overwrites", test_absent_never_overwrites());
    results.run("unchanged status does not resend", test_unchanged_status_does_not_resend());
    results.run("AThermal ladder maps to engine scale", test_ladder_mapping());
    results.run("off-Android reader is absent", test_off_android_reader_is_absent());
    std::cout << "Total tests: " << results.total << "\nPassed: " << results.passed
              << "\nFailed: " << (results.total - results.passed) << '\n';
    return results.passed == results.total ? 0 : 1;
}
