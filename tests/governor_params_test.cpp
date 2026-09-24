// Governor thermo parse tests (host-only: no model, no GPU).
//
// Pins the transport contract of cpp/rn-governor-params.cpp: parse-time
// validation is payload sanity only. The plugged baseline rule lives in the
// engine policy (vendor llama.cpp gate: t_idle_valid && isfinite && > 0 &&
// + 1 < 42); the binding must forward baselines untouched, or a second
// judge reappears at parse. The engine refuses at governor init with the
// same message: "governor: thermo profile invalid".

#include "rn-governor-params.h"
#include "rn-governor.h"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace rnllama;

// Test result tracking (same shape as simple_test.cpp)
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

static nlohmann::ordered_json base_governor() {
    return nlohmann::ordered_json{
        {"enabled", true},
        {"thermo", {
            {"sensor_valid", true},
            {"batt_temp_tenths_c", 350},
            {"plugged", true},
        }},
    };
}

static bool parses(const nlohmann::ordered_json & governor,
                    rnllama::governor_load_options * options = nullptr) {
    llama_governor_params params{};
    llama_governor_thermo_profile thermo{};
    rnllama::governor_load_options local_options{};
    try {
        const bool ok = rnllama::parse_governor_params(governor, params, thermo, local_options);
        if (options != nullptr) { *options = local_options; }
        return ok;
    } catch (const std::invalid_argument &) {
        return false;
    }
}

static bool refuses_with_engine_message(const nlohmann::ordered_json & governor) {
    llama_governor_params params{};
    llama_governor_thermo_profile thermo{};
    rnllama::governor_load_options options{};
    try {
        rnllama::parse_governor_params(governor, params, thermo, options);
    } catch (const std::invalid_argument & error) {
        return std::string(error.what()) == "governor: thermo profile invalid";
    }
    return false;
}

// Structural sanity stays a parse-time check: a dead battery sensor is
// refused here, before any engine sees the profile.
static bool test_dead_sensor_refused() {
    auto dead_sensor = base_governor();
    dead_sensor["thermo"]["sensor_valid"] = false;
    return refuses_with_engine_message(dead_sensor);
}

// The plugged baseline passes through even when it is out of range or
// missing; the engine's init refuses it (Case C), not the binding.
static bool test_out_of_range_baseline_forwarded() {
    auto hot_baseline = base_governor();
    hot_baseline["thermo"]["t_idle_valid"] = true;
    hot_baseline["thermo"]["t_idle_c"] = 41.5;
    return parses(hot_baseline);
}

static bool test_missing_baseline_flag_forwarded() {
    // plugged, t_idle_valid defaults to false; the engine judges that too.
    return parses(base_governor());
}

// Zero is the exact value the removed app gate refused (idle > 0); the
// binding forwards it for the engine, which now refuses it at init.
static bool test_zero_baseline_forwarded() {
    auto zero_baseline = base_governor();
    zero_baseline["thermo"]["t_idle_valid"] = true;
    zero_baseline["thermo"]["t_idle_c"] = 0.0;
    return parses(zero_baseline);
}

// Pins the rc-vs-engine-state discriminator both -2 filters key on: the
// flow-control -2s (thermal pause, chunking, reload-required) never set the
// engine's failed state, while decode_impl sets it before returning any
// engine rc, including -2 for GGML_STATUS_ALLOC_FAILED.
bool test_governor_decode_failed_discriminator() {
    return !rnllama::governor_decode_failed(0, false)
        && !rnllama::governor_decode_failed(-2, false)
        && rnllama::governor_decode_failed(-2, true)
        && rnllama::governor_decode_failed(-1, true)
        && rnllama::governor_decode_failed(1, false);
}

// governor.decode_repack is the P1 switch: default true (upstream repack
// on, the 12 GB+ shape), false = the 8 GB S23 lane (no_extra_bufts on the
// decode model). Both directions must parse; anything but a boolean refuses.
bool test_decode_repack_both_directions() {
    rnllama::governor_load_options options{};
    auto governor = base_governor();
    if (!parses(governor, &options) || !options.decode_repack) { return false; }
    governor["decode_repack"] = false;
    if (!parses(governor, &options) || options.decode_repack) { return false; }
    governor["decode_repack"] = true;
    if (!parses(governor, &options) || !options.decode_repack) { return false; }
    governor["decode_repack"] = 1;
    rnllama::governor_load_options ignored{};
    return !parses(governor, &ignored);
}

int main() {
    TestResults results;
    results.run_test("dead sensor refused at parse", test_dead_sensor_refused());
    results.run_test("out-of-range plugged baseline forwarded", test_out_of_range_baseline_forwarded());
    results.run_test("missing baseline flag forwarded", test_missing_baseline_flag_forwarded());
    results.run_test("zero plugged baseline forwarded for the engine", test_zero_baseline_forwarded());
    results.run_test("decode rc failure discriminates on engine state", test_governor_decode_failed_discriminator());
    results.run_test("governor decode_repack parses both directions", test_decode_repack_both_directions());
    results.print_summary();
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
