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

#include <cmath>
#include <iostream>
#include <limits>
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

// The engine's llama_governor_generation gained V81 (Snapdragon 8 Elite Gen 5, Hexagon v81,
// unqualified for GPU prefill in the engine); the binding parser must map the
// capability string through and keep refusing an arch it does not know.
static bool test_generation_parses_v81() {
    auto governor = base_governor();
    governor["generation"] = "V81";
    llama_governor_params params{};
    llama_governor_thermo_profile thermo{};
    governor_load_options options{};
    try {
        if (!parse_governor_params(governor, params, thermo, options)) {
            std::cerr << "parse_governor_params returned false" << std::endl;
            return false;
        }
    } catch (const std::exception & e) {
        std::cerr << "unexpected throw: " << e.what() << std::endl;
        return false;
    }
    if (params.generation != llama_governor_generation::V81) {
        std::cerr << "\"V81\" did not map to llama_governor_generation::V81" << std::endl;
        return false;
    }
    governor["generation"] = "V85";
    return !parses(governor);
}

// npu_lane_enabled used to throw "Governor NPU lane is not supported" at
// parse time; the engine now owns the lane (prefill_engine NPU branch), so
// the binding must forward the flag untouched.
static bool test_npu_lane_enabled_forwards() {
    auto governor = base_governor();
    governor["npu_lane_enabled"] = true;
    llama_governor_params params{};
    llama_governor_thermo_profile thermo{};
    governor_load_options options{};
    try {
        if (!parse_governor_params(governor, params, thermo, options)) {
            std::cerr << "parse_governor_params returned false" << std::endl;
            return false;
        }
    } catch (const std::exception & e) {
        std::cerr << "unexpected throw: " << e.what() << std::endl;
        return false;
    }
    return params.npu_lane_enabled;
}

// The runtime fallback records KALSA_HTP_RUNTIME_FALLBACK into the
// KALSA_HTP_FALLBACK env (cpp/rn-llama.cpp note_htp_runtime_fallback); the
// NEXT load of the process must then degrade exactly like an init-time
// failure: prefill off HTP, plan naming the reason the app logs on
// KALSA_GOVERNOR.
static bool test_htp_runtime_reason_degrades_next_load() {
    if (std::string(KALSA_HTP_RUNTIME_FALLBACK) != "htp-runtime-error") {
        std::cerr << "runtime fallback reason drifted: " << KALSA_HTP_RUNTIME_FALLBACK << std::endl;
        return false;
    }
    const auto plan = decide_governor_prefill_device(true, nullptr, KALSA_HTP_RUNTIME_FALLBACK);
    if (plan.use_device || std::string(plan.npu_device) != "GPU" ||
        plan.npu_fallback == nullptr || std::string(plan.npu_fallback) != "htp-runtime-error") {
        std::cerr << "runtime reason did not degrade a resolved HTP0 lane" << std::endl;
        return false;
    }
    return true;
}

// The lane-kill switch fires only on a compute failure of a batch the HTP
// device ran: the HTP-routed prefill, or an NPU decode hop (a 1-token batch
// on ctx_prefill, the HTP-pinned context — llama-governor.cpp decode_impl).
// Signal (engine sources): a batch's engine is its phase's stamp — the
// route latch for prefill (llama-governor-runtime.cpp
// stats_.prefill_engine), select_decode for decode (stats_.decode_engine);
// -3 is the compute-failure map on both paths (llama-context.cpp "case
// GGML_STATUS_FAILED: return -3"); and the reason prefix separates
// llama_decode's own failure from the host-side handoff failures.
// Everything else must keep the lane on.
static bool test_htp_runtime_failure_attribution() {
    const llama_governor_engine npu = llama_governor_engine::NPU;
    const llama_governor_engine cpu = llama_governor_engine::CPU;
    // The real thing: a prefill on the NPU route failing in compute.
    if (!htp_runtime_failure(5, npu, cpu, -3, "llama_decode failed with rc=-3")) {
        std::cerr << "HTP prefill compute failure was not attributed" << std::endl;
        return false;
    }
    // NPU decode hop (n_tokens == 1): the batch ran on ctx_prefill, the
    // HTP-pinned context, so the DSP did run it.
    if (!htp_runtime_failure(1, npu, npu, -3, "llama_decode failed with rc=-3")) {
        std::cerr << "NPU decode hop compute failure was not attributed" << std::endl;
        return false;
    }
    // CPU decode after an NPU prefill: the decode stamp decides, not the
    // stale prefill latch — the DSP did not run this batch.
    if (htp_runtime_failure(1, npu, cpu, -3, "llama_decode failed with rc=-3")) {
        std::cerr << "CPU decode failure was attributed to HTP" << std::endl;
        return false;
    }
    // CPU-latched prefill (LOWBAT/heat): the DSP did not run this batch.
    if (htp_runtime_failure(5, cpu, npu, -3, "llama_decode failed with rc=-3")) {
        std::cerr << "CPU-routed prefill failure was attributed to HTP" << std::endl;
        return false;
    }
    // Flow control (-2: admission wait, reload required, profile wait, plus
    // the alloc-failure map) shares one rc and never ran the DSP.
    if (htp_runtime_failure(5, npu, npu, -2, "llama_decode failed with rc=-2")) {
        std::cerr << "flow-control rc was attributed to HTP" << std::endl;
        return false;
    }
    // Host-side handoff failures never touched the DSP.
    if (htp_runtime_failure(5, npu, npu, -1, "KV commit failed: clSetUserEventStatus out of memory")) {
        std::cerr << "KV commit failure was attributed to HTP" << std::endl;
        return false;
    }
    if (htp_runtime_failure(5, npu, npu, -1, "route Reject during phase handoff")) {
        std::cerr << "route Reject was attributed to HTP" << std::endl;
        return false;
    }
    // No reason at all (rn_governor's null-reason path): keep the lane.
    if (htp_runtime_failure(5, npu, npu, -3, nullptr)) {
        std::cerr << "missing reason was attributed to HTP" << std::endl;
        return false;
    }
    return true;
}

// decide() is pure booleans/strings (the resolver's HTP0 name check is the
// engine's devices test): no ggml linkage, per this target's design.
static bool test_prefill_device_plan() {
    // Java-side HTP failure outranks a device that resolves.
    const auto degraded = decide_governor_prefill_device(true, nullptr, "htp-libs-missing");
    if (degraded.use_device || std::string(degraded.npu_device) != "GPU" ||
        degraded.npu_fallback == nullptr || std::string(degraded.npu_fallback) != "htp-libs-missing") {
        std::cerr << "java failure did not outrank the resolved device" << std::endl;
        return false;
    }
    // Java failure outranks even an engine-side degrade reason.
    const auto both = decide_governor_prefill_device(false, "htp-device-missing", "htp-env-missing");
    if (both.use_device || std::string(both.npu_fallback) != "htp-env-missing") {
        std::cerr << "java failure lost precedence over the engine reason" << std::endl;
        return false;
    }
    // No device, no java failure: degrade with the engine's own reason.
    const auto fallback = decide_governor_prefill_device(false, "htp-device-missing", nullptr);
    if (fallback.use_device || std::string(fallback.npu_device) != "GPU" ||
        fallback.npu_fallback == nullptr || std::string(fallback.npu_fallback) != "htp-device-missing") {
        std::cerr << "missing device did not degrade with htp-device-missing" << std::endl;
        return false;
    }
    // Resolved and quiet: the HTP0 lane with a clean plan.
    const auto lane = decide_governor_prefill_device(true, nullptr, nullptr);
    if (!lane.use_device || std::string(lane.npu_device) != "HTP0" || lane.npu_fallback != nullptr) {
        std::cerr << "resolved device did not open the HTP0 lane" << std::endl;
        return false;
    }
    // An empty reason string behaves like no reason.
    const auto empty = decide_governor_prefill_device(false, "htp-device-missing", "");
    if (empty.use_device || empty.npu_fallback == nullptr || std::string(empty.npu_fallback) != "htp-device-missing") {
        std::cerr << "empty reason string did not behave as absent" << std::endl;
        return false;
    }
    return true;
}

// governor_lane_policy_enabled is the flag rn-llama.cpp builds
// llama_governor_policy from: a lane asked for but not resolved must reach
// prefill_engine() off, or the engine claims NPU for a load whose devices
// are the default unqualified GPU (V81, Unknown). Only a resolved plan
// keeps it on, and both degrade reasons — the engine's own and the
// KALSA_HTP_FALLBACK env — clear it.
static bool test_lane_policy_flag_follows_resolution() {
    const auto resolved = decide_governor_prefill_device(true, nullptr, nullptr);
    if (!governor_lane_policy_enabled(true, resolved)) {
        std::cerr << "resolved lane did not stay on for the policy" << std::endl;
        return false;
    }
    const auto unresolved = decide_governor_prefill_device(false, "htp-device-missing", nullptr);
    if (governor_lane_policy_enabled(true, unresolved)) {
        std::cerr << "unresolved lane stayed on for the policy" << std::endl;
        return false;
    }
    const auto htp_env = decide_governor_prefill_device(true, nullptr, "htp-libs-missing");
    if (governor_lane_policy_enabled(true, htp_env)) {
        std::cerr << "KALSA_HTP_FALLBACK lane stayed on for the policy" << std::endl;
        return false;
    }
    // A lane nobody asked for stays off for the policy too.
    if (governor_lane_policy_enabled(false, resolved)) {
        std::cerr << "lane-off input was flipped on for the policy" << std::endl;
        return false;
    }
    return true;
}

// devices_excluding_registry backs the lane-off prefill pin (rn-llama.cpp):
// once the Hexagon backend registers, reproducing today's device list means
// dropping every HTP session and keeping the rest in registry order. The
// filter is pointer work by contract (rn-governor-params.h), so fabricated
// devices drive it without ggml linkage — same rule as decide() above.
#include "ggml-backend-impl.h"

namespace {

int fake_opencl_reg_ctx, fake_htp_reg_ctx, fake_cpu_reg_ctx;

ggml_backend_reg opencl_reg{0, {}, &fake_opencl_reg_ctx};
ggml_backend_reg htp_reg{0, {}, &fake_htp_reg_ctx};
ggml_backend_reg cpu_reg{0, {}, &fake_cpu_reg_ctx};

ggml_backend_device opencl_dev{{}, &opencl_reg, nullptr};
ggml_backend_device htp0_dev{{}, &htp_reg, nullptr};
ggml_backend_device htp1_dev{{}, &htp_reg, nullptr};
ggml_backend_device cpu_dev{{}, &cpu_reg, nullptr};

} // namespace

static bool test_devices_excluding_registry() {
    using rnllama::devices_excluding_registry;
    // Registry order as the ggml registry constructs it: OpenCL, then the
    // Hexagon sessions (GGML_HEXAGON_DEVICES=1 gives one, but keep two here
    // to pin that EVERY excluded-registry device goes), then CPU.
    const std::vector<ggml_backend_dev_t> mixed = {&opencl_dev, &htp0_dev, &htp1_dev, &cpu_dev};

    const auto without_htp = devices_excluding_registry(mixed, &htp_reg);
    if (without_htp.size() != 2 || without_htp[0] != &opencl_dev || without_htp[1] != &cpu_dev) {
        std::cerr << "HTP exclusion dropped or reordered the non-HTP devices" << std::endl;
        return false;
    }

    // An explicitly requested list reaches the loader WITH its null
    // terminator (JSI buildDeviceOverrides appends one) and may name HTP
    // (that path runs no HTP filter of its own — the lane-off loader
    // filters here). Terminators pass through skipped, HTP goes.
    const std::vector<ggml_backend_dev_t> explicit_list = {&htp0_dev, &opencl_dev, nullptr};
    const auto filtered = devices_excluding_registry(explicit_list, &htp_reg);
    if (filtered.size() != 1 || filtered[0] != &opencl_dev) {
        std::cerr << "explicit list with terminator did not filter to the non-HTP device" << std::endl;
        return false;
    }

    // HTP-only device set (no OpenCL): filters to the empty list, which the
    // loader turns into [nullptr] — the explicit CPU-only load. A bare
    // terminator-only list must survive the filter too, not crash on it.
    const auto htp_only = devices_excluding_registry({&htp0_dev, nullptr}, &htp_reg);
    const auto terminator_only = devices_excluding_registry({nullptr}, &htp_reg);
    if (!htp_only.empty() || !terminator_only.empty()) {
        std::cerr << "HTP-only or terminator-only input did not filter to empty" << std::endl;
        return false;
    }

    // Hexagon not compiled in: no registry to exclude, the list passes through.
    const auto passthrough = devices_excluding_registry(mixed, nullptr);
    if (passthrough.size() != mixed.size()) {
        std::cerr << "null exclusion did not pass the devices through" << std::endl;
        return false;
    }

    // An empty input stays empty.
    const auto empty = devices_excluding_registry({}, &htp_reg);
    if (!empty.empty()) {
        std::cerr << "empty input did not stay empty" << std::endl;
        return false;
    }
    return true;
}

// platform_thermal_status is optional and the engine owns its range (it
// clamps out-of-range to absent at ingress), so the binding forwards an
// integral number untouched and maps absent / wrong type / fractional to
// -1 (no platform vote) without throwing.
static bool test_platform_thermal_status_parses() {
    auto status_of = [](const nlohmann::ordered_json & thermo, int32_t * out) {
        try {
            *out = parse_governor_thermo(thermo).platform_thermal_status;
            return true;
        } catch (const std::exception & e) {
            std::cerr << "unexpected throw: " << e.what() << std::endl;
            return false;
        }
    };
    struct status_case {
        const char * name;
        nlohmann::ordered_json thermo;
        int32_t expected;
    };
    const status_case cases[] = {
        {"absent", {{"sensor_valid", true}}, -1},
        {"integer 3", {{"sensor_valid", true}, {"platform_thermal_status", 3}}, 3},
        {"integer 9 passes through", {{"sensor_valid", true}, {"platform_thermal_status", 9}}, 9},
        {"string 3 is absent", {{"sensor_valid", true}, {"platform_thermal_status", "3"}}, -1},
        {"null is absent", {{"sensor_valid", true}, {"platform_thermal_status", nullptr}}, -1},
        {"boolean is absent", {{"sensor_valid", true}, {"platform_thermal_status", true}}, -1},
        {"1e300 is absent", {{"sensor_valid", true}, {"platform_thermal_status", 1e300}}, -1},
        {"negative zero is zero", {{"sensor_valid", true}, {"platform_thermal_status", -0.0}}, 0},
        {"fractional is absent", {{"sensor_valid", true}, {"platform_thermal_status", 3.5}}, -1},
    };
    for (const auto & c : cases) {
        int32_t got = 0;
        if (!status_of(c.thermo, &got)) {
            std::cerr << "case threw: " << c.name << std::endl;
            return false;
        }
        if (got != c.expected) {
            std::cerr << c.name << ": got " << got << ", want " << c.expected << std::endl;
            return false;
        }
    }
    // End to end: binding-side validity must not reject an out-of-range
    // status either - 9 is forwarded for the engine to clamp.
    auto governor = base_governor();
    governor["thermo"]["platform_thermal_status"] = 9;
    llama_governor_params params{};
    llama_governor_thermo_profile thermo{};
    governor_load_options options{};
    try {
        if (!parse_governor_params(governor, params, thermo, options)) {
            std::cerr << "status 9 rejected by profile validity" << std::endl;
            return false;
        }
    } catch (const std::exception & e) {
        std::cerr << "unexpected throw: " << e.what() << std::endl;
        return false;
    }
    return thermo.platform_thermal_status == 9;
}

static bool test_decode_hop_tokens_parses() {
    llama_governor_params params{};
    llama_governor_thermo_profile thermo{};
    governor_load_options options{};
    auto governor = base_governor();
    if (!parse_governor_params(governor, params, thermo, options) || params.decode_hop_tokens != 0) {
        std::cerr << "absent decode_hop_tokens did not default to 0" << std::endl;
        return false;
    }
    governor["decode_hop_tokens"] = 64;
    if (!parse_governor_params(governor, params, thermo, options) || params.decode_hop_tokens != 64) {
        std::cerr << "decode_hop_tokens 64 did not parse" << std::endl;
        return false;
    }
    governor["decode_hop_tokens"] = -1;
    try {
        parse_governor_params(governor, params, thermo, options);
    } catch (const std::invalid_argument &) {
        return true;
    }
    std::cerr << "negative decode_hop_tokens did not refuse" << std::endl;
    return false;
}

// Rule v3 decode knobs (llama-ext.h decode_leg_weighting and friends):
// absent keys must land on the engine's fresh defaults, so a payload
// written before rule v3 keeps today's behaviour byte for byte.
static bool test_decode_rule_v3_defaults() {
    llama_governor_params params{};
    llama_governor_thermo_profile thermo{};
    governor_load_options options{};
    try {
        if (!parse_governor_params(base_governor(), params, thermo, options)) {
            std::cerr << "base payload returned false" << std::endl;
            return false;
        }
    } catch (const std::exception & e) {
        std::cerr << "unexpected throw: " << e.what() << std::endl;
        return false;
    }
    return params.decode_leg_weighting == llama_governor_leg_weighting::NPU_FIRST &&
        params.decode_heat_weight == 0.0f &&
        params.decode_heat_per_token_npu == 0.0f &&
        params.decode_heat_per_token_gpu == 0.0f &&
        params.decode_heat_per_token_cpu == 0.0f &&
        params.decode_load_step_npu_c == 0.0f &&
        params.decode_load_step_gpu_c == 0.0f &&
        params.decode_load_step_cpu_c == 0.0f &&
        params.decode_guard_headroom_c == 5.0f &&
        params.decode_headroom_tau_s == 0.0f;
}

// heat_rank plus a value on every new key parses through to the engine
// fields; the same payload then re-parses with gpu_burst and a partial
// legs object whose missing legs keep the engine default (0).
static bool test_decode_rule_v3_parses() {
    llama_governor_params params{};
    llama_governor_thermo_profile thermo{};
    governor_load_options options{};
    auto governor = base_governor();
    governor["decode_leg_weighting"] = "heat_rank";
    governor["decode_heat_weight"] = 0.25;
    governor["decode_heat_per_token"] = {{"npu", 0.15}, {"gpu", 0.35}, {"cpu", 0.30}};
    governor["decode_load_step"] = {{"npu", 1.0}, {"gpu", 2.0}, {"cpu", 1.5}};
    governor["decode_guard_headroom_c"] = 6.5;
    governor["decode_headroom_tau_s"] = 2.0;
    try {
        if (!parse_governor_params(governor, params, thermo, options)) {
            std::cerr << "rule v3 payload returned false" << std::endl;
            return false;
        }
    } catch (const std::exception & e) {
        std::cerr << "unexpected throw: " << e.what() << std::endl;
        return false;
    }
    if (params.decode_leg_weighting != llama_governor_leg_weighting::HEAT_RANK ||
        params.decode_heat_weight != 0.25f ||
        params.decode_heat_per_token_npu != 0.15f ||
        params.decode_heat_per_token_gpu != 0.35f ||
        params.decode_heat_per_token_cpu != 0.30f ||
        params.decode_load_step_npu_c != 1.0f ||
        params.decode_load_step_gpu_c != 2.0f ||
        params.decode_load_step_cpu_c != 1.5f ||
        params.decode_guard_headroom_c != 6.5f ||
        params.decode_headroom_tau_s != 2.0f) {
        std::cerr << "rule v3 values did not reach the engine fields" << std::endl;
        return false;
    }
    governor["decode_leg_weighting"] = "gpu_burst";
    governor["decode_heat_per_token"] = {{"npu", 0.15}, {"gpu", 0.35}, {"cpu", 0.30}};
    if (!parse_governor_params(governor, params, thermo, options) ||
        params.decode_leg_weighting != llama_governor_leg_weighting::GPU_BURST ||
        params.decode_heat_per_token_cpu != 0.30f) {
        std::cerr << "gpu_burst did not parse" << std::endl;
        return false;
    }
    // F3: a partial legs object refuses - it would half-apply a row's
    // measured triple. Full objects parse (above); partial ones do not.
    governor["decode_heat_per_token"] = {{"cpu", 0.30}};
    return !parses(governor);
}

// Out-of-range refuses instead of sanitizing: the heat weight and the
// per-token costs reach the policy unsanitized (only tau, the load steps
// and the guard get an engine-side net), and a silently zeroed weight
// would look like the v1 rule in the telemetry.
static bool test_decode_rule_v3_refuses_invalid() {
    auto governor = base_governor();
    governor["decode_leg_weighting"] = "cheapest";
    if (parses(governor)) {
        std::cerr << "unknown decode_leg_weighting parsed" << std::endl;
        return false;
    }
    governor["decode_leg_weighting"] = 3;
    if (parses(governor)) {
        std::cerr << "non-string decode_leg_weighting parsed" << std::endl;
        return false;
    }
    governor["decode_leg_weighting"] = "heat_rank";
    governor["decode_heat_weight"] = -0.1;
    if (parses(governor)) {
        std::cerr << "negative decode_heat_weight parsed" << std::endl;
        return false;
    }
    governor["decode_heat_weight"] = std::nan("");
    if (parses(governor)) {
        std::cerr << "NaN decode_heat_weight parsed" << std::endl;
        return false;
    }
    governor["decode_heat_weight"] = 0.25;
    governor["decode_heat_per_token"] = "0.15,0.35,0.30";
    if (parses(governor)) {
        std::cerr << "non-object decode_heat_per_token parsed" << std::endl;
        return false;
    }
    governor["decode_heat_per_token"] = {{"npu", 0.15}, {"gpu", -1.0}, {"cpu", 0.30}};
    if (parses(governor)) {
        std::cerr << "negative decode_heat_per_token leg parsed" << std::endl;
        return false;
    }
    governor["decode_heat_per_token"] = {{"npu", 0.15}, {"gpu", 0.35}, {"cpu", 0.30}};
    governor["decode_load_step"] = {{"cpu", std::nan("")}};
    if (parses(governor)) {
        std::cerr << "NaN decode_load_step leg parsed" << std::endl;
        return false;
    }
    governor["decode_load_step"] = {{"npu", 1.0}, {"gpu", 2.0}, {"cpu", 1.5}};
    governor["decode_guard_headroom_c"] = std::numeric_limits<double>::infinity();
    if (parses(governor)) {
        std::cerr << "infinite decode_guard_headroom_c parsed" << std::endl;
        return false;
    }
    governor["decode_guard_headroom_c"] = 6.5;
    governor["decode_headroom_tau_s"] = -1.0;
    if (parses(governor)) {
        std::cerr << "negative decode_headroom_tau_s parsed" << std::endl;
        return false;
    }
    // F3: the legs objects carry all three legs and only them.
    governor["decode_headroom_tau_s"] = 1.0;
    governor["decode_heat_per_token"] = {{"npu", 0.15}, {"gpu", 0.35}};
    if (parses(governor)) {
        std::cerr << "two-leg decode_heat_per_token parsed" << std::endl;
        return false;
    }
    governor["decode_heat_per_token"] = {{"npu", 0.15}, {"gpu", 0.35}, {"cpu", 0.30}, {"dsp", 1.0}};
    if (parses(governor)) {
        std::cerr << "unknown leg key parsed" << std::endl;
        return false;
    }
    // F6: a wrong-typed leg leaf names the parent key in the error.
    governor["decode_heat_per_token"] = {{"npu", "0.15"}, {"gpu", 0.35}, {"cpu", 0.30}};
    try {
        llama_governor_params sink{};
        llama_governor_thermo_profile thermo{};
        governor_load_options options{};
        parse_governor_params(governor, sink, thermo, options);
        std::cerr << "string decode_heat_per_token leaf parsed" << std::endl;
        return false;
    } catch (const std::invalid_argument & error) {
        if (std::string(error.what()).find("governor.decode_heat_per_token.npu") ==
            std::string::npos) {
            std::cerr << "leg leaf error does not name the parent key: " << error.what()
                      << std::endl;
            return false;
        }
    }
    return true;
}

// Precedence on the one-copy load (audit F1/P1): the matched row supplies
// the default for every rule v3 key the payload omits; an explicitly sent
// key wins (bench keys / kalsa.bench.* overrides keep working). The row
// values are the S23 row's (cpp/rn-legs-table.cpp, round 4); fabricated
// here because this target links only the parse TU.
static rn_leg_set s23_row() {
    rn_leg_set row;
    row.one_copy = row.npu = row.gpu = row.cpu = true;
    row.decode_headroom_tau_s = 10.0f;
    row.decode_heat_weight = 7.0f;
    row.decode_heat_per_token_npu = 2.8f;
    row.decode_heat_per_token_gpu = 3.2f;
    row.decode_heat_per_token_cpu = 4.7f;
    row.decode_hop_tokens = 32;
    row.decode_leg_weighting = llama_governor_leg_weighting::HEAT_RANK;
    row.decode_load_step_npu_c = 19.0f;
    row.decode_load_step_gpu_c = 9.0f;
    row.decode_load_step_cpu_c = 22.0f;
    row.decode_guard_headroom_c = 5.0f;
    return row;
}

static bool test_v3_row_defaults_merge() {
    llama_governor_params params{};
    llama_governor_thermo_profile thermo{};
    governor_load_options options{};
    try {
        if (!parse_governor_params(base_governor(), params, thermo, options)) {
            std::cerr << "base payload returned false" << std::endl;
            return false;
        }
    } catch (const std::exception & e) {
        std::cerr << "unexpected throw: " << e.what() << std::endl;
        return false;
    }
    if (options.v3_sent.hop_tokens || options.v3_sent.leg_weighting ||
        options.v3_sent.heat_weight || options.v3_sent.heat_per_token ||
        options.v3_sent.load_step || options.v3_sent.guard_headroom_c ||
        options.v3_sent.headroom_tau_s) {
        std::cerr << "absent keys were recorded as sent" << std::endl;
        return false;
    }
    const llama_governor_params merged =
        merge_leg_row_defaults(params, options.v3_sent, s23_row());
    if (merged.decode_hop_tokens != 32 ||
        merged.decode_leg_weighting != llama_governor_leg_weighting::HEAT_RANK ||
        merged.decode_heat_weight != 7.0f ||
        merged.decode_heat_per_token_npu != 2.8f || merged.decode_heat_per_token_gpu != 3.2f ||
        merged.decode_heat_per_token_cpu != 4.7f ||
        merged.decode_load_step_npu_c != 19.0f || merged.decode_load_step_gpu_c != 9.0f ||
        merged.decode_load_step_cpu_c != 22.0f ||
        merged.decode_guard_headroom_c != 5.0f || merged.decode_headroom_tau_s != 10.0f) {
        std::cerr << "row defaults did not fill the absent v3 keys" << std::endl;
        return false;
    }
    // Explicit keys win; the absent ones still take the row.
    auto governor = base_governor();
    governor["decode_hop_tokens"] = 64;
    governor["decode_guard_headroom_c"] = 8.0;
    governor["decode_heat_weight"] = 0.25;
    governor["decode_leg_weighting"] = "npu_first";
    try {
        if (!parse_governor_params(governor, params, thermo, options)) {
            std::cerr << "explicit payload returned false" << std::endl;
            return false;
        }
    } catch (const std::exception & e) {
        std::cerr << "unexpected throw: " << e.what() << std::endl;
        return false;
    }
    if (!options.v3_sent.hop_tokens || !options.v3_sent.guard_headroom_c ||
        !options.v3_sent.heat_weight || !options.v3_sent.leg_weighting) {
        std::cerr << "explicit keys were not recorded as sent" << std::endl;
        return false;
    }
    const llama_governor_params overridden =
        merge_leg_row_defaults(params, options.v3_sent, s23_row());
    return overridden.decode_hop_tokens == 64 &&
        overridden.decode_guard_headroom_c == 8.0f &&
        overridden.decode_heat_weight == 0.25f &&
        overridden.decode_leg_weighting == llama_governor_leg_weighting::NPU_FIRST &&
        overridden.decode_headroom_tau_s == 10.0f &&
        overridden.decode_load_step_npu_c == 19.0f;
}

int main() {
    TestResults results;
    results.run_test("dead sensor refused at parse", test_dead_sensor_refused());
    results.run_test("out-of-range plugged baseline forwarded", test_out_of_range_baseline_forwarded());
    results.run_test("missing baseline flag forwarded", test_missing_baseline_flag_forwarded());
    results.run_test("zero plugged baseline forwarded for the engine", test_zero_baseline_forwarded());
    results.run_test("decode rc failure discriminates on engine state", test_governor_decode_failed_discriminator());
    results.run_test("governor decode_repack parses both directions", test_decode_repack_both_directions());
    results.run_test("npu_lane_enabled forwards without throwing", test_npu_lane_enabled_forwards());
    results.run_test("generation V81 parses, unknown arch refuses", test_generation_parses_v81());
    results.run_test("prefill device plan degrades / opens HTP0", test_prefill_device_plan());
    results.run_test("lane policy flag follows resolution", test_lane_policy_flag_follows_resolution());
    results.run_test("runtime htp reason degrades the next load", test_htp_runtime_reason_degrades_next_load());
    results.run_test("only a batch the HTP device ran kills the lane", test_htp_runtime_failure_attribution());
    results.run_test("registry exclusion keeps non-HTP order", test_devices_excluding_registry());
    results.run_test("platform_thermal_status optional, engine owns range", test_platform_thermal_status_parses());
    results.run_test("decode_hop_tokens non-negative integer, 0 off", test_decode_hop_tokens_parses());
    results.run_test("decode rule v3 absent keys keep engine defaults", test_decode_rule_v3_defaults());
    results.run_test("decode rule v3 heat_rank values parse", test_decode_rule_v3_parses());
    results.run_test("decode rule v3 out-of-range keys refuse", test_decode_rule_v3_refuses_invalid());
    results.run_test("rule v3 row defaults merge, explicit keys win", test_v3_row_defaults_merge());
    results.print_summary();
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
