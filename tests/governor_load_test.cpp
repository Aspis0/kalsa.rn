// governor_load wire shape tests (host-only: no model, no ggml, no device).
//
// Pins cpp/rn-governor-params.cpp governor_load_json - the record the init
// result hands to JS. The app's uploaded report reads the JS console, so this
// mapping is the only way the load facts (which legs the engine built, the
// effective KV types, the matched row, the zone counts) reach it at all; a
// renamed or dropped key loses a fact with nothing failing. The facts THEMSELVES
// come from the loader and the governor, which need a device and a model:
// governor_params_test's neighbour that CAN reach them is a lab run.

#include "rn-governor-params.h"

#include <iostream>
#include <string>
#include <vector>

using namespace rnllama;

struct TestResults {
    int total_tests = 0;
    int passed_tests = 0;

    void run_test(const std::string& name, bool result) {
        total_tests++;
        std::cout << "TEST: " << name << " ... "
                  << (result ? "PASSED" : "FAILED") << std::endl;
        if (result) { passed_tests++; }
    }

    void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed: " << passed_tests << std::endl;
        std::cout << "Failed: " << (total_tests - passed_tests) << std::endl;
    }
};

namespace {

// The exact keys the record carries, in wire order. A key added here without
// being added there fails, so the shape cannot drift silently in either
// direction - src/__tests__/nativeSourceGuards.test.ts pins the same list
// against src/types.ts and the generated declarations.
const std::vector<std::string> k_wire_keys = {
    "one_copy", "npu_leg", "gpu_leg", "cpu_leg", "prefill_thermal_handoff",
    "soc", "cache_type_k", "cache_type_v", "kv_reason",
    "hop_tokens", "zones_cpu", "zones_npu", "zones_gpu",
};

// One S23 one-copy load: the matched row, all three legs built, the HTP lane
// having upgraded the caller's KV types, the row's hop cadence, and the zones
// its reader found.
governor_load_facts s23_one_copy() {
    governor_load_facts facts;
    facts.governor = true;
    facts.one_copy = true;
    facts.legs.npu = true;
    facts.legs.gpu = true;
    facts.prefill_thermal_handoff = true;
    facts.soc = "SM8550";
    facts.cache_type_k = "q8_0";
    facts.cache_type_v = "q8_0";
    facts.kv_reason = "htp-prefill-kv";
    facts.hop_tokens = 32;
    facts.zones_cpu = 5;
    facts.zones_npu = 3;
    facts.zones_gpu = 2;
    return facts;
}

std::vector<std::string> keys_of(const nlohmann::ordered_json & object) {
    std::vector<std::string> keys;
    for (auto it = object.begin(); it != object.end(); ++it) {
        keys.push_back(it.key());
    }
    return keys;
}

} // namespace

int main() {
    TestResults results;

    // The two-model fact resolution, before anything else: it is the only pure
    // part of the load path, and the HTP-pinned lane is the case it exists for.
    {
        const governor_load_legs lane_resolved =
            governor_two_model_legs(true, true);
        // A resolved lane puts ctx_prefill on the HTP alone
        // ({resolved.device, nullptr}), so even a GPU that was allowed and fits
        // is nowhere in that context. prefill_cpu_only is FALSE here - the CPU
        // fallback did not fire - so any resolver reading the GPU leg off it
        // claims a leg this load never placed.
        results.run_test("a resolved lane pins prefill to the NPU, not the GPU",
            lane_resolved.npu && !lane_resolved.gpu && lane_resolved.cpu);
        results.run_test("a resolved lane reports no GPU leg without the gate too",
            !governor_two_model_legs(true, false).gpu);
        // Lane off: the prefill context keeps the default device sweep, so the
        // GPU runs it exactly when placement was allowed. With the lane off the
        // gate is open by construction; the false case pins the resolver as
        // total anyway, for a load whose own gate says no.
        const governor_load_legs lane_off = governor_two_model_legs(false, true);
        results.run_test("no lane places prefill on the GPU when it is allowed",
            !lane_off.npu && lane_off.gpu && lane_off.cpu);
        results.run_test("no lane and a closed gate leaves the GPU leg absent",
            !governor_two_model_legs(false, false).gpu &&
                !governor_two_model_legs(false, false).npu);
    }

    // A load that is not a governor one published nothing, so the init result
    // carries no record at all - not an empty object, which would read as a
    // governor load that built no legs and matched no row.
    results.run_test("a non-governor load crosses as JSON null",
        governor_load_json(governor_load_facts{}).is_null());

    const nlohmann::ordered_json one_copy = governor_load_json(s23_one_copy());
    results.run_test("the record carries exactly the wire keys",
        keys_of(one_copy) == k_wire_keys);

    // Every value must survive the mapping verbatim: a record that reports the
    // facts is the whole point, so a silently defaulted field is a lost fact.
    results.run_test("the one-copy load's facts cross verbatim",
        one_copy["one_copy"] == true &&
        one_copy["npu_leg"] == true &&
        one_copy["gpu_leg"] == true &&
        one_copy["cpu_leg"] == true &&
        one_copy["prefill_thermal_handoff"] == true &&
        one_copy["soc"] == "SM8550" &&
        one_copy["cache_type_k"] == "q8_0" &&
        one_copy["cache_type_v"] == "q8_0" &&
        one_copy["kv_reason"] == "htp-prefill-kv" &&
        one_copy["hop_tokens"] == 32 &&
        one_copy["zones_cpu"] == 5 &&
        one_copy["zones_npu"] == 3 &&
        one_copy["zones_gpu"] == 2);

    // A leg the engine refused is false, and the CPU leg is true whatever the
    // load resolved - the engine refuses a governor without it.
    governor_load_facts demoted = s23_one_copy();
    demoted.legs.gpu = false;
    const nlohmann::ordered_json demoted_json = governor_load_json(demoted);
    results.run_test("a refused leg reads false and the CPU leg true",
        demoted_json["gpu_leg"] == false && demoted_json["cpu_leg"] == true);

    // A row that did not match, and a load that left the KV types and their
    // placement exactly as sent: absent as null, never "" - otherwise an
    // unmatched row reads as a row whose SoC model is blank, and "nothing
    // overrode KV" as "overridden by".
    governor_load_facts plain = s23_one_copy();
    plain.one_copy = false;
    plain.soc.clear();
    plain.kv_reason.clear();
    const nlohmann::ordered_json plain_json = governor_load_json(plain);
    results.run_test("an absent row or KV reason crosses as null",
        plain_json["one_copy"] == false &&
        plain_json["soc"].is_null() &&
        plain_json["kv_reason"].is_null());

    // A load that matched a row but ran the two-model form still names the row:
    // only one_copy=false distinguishes that from one that matched nothing.
    governor_load_facts lane_off = s23_one_copy();
    lane_off.one_copy = false;
    lane_off.legs.npu = false;
    lane_off.hop_tokens = 0;
    lane_off.zones_cpu = 0;
    lane_off.zones_npu = 0;
    lane_off.zones_gpu = 0;
    const nlohmann::ordered_json lane_off_json = governor_load_json(lane_off);
    results.run_test("a matched row with the two-model form keeps the soc",
        lane_off_json["soc"] == "SM8550" &&
        lane_off_json["one_copy"] == false &&
        lane_off_json["npu_leg"] == false &&
        lane_off_json["gpu_leg"] == true &&
        lane_off_json["hop_tokens"] == 0);

    // The record stays numbers, booleans and short enums: a device string or a
    // driver line here would put hardware detail in the uploaded report.
    bool scalars_only = true;
    for (const auto & item : one_copy.items()) {
        scalars_only = scalars_only &&
            (item.value().is_boolean() || item.value().is_number() ||
             item.value().is_string() || item.value().is_null());
    }
    results.run_test("every value is a scalar", scalars_only);

    results.print_summary();
    return results.passed_tests == results.total_tests ? 0 : 1;
}