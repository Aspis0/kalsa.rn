// One-copy capability table tests (host-only: the pure row match).
//
// Pins cpp/rn-legs-table.cpp rn_legs_for: validated SoC rows, every negative
// that must keep today's loads (R2), and the J(1) consequence - a fact set
// whose GPU clause fails is two-copy, never an NPU-shared load without the
// OpenCL HOST leg.

#include "rn-legs-table.h"

#include <iostream>
#include <string>

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

// The S23's facts as the readers see them (lab oracle_consumer_s23): the
// 740 lives in CL_DEVICE_VERSION, the compiler E031.41.03.62 in
// CL_DRIVER_VERSION.
rn_hw_facts s23_facts() {
    rn_hw_facts facts;
    facts.soc_model = "SM8550";
    facts.hexagon_arch = "Hexagon v73";
    facts.gpu_name = "QUALCOMM Adreno(TM)";
    facts.gpu_version = "OpenCL 3.0 Adreno(TM) 740";
    facts.gpu_driver = "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.41.03.62";
    facts.dotprod = true;
    return facts;
}

rn_hw_facts sm8650_facts() {
    rn_hw_facts facts;
    facts.soc_model = "SM8650";
    facts.hexagon_arch = "Hexagon v75";
    facts.gpu_name = "QUALCOMM Adreno(TM) 750";
    facts.gpu_version = "OpenCL 3.0 Adreno(TM) 750";
    facts.gpu_driver = "OpenCL 3.0 QUALCOMM Compiler E031.45.02.25";
    facts.dotprod = true;
    return facts;
}

bool all_three(const rn_leg_set & legs) {
    return legs.one_copy && legs.npu && legs.gpu && legs.cpu;
}

bool none(const rn_leg_set & legs) {
    return !legs.one_copy && !legs.npu && !legs.gpu && !legs.cpu;
}

bool tuning_is(const rn_leg_set & legs, float tau_s, float heat_weight, float npu, float gpu,
               float cpu) {
    return legs.decode_headroom_tau_s == tau_s && legs.decode_heat_weight == heat_weight &&
           legs.decode_heat_per_token_npu == npu && legs.decode_heat_per_token_gpu == gpu &&
           legs.decode_heat_per_token_cpu == cpu;
}

bool rule_v3_is(const rn_leg_set & legs, uint32_t hop_tokens,
                llama_governor_leg_weighting weighting, float step_npu, float step_gpu,
                float step_cpu, float guard_c) {
    return legs.decode_hop_tokens == hop_tokens && legs.decode_leg_weighting == weighting &&
           legs.decode_load_step_npu_c == step_npu && legs.decode_load_step_gpu_c == step_gpu &&
           legs.decode_load_step_cpu_c == step_cpu && legs.decode_guard_headroom_c == guard_c;
}

bool test_row_matches() {
    return all_three(rn_legs_for(s23_facts()));
}

// The matched row ships its measured rule v3 tuning (owner decisions
// 2026-10-07 and 2026-10-09): tau 10 s smoothing, heat weight 7, skin C
// heat per leg, and the round-4 hop cadence - hop 32, HEAT_RANK, load step
// 19/9/22 C, guard 5 C (step5-vsleg4, 2026-10-08).
bool test_row_returns_tuning() {
    return all_three(rn_legs_for(s23_facts())) &&
           tuning_is(rn_legs_for(s23_facts()), 10.0f, 7.0f, 2.8f, 3.2f, 4.7f) &&
           rule_v3_is(rn_legs_for(s23_facts()), 32,
                      llama_governor_leg_weighting::HEAT_RANK, 19.0f, 9.0f, 22.0f, 5.0f);
}

bool test_sm8650_row_and_driver_gate() {
    const rn_leg_set sm8650 = rn_legs_for(sm8650_facts());
    if (!all_three(sm8650) ||
        !tuning_is(sm8650, 10.0f, 7.0f, 1.1f, 3.2f, 7.3f) ||
        !rule_v3_is(sm8650, 32, llama_governor_leg_weighting::HEAT_RANK,
                    5.0f, 11.0f, 26.0f, 5.0f)) {
        return false;
    }
    auto wrong_driver = sm8650_facts();
    wrong_driver.gpu_driver = "OpenCL 3.0 QUALCOMM Compiler E031.41.02.25";
    if (!none(rn_legs_for(wrong_driver))) { return false; }
    const rn_leg_set s23 = rn_legs_for(s23_facts());
    return all_three(s23) && tuning_is(s23, 10.0f, 7.0f, 2.8f, 3.2f, 4.7f) &&
        rule_v3_is(s23, 32, llama_governor_leg_weighting::HEAT_RANK,
                   19.0f, 9.0f, 22.0f, 5.0f);
}

// Zero = off when no row matched: today's raw headroom rule on every
// non-validated device.
bool test_no_match_tuning_is_zero() {
    if (!tuning_is(rn_legs_for(rn_hw_facts{}), 0, 0, 0, 0, 0) ||
        !rule_v3_is(rn_legs_for(rn_hw_facts{}), 0,
                    llama_governor_leg_weighting::NPU_FIRST, 0, 0, 0, 0)) {
        return false;
    }
    auto facts = s23_facts();
    facts.gpu_driver = "OpenCL 3.0 QUALCOMM build: 9999.99 Compiler E031.50.00.00";
    return tuning_is(rn_legs_for(facts), 0, 0, 0, 0, 0) &&
           rule_v3_is(rn_legs_for(facts), 0, llama_governor_leg_weighting::NPU_FIRST, 0, 0, 0, 0);
}

bool test_model_token_boundaries() {
    auto facts = s23_facts();
    facts.gpu_version = "OpenCL 3.0 Adreno(TM) 7401";
    if (all_three(rn_legs_for(facts))) { return false; }  // another part number
    facts.gpu_version = "OpenCL 3.0 Adreno(TM) 1740";
    if (all_three(rn_legs_for(facts))) { return false; }
    // the model may live in the name instead of the version
    facts.gpu_name = "QUALCOMM Adreno(TM) 740";
    facts.gpu_version = "OpenCL 3.0";
    return all_three(rn_legs_for(facts));
}

bool test_driver_compiler_line() {
    auto facts = s23_facts();
    facts.gpu_driver = "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.45.03.01";
    if (all_three(rn_legs_for(facts))) { return false; }  // other minor line
    facts.gpu_driver = "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.411.03";
    if (all_three(rn_legs_for(facts))) { return false; }  // not major.minor 41
    facts.gpu_driver = "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.41.99.99";
    return all_three(rn_legs_for(facts));                 // patch line is fine
}

// The row requires the Adreno identity, not just the model number (audit F6):
// a non-Adreno GPU that happens to carry a whole "740" token must not match,
// and Adreno alone without the model token does not either.
bool test_non_adreno_740_refused() {
    auto facts = s23_facts();
    facts.gpu_name = "PowerVR-BX-740";
    facts.gpu_version = "OpenCL 3.0 PowerVR B-Series 740";
    if (all_three(rn_legs_for(facts))) { return false; }
    facts = s23_facts();
    facts.gpu_version = "OpenCL 3.0 Adreno(TM) 750";
    return !all_three(rn_legs_for(facts));
}

bool test_j1_gpu_clause_demotes_everything() {
    // The J(1)/R2 consequence: with the GPU clause failed (driver update),
    // one_copy is false - never NPU+CPU one-copy without the OpenCL HOST leg.
    auto facts = s23_facts();
    facts.gpu_driver = "OpenCL 3.0 QUALCOMM build: 9999.99 Compiler E031.50.00.00";
    return none(rn_legs_for(facts));
}

bool test_unreadable_fact_is_two_copy() {
    const rn_hw_facts unreadable{};
    if (!none(rn_legs_for(unreadable))) { return false; }
    auto facts = s23_facts();
    facts.soc_model.clear();
    if (!none(rn_legs_for(facts))) { return false; }
    facts = s23_facts();
    facts.hexagon_arch.clear();  // skel missing, HTP device never registered
    if (!none(rn_legs_for(facts))) { return false; }
    facts = s23_facts();
    facts.dotprod = false;
    return none(rn_legs_for(facts));  // no dotprod = no one-copy at all
}

bool test_other_phones() {
    auto facts = s23_facts();
    facts.soc_model = "SM8650";
    if (!none(rn_legs_for(facts))) { return false; }
    facts = s23_facts();
    facts.hexagon_arch = "Hexagon v79";
    if (!none(rn_legs_for(facts))) { return false; }
    facts = s23_facts();
    facts.gpu_version = "OpenCL 3.0 Adreno(TM) 830";
    return none(rn_legs_for(facts));
}

} // namespace

int main() {
    TestResults results;
    results.run_test("S23 row matches", test_row_matches());
    results.run_test("S23 row returns the hop tuning", test_row_returns_tuning());
    results.run_test("SM8650 row, compiler gate, and S23 row", test_sm8650_row_and_driver_gate());
    results.run_test("no row matched returns zero tuning", test_no_match_tuning_is_zero());
    results.run_test("model token boundaries", test_model_token_boundaries());
    results.run_test("driver compiler major.minor line", test_driver_compiler_line());
    results.run_test("non-Adreno 740 refused (Adreno identity required)",
                     test_non_adreno_740_refused());
    results.run_test("failed GPU clause demotes to two-copy (J1/R2)",
                     test_j1_gpu_clause_demotes_everything());
    results.run_test("unreadable fact is two-copy", test_unreadable_fact_is_two_copy());
    results.run_test("other phones keep today's loads", test_other_phones());
    results.print_summary();
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
