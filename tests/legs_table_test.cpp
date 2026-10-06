// One-copy capability table tests (host-only: the pure row match).
//
// Pins cpp/rn-legs-table.cpp rn_legs_for: the one validated S23 row (FABLE
// (c)), every negative that must keep today's loads (R2), and the J(1)
// consequence - a fact set whose GPU clause fails is two-copy, never an
// NPU-shared load without the OpenCL HOST leg.

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

bool all_three(const rn_leg_set & legs) {
    return legs.one_copy && legs.npu && legs.gpu && legs.cpu;
}

bool none(const rn_leg_set & legs) {
    return !legs.one_copy && !legs.npu && !legs.gpu && !legs.cpu;
}

bool test_row_matches() {
    return all_three(rn_legs_for(s23_facts()));
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
    facts.soc_model = "SM8650";  // Xiaomi 14: a row only with a lab report
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
    results.run_test("model token boundaries", test_model_token_boundaries());
    results.run_test("driver compiler major.minor line", test_driver_compiler_line());
    results.run_test("failed GPU clause demotes to two-copy (J1/R2)",
                     test_j1_gpu_clause_demotes_everything());
    results.run_test("unreadable fact is two-copy", test_unreadable_fact_is_two_copy());
    results.run_test("other phones keep today's loads", test_other_phones());
    results.print_summary();
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
