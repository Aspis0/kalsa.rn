// Decode rule v3 parsing and per-SoC row precedence.

#include "rn-governor-params.h"
#include "rn-legs-table.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace rnllama;

namespace {

struct TestResults {
    int total = 0;
    int passed = 0;

    void run(const char * name, bool ok) {
        ++total;
        passed += ok;
        std::cout << "TEST: " << name << " ... " << (ok ? "PASSED" : "FAILED") << '\n';
    }
};

nlohmann::ordered_json base_governor() {
    return {{"enabled", true}, {"thermo", {
        {"sensor_valid", true}, {"batt_temp_tenths_c", 350}, {"plugged", true},
    }}};
}

rn_hw_facts s23_facts() {
    rn_hw_facts facts;
    facts.soc_model = "SM8550";
    facts.hexagon_arch = "Hexagon v73";
    facts.gpu_name = "QUALCOMM Adreno(TM)";
    facts.gpu_version = "OpenCL 3.0 Adreno(TM) 740";
    facts.gpu_driver = "OpenCL 3.0 QUALCOMM Compiler E031.41.03.62";
    facts.dotprod = true;
    return facts;
}

bool parse(const nlohmann::ordered_json & json, llama_governor_params & params,
           governor_load_options & options) {
    llama_governor_thermo_profile thermo{};
    try {
        return parse_governor_params(json, params, thermo, options);
    } catch (const std::exception & error) {
        std::cerr << "parse failed: " << error.what() << '\n';
        return false;
    }
}

bool all_unsent(const governor_v3_keys & sent) {
    return !sent.hop_tokens && !sent.leg_weighting && !sent.heat_weight &&
        !sent.heat_per_token && !sent.load_step && !sent.guard_headroom_c &&
        !sent.headroom_tau_s;
}

bool defaults_match(const llama_governor_params & params, const rn_leg_set & row) {
    return params.decode_hop_tokens == row.decode_hop_tokens &&
        params.decode_leg_weighting == row.decode_leg_weighting &&
        params.decode_heat_weight == row.decode_heat_weight &&
        params.decode_heat_per_token_npu == row.decode_heat_per_token_npu &&
        params.decode_heat_per_token_gpu == row.decode_heat_per_token_gpu &&
        params.decode_heat_per_token_cpu == row.decode_heat_per_token_cpu &&
        params.decode_load_step_npu_c == row.decode_load_step_npu_c &&
        params.decode_load_step_gpu_c == row.decode_load_step_gpu_c &&
        params.decode_load_step_cpu_c == row.decode_load_step_cpu_c &&
        params.decode_guard_headroom_c == row.decode_guard_headroom_c &&
        params.decode_headroom_tau_s == row.decode_headroom_tau_s;
}

bool test_null_v3_keys_use_real_row_defaults() {
    const rn_leg_set row = rn_legs_for(s23_facts());
    if (!row.one_copy) return false;
    const char * keys[] = {
        "decode_hop_tokens", "decode_leg_weighting", "decode_heat_weight",
        "decode_heat_per_token", "decode_load_step", "decode_guard_headroom_c",
        "decode_headroom_tau_s",
    };
    for (const char * key : keys) {
        auto json = base_governor();
        json[key] = nullptr;
        llama_governor_params parsed{};
        governor_load_options options{};
        if (!parse(json, parsed, options) || !all_unsent(options.v3_sent)) return false;
        const auto merged = merge_leg_row_defaults(parsed, options.v3_sent, row);
        if (!defaults_match(merged, row)) return false;
    }
    return true;
}

bool test_sent_values_win_for_each_v3_key() {
    auto json = base_governor();
    json["decode_hop_tokens"] = 0;
    json["decode_leg_weighting"] = "npu_first";
    json["decode_heat_weight"] = 0.25;
    json["decode_heat_per_token"] = {{"npu", 0.15}, {"gpu", 0.35}, {"cpu", 0.30}};
    json["decode_load_step"] = {{"npu", 1.0}, {"gpu", 2.0}, {"cpu", 1.5}};
    json["decode_guard_headroom_c"] = 8.0;
    json["decode_headroom_tau_s"] = 2.0;
    llama_governor_params parsed{};
    governor_load_options options{};
    if (!parse(json, parsed, options)) return false;
    const governor_v3_keys & sent = options.v3_sent;
    if (!(sent.hop_tokens && sent.leg_weighting && sent.heat_weight && sent.heat_per_token &&
          sent.load_step && sent.guard_headroom_c && sent.headroom_tau_s)) return false;
    const auto merged = merge_leg_row_defaults(parsed, sent, rn_legs_for(s23_facts()));
    return merged.decode_hop_tokens == 0 &&
        merged.decode_leg_weighting == llama_governor_leg_weighting::NPU_FIRST &&
        merged.decode_heat_weight == 0.25f &&
        merged.decode_heat_per_token_npu == 0.15f &&
        merged.decode_heat_per_token_gpu == 0.35f &&
        merged.decode_heat_per_token_cpu == 0.30f &&
        merged.decode_load_step_npu_c == 1.0f &&
        merged.decode_load_step_gpu_c == 2.0f &&
        merged.decode_load_step_cpu_c == 1.5f &&
        merged.decode_guard_headroom_c == 8.0f &&
        merged.decode_headroom_tau_s == 2.0f;
}

bool test_invalid_v3_values_refuse() {
    llama_governor_params params{};
    governor_load_options options{};
    auto json = base_governor();
    json["decode_leg_weighting"] = "";
    if (parse(json, params, options)) return false;
    json["decode_leg_weighting"] = "heat_rank";
    json["decode_heat_weight"] = static_cast<double>(std::numeric_limits<float>::max()) * 2.0;
    if (parse(json, params, options)) return false;
    json["decode_heat_weight"] = 0.25;
    json["decode_heat_per_token"] = {{"npu", 0.15}, {"gpu", 0.35}};
    return !parse(json, params, options);
}

bool test_presence_waits_for_thermo_validation() {
    auto json = base_governor();
    json["decode_hop_tokens"] = 32;
    json["thermo"]["sensor_valid"] = false;
    llama_governor_params params{};
    governor_load_options options{};
    options.v3_sent.hop_tokens = true;
    return !parse(json, params, options) && !options.v3_sent.hop_tokens;
}

} // namespace

int main() {
    TestResults results;
    results.run("null v3 keys are absent and take the real S23 row", test_null_v3_keys_use_real_row_defaults());
    results.run("sent v3 values win, including hop_tokens zero", test_sent_values_win_for_each_v3_key());
    results.run("empty weighting, float overflow, and partial legs refuse", test_invalid_v3_values_refuse());
    results.run("presence is recorded only after thermo validation", test_presence_waits_for_thermo_validation());
    std::cout << "Passed " << results.passed << '/' << results.total << '\n';
    return results.passed == results.total ? 0 : 1;
}
