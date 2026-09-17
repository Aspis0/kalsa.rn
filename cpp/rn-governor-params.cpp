#include "rn-governor-params.h"

#include <cmath>
#include <string>
#include <stdexcept>

namespace rnllama {

namespace {

template <typename T>
T value_or(const nlohmann::ordered_json & object, const char * name, T fallback) {
    if (!object.contains(name)) {
        return fallback;
    }
    if (!object.at(name).is_number()) {
        throw std::invalid_argument(std::string("governor.") + name + " must be a number");
    }
    return object.at(name).get<T>();
}

bool bool_or(const nlohmann::ordered_json & object, const char * name, bool fallback) {
    if (!object.contains(name)) {
        return fallback;
    }
    if (!object.at(name).is_boolean()) {
        throw std::invalid_argument(std::string("governor.") + name + " must be a boolean");
    }
    return object.at(name).get<bool>();
}

std::string string_or(const nlohmann::ordered_json & object, const char * name) {
    if (!object.contains(name)) {
        return {};
    }
    if (!object.at(name).is_string()) {
        throw std::invalid_argument(std::string("governor.") + name + " must be a string");
    }
    return object.at(name).get<std::string>();
}

llama_governor_generation generation_from(const std::string & value) {
    if (value.empty() || value == "Unknown") return llama_governor_generation::Unknown;
    if (value == "NoHTP") return llama_governor_generation::NoHTP;
    if (value == "V73") return llama_governor_generation::V73;
    if (value == "V75") return llama_governor_generation::V75;
    if (value == "V79") return llama_governor_generation::V79;
    throw std::invalid_argument("Unsupported governor.generation: " + value);
}

llama_governor_model_kind model_kind_from(const std::string & value) {
    if (value.empty() || value == "Unknown") return llama_governor_model_kind::Unknown;
    if (value == "Dense") return llama_governor_model_kind::Dense;
    if (value == "Hybrid") return llama_governor_model_kind::Hybrid;
    if (value == "MoE") return llama_governor_model_kind::MoE;
    throw std::invalid_argument("Unsupported governor.model_kind: " + value);
}

llama_governor_fit fit_from(const std::string & value, const char * name) {
    if (value.empty() || value == "Unknown") return llama_governor_fit::Unknown;
    if (value == "Fit") return llama_governor_fit::Fit;
    if (value == "NoFit" || value == "NotFit") return llama_governor_fit::NotFit;
    throw std::invalid_argument(std::string("Unsupported governor.") + name + ": " + value);
}

llama_governor_cool_pays cool_pays_from(const std::string & value) {
    if (value.empty() || value == "Unknown") return llama_governor_cool_pays::Unknown;
    if (value == "No") return llama_governor_cool_pays::No;
    if (value == "Yes") return llama_governor_cool_pays::Yes;
    throw std::invalid_argument("Unsupported governor.cool_pays: " + value);
}

} // namespace

llama_governor_thermo_profile parse_governor_thermo(
    const nlohmann::ordered_json & thermo) {
    if (!thermo.is_object()) {
        throw std::invalid_argument("governor.thermo must be a JSON object");
    }

    llama_governor_thermo_profile result{};
    result.batt_temp_tenths_c = value_or(thermo, "batt_temp_tenths_c", 0);
    result.batt_level_pct = value_or(thermo, "batt_level_pct", 100);
    result.plugged = bool_or(thermo, "plugged", false);
    result.sensor_valid = bool_or(thermo, "sensor_valid", false);
    result.t_idle_valid = bool_or(thermo, "t_idle_valid", false);
    result.t_idle_c = value_or(thermo, "t_idle_c", 0.0f);
    result.trend_c_per_min = value_or(thermo, "trend_c_per_min", 0.0f);
    return result;
}

bool governor_thermo_profile_is_valid(
    const llama_governor_thermo_profile & thermo) {
    if (!thermo.sensor_valid || !std::isfinite(thermo.trend_c_per_min)) {
        return false;
    }
    if (thermo.batt_level_pct < 0 || thermo.batt_level_pct > 100) {
        return false;
    }
    if (!thermo.plugged) {
        return true;
    }
    return thermo.t_idle_valid && std::isfinite(thermo.t_idle_c) &&
           thermo.t_idle_c + 1.0f < 42.0f;
}

bool parse_governor_params(
    const nlohmann::ordered_json & governor,
    llama_governor_params & params,
    llama_governor_thermo_profile & thermo) {
    if (!governor.is_object()) {
        throw std::invalid_argument("governor must be a JSON object");
    }
    if (!governor.contains("enabled") || !governor.at("enabled").is_boolean()) {
        throw std::invalid_argument("governor.enabled must be a boolean");
    }
    if (!governor.at("enabled").get<bool>()) {
        return false;
    }
    if (!governor.contains("thermo")) {
        throw std::invalid_argument("governor.thermo is required");
    }

    params = llama_governor_params{};
    params.reload_budget_available = bool_or(governor, "reload_budget_available", false);
    params.schema_version = value_or(governor, "schema_version", params.schema_version);
    params.capability_schema_version = value_or(
        governor, "capability_schema_version", params.capability_schema_version);
    params.generation = generation_from(string_or(governor, "generation"));
    params.model_kind = model_kind_from(string_or(governor, "model_kind"));
    params.gpu_fit = fit_from(string_or(governor, "gpu_fit"), "gpu_fit");
    params.npu_fit = fit_from(string_or(governor, "npu_fit"), "npu_fit");
    params.htp_trunk_readable = bool_or(governor, "htp_trunk_readable", false);
    params.htp_experts_readable = bool_or(governor, "htp_experts_readable", false);
    params.npu_lane_enabled = bool_or(governor, "npu_lane_enabled", false);
    params.gpu_prefill_measured = bool_or(governor, "gpu_prefill_measured", false);
    params.bench_force_gpu_prefill = bool_or(governor, "bench_force_gpu_prefill", false);
    params.cool_prefill_eligible = bool_or(governor, "cool_prefill_eligible", false);
    params.cool_delta_measured = bool_or(governor, "cool_delta_measured", false);
    params.kexp_cool_scope = bool_or(governor, "kexp_cool_scope", false);
    params.cool_pays = cool_pays_from(string_or(governor, "cool_pays"));
    params.admission_margin_c = value_or(
        governor, "admission_margin_c", params.admission_margin_c);
    params.cache_budget_bytes = value_or(
        governor, "cache_budget_bytes", params.cache_budget_bytes);
    params.expert_cycle_bytes = value_or(
        governor, "expert_cycle_bytes", params.expert_cycle_bytes);
    params.expert_substitution_lambda = value_or(
        governor, "expert_substitution_lambda", params.expert_substitution_lambda);
    if (params.npu_lane_enabled) {
        throw std::invalid_argument("Governor NPU lane is not supported");
    }

    thermo = parse_governor_thermo(governor.at("thermo"));
    if (!governor_thermo_profile_is_valid(thermo)) {
        throw std::invalid_argument("governor: thermo profile invalid");
    }
    return true;
}

} // namespace rnllama
