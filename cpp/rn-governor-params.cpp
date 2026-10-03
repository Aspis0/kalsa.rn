#include "rn-governor-params.h"

// struct ggml_backend_device definition: the registry filter below reads
// dev->reg. Header-only — the TU still links without ggml.
#include "ggml-backend-impl.h"

#include <cmath>
#include <string>
#include <string_view>
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
    if (value == "V81") return llama_governor_generation::V81;
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
    // Absent, wrong type or fractional is -1: no platform vote, never a
    // throw (value_or would throw on a string). JSI numbers arrive as
    // doubles (toJson keeps number precision), so integrality is a value
    // check; the int32 bounds keep the cast defined for Infinity/NaN.
    // The engine owns the semantic range - it clamps out-of-range to
    // absent at ingress - so no range check duplicates it here.
    result.platform_thermal_status = -1;
    const auto status = thermo.find("platform_thermal_status");
    if (status != thermo.end() && status->is_number()) {
        const double raw = status->get<double>();
        if (raw == std::trunc(raw) && raw >= -2147483648.0 && raw <= 2147483647.0) {
            result.platform_thermal_status = static_cast<int32_t>(raw);
        }
    }
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
    return true;
}

bool parse_governor_params(
    const nlohmann::ordered_json & governor,
    llama_governor_params & params,
    llama_governor_thermo_profile & thermo,
    governor_load_options & options) {
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
    options.decode_repack = bool_or(governor, "decode_repack", true);
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
    // N > 0 alternates decode CPU / NPU lane every N generated tokens; 0 or
    // absent keeps CPU. JSI numbers are doubles: the range check also refuses
    // NaN before the cast.
    const double hop = value_or(governor, "decode_hop_tokens", 0.0);
    if (!(hop >= 0.0 && hop <= 4294967295.0)) {
        throw std::invalid_argument("governor.decode_hop_tokens must be a non-negative uint32");
    }
    params.decode_hop_tokens = static_cast<uint32_t>(hop);
    // npu_lane_enabled is forwarded: the engine's prefill_engine owns the
    // lane (owner rule 2026-09-28) and load_governor_models decides the
    // device with llama_governor_resolve_prefill_device + degrade.

    thermo = parse_governor_thermo(governor.at("thermo"));
    if (!governor_thermo_profile_is_valid(thermo)) {
        throw std::invalid_argument("governor: thermo profile invalid");
    }
    return true;
}

governor_prefill_device_plan decide_governor_prefill_device(
        bool device_resolved, const char * engine_fallback, const char * htp_init_reason) {
    governor_prefill_device_plan plan{};
    if (htp_init_reason != nullptr && *htp_init_reason != '\0') {
        plan.npu_fallback = htp_init_reason;
        return plan;
    }
    if (device_resolved) {
        plan.use_device = true;
        plan.npu_device = "HTP0";
        plan.npu_fallback = engine_fallback;
        return plan;
    }
    plan.npu_fallback = engine_fallback;
    return plan;
}

bool governor_lane_policy_enabled(
        bool lane_enabled, const governor_prefill_device_plan & plan) {
    return lane_enabled && plan.use_device;
}

bool htp_runtime_failure(
        int32_t n_tokens, llama_governor_engine prefill_engine,
        llama_governor_engine decode_engine, int32_t rc,
        const char * failure_reason) {
    // A batch's engine is its phase's stamp: the route latch for prefill,
    // select_decode for the 1-token batches it routes (an NPU decode runs
    // on ctx_prefill, the HTP-pinned context).
    const llama_governor_engine engine =
        n_tokens > 1 ? prefill_engine : decode_engine;
    if (engine != llama_governor_engine::NPU || rc != -3) {
        return false;
    }
    return failure_reason != nullptr &&
        std::string_view(failure_reason).rfind("llama_decode failed with rc=", 0) == 0;
}

std::vector<ggml_backend_dev_t> devices_excluding_registry(
        const std::vector<ggml_backend_dev_t> & devs, ggml_backend_reg_t excluded) {
    std::vector<ggml_backend_dev_t> kept;
    kept.reserve(devs.size());
    for (ggml_backend_dev_t dev : devs) {
        if (dev != nullptr && dev->reg != excluded) {
            kept.push_back(dev);
        }
    }
    return kept;
}

} // namespace rnllama
