#pragma once

#include "llama-ext.h"
#include "nlohmann/json.hpp"

#include "ggml-backend.h"

#include <vector>

namespace rnllama {

// Binding-level governor load options: they shape how the two models are
// loaded here, so they do not travel through the engine's params struct.
struct governor_load_options {
    // Repack on is upstream behaviour and the right default on 12 GB+ phones.
    // false is the 8 GB S23 shape, where the lane only fits without the CPU
    // repack copy (P1): repack-off costs ~1.41x lane decode speed and KLD
    // p99 0.0341 -> 0.0422.
    bool decode_repack = true;
};

bool parse_governor_params(
    const nlohmann::ordered_json & governor,
    llama_governor_params & params,
    llama_governor_thermo_profile & thermo,
    governor_load_options & options);

llama_governor_thermo_profile parse_governor_thermo(
    const nlohmann::ordered_json & thermo);

bool governor_thermo_profile_is_valid(
    const llama_governor_thermo_profile & thermo);

/** The prefill-device plan the loader applies and the app reports on the
 *  KALSA_GOVERNOR_PLAN line: use_device (the loader sets
 *  llama_model_params.devices = {device, nullptr} from the resolver's own
 *  result), npu_device ("HTP0" | "GPU") and npu_fallback (null when the lane
 *  resolves, otherwise the reason it degraded). Pure booleans/strings, so
 *  this TU stays linkable without ggml. */
struct governor_prefill_device_plan {
    bool use_device = false;
    const char * npu_device = "GPU";
    const char * npu_fallback = nullptr;
};

/** Resolve-to-device decision, pure. `htp_init_reason` is the Java-side
 *  KALSA_HTP_FALLBACK env (RNLlama.java noteHtpFallback) and outranks a
 *  device that resolves: no ADSP dir/libs/env means HTP0 is unusable even
 *  when the backend registers. `engine_fallback` is the resolver's own
 *  npu_fallback ("htp-device-missing" | nullptr). */
governor_prefill_device_plan decide_governor_prefill_device(
    bool device_resolved,
    const char * engine_fallback,
    const char * htp_init_reason);

/** Registered devices minus every device of the `excluded` registry, order
 *  preserved. Inputs may carry the null terminator the engine's device loop
 *  expects (JSI lists append one); terminators pass through skipped — the
 *  caller re-appends one. Pointer filtering only (reads dev->reg, calls no
 *  ggml symbol), so this TU stays linkable without ggml and the host test
 *  drives it with fabricated devices. The caller resolves `excluded`
 *  (ggml_backend_reg_by_name("HTP") — nullptr when Hexagon is not compiled
 *  in). */
std::vector<ggml_backend_dev_t> devices_excluding_registry(
    const std::vector<ggml_backend_dev_t> & devs, ggml_backend_reg_t excluded);

} // namespace rnllama
