#pragma once

#include "llama-ext.h"
#include "nlohmann/json.hpp"

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

} // namespace rnllama
