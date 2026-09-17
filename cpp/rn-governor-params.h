#pragma once

#include "llama-ext.h"
#include "nlohmann/json.hpp"

namespace rnllama {

bool parse_governor_params(
    const nlohmann::ordered_json & governor,
    llama_governor_params & params,
    llama_governor_thermo_profile & thermo);

llama_governor_thermo_profile parse_governor_thermo(
    const nlohmann::ordered_json & thermo);

bool governor_thermo_profile_is_valid(
    const llama_governor_thermo_profile & thermo);

} // namespace rnllama
