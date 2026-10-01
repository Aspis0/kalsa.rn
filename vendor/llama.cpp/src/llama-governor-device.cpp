#include "llama-governor-device.h"

#include <cstddef>
#include <map>

std::string llama_governor_majority_device(const std::vector<std::string> & layer_devices) {
    // `>=` hands the lead to a device that ties the current best. Layers are
    // counted in order, so that device is the one seen last, and a tie goes to
    // the device of the higher layer index.
    std::map<std::string, std::size_t> counts;
    std::string best;
    std::size_t best_count = 0;
    for (const std::string & name : layer_devices) {
        const std::size_t count = ++counts[name];
        if (count >= best_count) {
            best_count = count;
            best = name;
        }
    }
    return best;
}

llama_governor_prefill_device llama_governor_resolve_prefill_device() {
    llama_governor_prefill_device result; // device nullptr, npu_device "GPU"
    // No ggml_backend_load_all() here: registration is the load step's job
    // (see the header contract), and this function must never race it.
    auto * dev = ggml_backend_dev_by_name("HTP0");
    // Same rejection as common/arg.cpp parse_device_list, but as a degrade
    // decision instead of the exception that would kill the governor load.
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        result.npu_fallback = "htp-device-missing";
        return result;
    }
    result.device = dev;
    result.npu_device = "HTP0";
    result.npu_fallback = nullptr;
    return result;
}
