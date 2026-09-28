#include "llama-governor-device.h"

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
