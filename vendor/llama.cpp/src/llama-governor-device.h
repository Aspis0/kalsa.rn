#pragma once

#include "ggml-backend.h"

/**
 * Resolution of the HTP prefill device for the governor's NPU lane.
 *
 * A phone without the lane must DEGRADE to GPU, never fail: the CLI-style
 * device list (common/arg.cpp parse_device_list) throws "invalid device" for
 * an unresolvable name, so the loader must resolve FIRST and set
 * llama_model_params.devices only when the returned device is non-null.
 *
 * npu_device / npu_fallback are the plan-field values (KALSA_GOVERNOR_PLAN):
 * "HTP0" when the lane resolves, otherwise the degrade target "GPU" with
 * npu_fallback naming the reason. Only the engine-side reason lives here —
 * the binding adds its own (htp-dir / htp-libs / htp-env) when it degrades
 * before the engine ever sees a device.
 */
struct llama_governor_prefill_device {
    ggml_backend_dev_t device = nullptr;
    const char * npu_device = "GPU";
    const char * npu_fallback = nullptr;
};

/**
 * Resolve the HTP0 prefill device; device == nullptr means degrade to GPU.
 *
 * Registration contract: backends must ALREADY be registered by the load
 * step that owns this call — the binding calls it from load_governor_models,
 * whose JSI task ran ensureBackendInitialized() first (RNLlamaJSI.cpp:660,
 * std::call_once + llama_backend_init); the devices test calls it after
 * llama_backend_init() in main (tests/test-governor-v0-devices.cpp). The
 * resolver itself never registers and is never called from completion
 * threads, so the ggml backend registry is touched only inside that one
 * load step.
 */
llama_governor_prefill_device llama_governor_resolve_prefill_device();
