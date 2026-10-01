#pragma once

#include "ggml-backend.h"

#include <string>
#include <vector>

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

/**
 * The device that holds the most of a model's repeating layers, from the
 * per-layer device names in layer order. A tie goes to the device of the
 * higher layer index; an empty list gives an empty name. Pure over names, so
 * the rule is testable without a model or a backend registry.
 *
 * This is deliberately NOT the output layer's device (llama_model::dev_output):
 * under partial offload or a multi-device split the output layer sits on
 * another device than the bulk of the layers, and with the KV cache not
 * offloaded the attention between the KV store and its output runs on the CPU.
 */
std::string llama_governor_majority_device(const std::vector<std::string> & layer_devices);
