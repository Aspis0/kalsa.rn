#pragma once

#include "llama-ext.h"
#include "nlohmann/json.hpp"
#include "rn-legs-table.h"

#include "ggml-backend.h"

#include <vector>

namespace rnllama {

// Which rule v3 governor keys the app JSON explicitly carried (audit F1/P1).
// On the one-copy load (matched row plus npu_lane_enabled), the row supplies
// the DEFAULT for every rule v3 field; a key the app sent wins, so bench keys /
// kalsa.bench.* overrides keep working. Produced by parse_governor_params -
// no globals.
struct governor_v3_keys {
    bool hop_tokens = false;
    bool leg_weighting = false;
    bool heat_weight = false;
    bool heat_per_token = false;
    bool load_step = false;
    bool guard_headroom_c = false;
    bool headroom_tau_s = false;
};

// Binding-level governor load inputs parse_governor_params produces; they do
// not travel through the engine's params struct.
struct governor_load_options {
    // Repack on is upstream behaviour and the right default on 12 GB+ phones.
    // false is the 8 GB S23 shape, where the lane only fits without the CPU
    // repack copy (P1): repack-off costs ~1.41x lane decode speed and KLD
    // p99 0.0341 -> 0.0422.
    bool decode_repack = true;
    // The explicit-key record for the row-defaults precedence on the
    // one-copy load (merge_leg_row_defaults).
    governor_v3_keys v3_sent{};
};

bool parse_governor_params(
    const nlohmann::ordered_json & governor,
    llama_governor_params & params,
    llama_governor_thermo_profile & thermo,
    governor_load_options & options);

/** The effective decode params of the one-copy load (matched row plus
 *  npu_lane_enabled): the row supplies each rule v3 default, and an explicitly
 *  sent app key wins. Only applied on the one-copy branch; pure. */
llama_governor_params merge_leg_row_defaults(
    const llama_governor_params & params,
    const governor_v3_keys & sent,
    const rn_leg_set & row);

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

/** The reason recorded when the HTP prefill compute fails mid-session (the
 *  engine returns an error instead of hanging on a DSP timeout/skew). The
 *  recorder stores it via owner.setGovernorNpuFallback - getGovernorStats
 *  publishes it and the binding's KALSA_GOVERNOR_FALLBACK log line carries
 *  it at failure time - and in the KALSA_HTP_FALLBACK env, which
 *  decide_governor_prefill_device reads at every governor load: the app's
 *  retry reload itself runs CPU-only (governor off), and the env is what
 *  keeps the NEXT governor load off HTP even when the device resolves. */
constexpr char KALSA_HTP_RUNTIME_FALLBACK[] = "htp-runtime-error";

/** True when one governor decode failure is attributable to the HTP device —
 *  the only failure that may kill the lane for the session. The attribution
 *  question is "did the failing batch run on the NPU", and a batch's engine
 *  is its phase's stamp: prefill (n_tokens > 1, the engine's own is_prefill
 *  predicate, llama-governor.cpp decode_impl) reports the route latch's
 *  stamp for THIS batch (llama-governor-runtime.cpp: stats_.prefill_engine =
 *  admission.engine — a CPU-latched prefill reports CPU); a 1-token decode
 *  batch reports select_decode's stamp (llama-governor-runtime.cpp:
 *  stats_.decode_engine = selection.engine) — an NPU decode runs on
 *  ctx_prefill, the HTP-pinned context (decode_impl), so a decode-hop
 *  failure reaches the DSP exactly like an HTP prefill failure.
 *  rc == -3 is the engine's compute-failure map on both paths
 *  (llama-context.cpp "case GGML_STATUS_FAILED: return -3"); and the reason
 *  prefix confirms the batch failed inside llama_decode itself, not in a
 *  host-side phase handoff ("KV commit failed: ..." from the OpenCL
 *  set_tensor path, "route Reject during ...", llama-governor.cpp
 *  commit_side) — those never touched the DSP. Anything else keeps the lane
 *  on: only a confident HTP attribution disables it. Pure, so the host test
 *  drives it with literals. */
bool htp_runtime_failure(
    int32_t n_tokens,
    llama_governor_engine prefill_engine,
    llama_governor_engine decode_engine,
    int32_t rc,
    const char * failure_reason);

/** Resolve-to-device decision, pure. `htp_init_reason` is the Java-side
 *  KALSA_HTP_FALLBACK env (RNLlama.java noteHtpFallback) and outranks a
 *  device that resolves: no ADSP dir/libs/env means HTP0 is unusable even
 *  when the backend registers. `engine_fallback` is the resolver's own
 *  npu_fallback ("htp-device-missing" | nullptr). */
governor_prefill_device_plan decide_governor_prefill_device(
    bool device_resolved,
    const char * engine_fallback,
    const char * htp_init_reason);

/** npu_lane_enabled as the engine policy must see it. The loader's device
 *  decision and the engine's engine decision are separate layers: when the
 *  lane is asked for but does not resolve, the load keeps the default
 *  device list (on Android the unqualified OpenCL GPU) while
 *  prefill_engine(), fed the raw flag, would still claim NPU. Only a plan
 *  whose device actually resolved may leave the lane on for the policy —
 *  the loader builds llama_governor_policy from this verdict and keeps
 *  every report line on the original params. Pure. */
bool governor_lane_policy_enabled(
    bool lane_enabled, const governor_prefill_device_plan & plan);

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
