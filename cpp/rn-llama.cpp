#include "rn-llama.h"
#include "bmoe_stream.h"
#include "ggml-cpu.h"
#include "rn-tts.h"
#include "rn-mtmd.hpp"
#include "rn-completion.h"
#include "rn-governor.h"
#include "rn-governor-params.h"
#include "rn-legs-table.h"
#include "llama-governor-device.h"
#include "ggml-hexagon.h"
#include "rn-slot-manager.h"
#include "rn-common.hpp"

// Include multimodal support
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include "tools/mtmd/clip.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

#if defined(__ANDROID__)
#include <android/set_abort_message.h>
#endif

namespace rnllama {

namespace {

void populate_lora_metadata(common_adapter_lora_info &la) {
    if (la.ptr == nullptr) {
        la.task_name.clear();
        la.prompt_prefix.clear();
        return;
    }

    char buf[1024];
    llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
    la.task_name = buf;
    llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
    la.prompt_prefix = buf;
}

void clear_init_lora_ownership(common_init_result_ptr &llama_init) {
    if (llama_init != nullptr) {
        // common_init_from_params() owns adapters loaded during init. Once runtime
        // adapter state changes, those original handles should be released too.
        llama_init->lora().clear();
    }
}

bool has_speculative_type(const common_params_speculative &speculative, common_speculative_type type) {
    return std::find(speculative.types.begin(), speculative.types.end(), type) != speculative.types.end();
}

bool has_speculative_mode(const common_params & params) {
    return std::any_of(params.speculative.types.begin(), params.speculative.types.end(), [](auto type) {
        return type != COMMON_SPECULATIVE_TYPE_NONE;
    });
}

void log_governor_fallback(const char * stage, int models_loaded,
                           const std::string & reason,
                           llama_governor_fit gpu_fit, bool profile_valid) {
    LOG_ERROR(
        "KALSA_GOVERNOR_FALLBACK {stage:\"%s\", models_loaded:%d, reason:\"%s\", gpu_fit:%d, profile_valid:%d}",
        stage, models_loaded, reason.c_str(), (int) gpu_fit, (int) profile_valid);
}

#if defined(__ANDROID__)
// Device list for a lane-off load: HTP must appear nowhere, neither in
// prefill placement nor in the decode model's host-buffer pick (the Hexagon
// device exposes a host buffer type, ggml-hexagon.cpp, so an HTP entry would
// map decode weights through FastRPC). A pinned list is filtered as-is —
// the JSI default list already excludes HTP, an explicitly requested one
// does not — and with no list at all the registered non-HTP GPUs replace
// the engine's implicit sweep, which would include HTP. reg_by_name
// registers nothing: the JSI task has already run
// ensureBackendInitialized() before this runs.
std::vector<ggml_backend_dev_t> lane_off_devices(const std::vector<ggml_backend_dev_t> & requested) {
    std::vector<ggml_backend_dev_t> source = requested;
    if (source.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                source.push_back(dev);
            }
        }
    }
    return devices_excluding_registry(source, ggml_backend_reg_by_name("HTP"));
}
#endif

// The cache types the HTP prefill device can write: SET_ROWS takes only
// F32/F16/Q8_0 destinations (ggml_hexagon_supported_set_rows,
// ggml-hexagon.cpp) and FLASH_ATTN_EXT reads K/V as F16/Q8_0. A caller type
// outside this set aborts the lane's load with a pre-allocated KV
// destination no backend can write (S23, ggml-backend.cpp).
bool htp_kv_type_supported(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0;
}

// The lane's one KV decision, applied to both contexts. With flash
// attention explicitly disabled it keeps the pre-lane-KV behaviour: the
// engine refuses a quantized V cache without FA (llama-context.cpp
// "quantized V cache requires flash_attn to be enabled") and stores the V
// rows transposed (llama-model.cpp attn_v_trans = !cparams.flash_attn) —
// a layout the HTP path was never validated for.
struct governor_lane_kv_plan {
    bool kv_on_device;
    ggml_type type_k;
    ggml_type type_v;
};

governor_lane_kv_plan decide_governor_lane_kv(
    ggml_type requested_k, ggml_type requested_v,
    enum llama_flash_attn_type flash_attn) {
    if (flash_attn == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        return { false, requested_k, requested_v };
    }
    return {
        true,
        htp_kv_type_supported(requested_k) ? requested_k : GGML_TYPE_Q8_0,
        htp_kv_type_supported(requested_v) ? requested_v : GGML_TYPE_Q8_0,
    };
}

// The OpenCL leg holds cl_mem imports of the model's Hexagon dma-bufs; they
// must be released after the contexts that use them are gone and before the
// model's buffers free. A build without the OpenCL backend (or an idle one,
// no imports) is a no-op. F1: the import cache is process-wide, so this runs
// ONLY in a one-copy teardown - never on plain reloads, where another owner's
// in-flight compute could still hold an imported cl_mem.
void release_opencl_imports() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("OpenCL");
    if (reg == nullptr) {
        return;
    }
    auto release = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_release_imports"));
    if (release != nullptr) {
        release();
    }
}

// F3: the shared-weights setter changes unsynchronized backend globals
// (ggml-hexagon.h caller contract) and JSI loads run on pool workers, so the
// one-copy load (setter through governor init) and the one-copy teardown
// (through the restore) serialize process-wide on this one mutex.
std::mutex & onecopy_load_mutex() {
    static std::mutex mutex;
    return mutex;
}

// The hexagon shared-weights setter, looked up once per use; null when the
// build has no Hexagon backend or it predates the proc.
ggml_backend_hexagon_set_shared_weights_t hexagon_shared_weights_setter() {
    ggml_backend_reg_t htp_reg = ggml_backend_reg_by_name("HTP");
    return htp_reg == nullptr ? nullptr :
        reinterpret_cast<ggml_backend_hexagon_set_shared_weights_t>(
            ggml_backend_reg_get_proc_address(htp_reg,
                                              "ggml_backend_hexagon_set_shared_weights"));
}

// F2: restore the Hexagon globals the one-copy setter changed (shared weights
// off, buffer cap back). No getter exists (ggml-hexagon.h), so the prior value
// is the engine's documented state, reconstructed exactly as the process
// started: opt_mbuf defaults to 1 GiB (vendored ggml-hexagon.cpp:100) and
// GGML_HEXAGON_MBUF seeds it at init as strtoul(value, NULL, 0) MiB (vendored
// ggml-hexagon.cpp:8756). Returns false when the setter refused - a shared
// buffer is still alive - which the callers must not ignore.
bool restore_hexagon_shared_weights() {
    auto set_shared_weights = hexagon_shared_weights_setter();
    if (set_shared_weights == nullptr) {
        return true;  // nothing was ever changed in this build
    }
    size_t max_buffer_bytes = 1ull * 1024 * 1024 * 1024;
    if (const char * mbuf = std::getenv("GGML_HEXAGON_MBUF")) {
        const unsigned long mib = std::strtoul(mbuf, nullptr, 0);
        if (mib > 0) {
            max_buffer_bytes = (size_t) mib * 1024 * 1024;
        }
    }
    const ggml_hexagon_shared_weights restore { false, max_buffer_bytes };
    return set_shared_weights(&restore) == 0;
}

// The one-copy teardown, caller holds onecopy_load_mutex (F3): the governor's
// leg contexts free first, then the process-wide OpenCL imports (F1), then
// the models.
void teardown_onecopy_unlocked(llama_rn_context & owner) {
    owner.governor.reset();
    release_opencl_imports();
    owner.governor_onecopy_init.reset();
    owner.governor_decode_init.reset();
    owner.governor_prefill_init.reset();
    owner.llama_init.reset();
    owner.model = nullptr;
    owner.ctx = nullptr;
}

// One teardown for every governor load, both forms. A one-copy owner adds the
// OpenCL import release and the Hexagon globals restore (F1/F2), the whole
// sequence under the one-copy mutex (F3); the restore can be refused only
// when another owner still holds shared Hexagon buffers, in which case the
// globals stay as that live load needs them - logged, nothing of this owner's
// is left alive.
void teardown_governor_load(llama_rn_context & owner) {
    if (owner.governor_onecopy_init == nullptr) {
        owner.governor.reset();
        owner.governor_decode_init.reset();
        owner.governor_prefill_init.reset();
        owner.llama_init.reset();
        owner.model = nullptr;
        owner.ctx = nullptr;
        return;
    }
    const std::lock_guard<std::mutex> lock(onecopy_load_mutex());
    teardown_onecopy_unlocked(owner);
    if (!restore_hexagon_shared_weights()) {
        LOG_ERROR("KALSA_ONECOPY_RESTORE_REFUSED {reason:\"shared Hexagon buffers still live\"}");
    }
}

// F-12: the legs pick the accelerators by their EXACT registered names -
// never a prefix or "first other GPU" match, which could load the wrong
// backend.
ggml_backend_dev_t find_registered_device(const char * wanted) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_dev_name(dev), wanted) == 0) {
            return dev;
        }
    }
    return nullptr;
}

// Cap for the shared Hexagon buffers: every TILE32/HOST buffer becomes one
// dma-buf the OpenCL leg imports, so each must stay under the S23's
// CL_DEVICE_MAX_MEM_ALLOC_SIZE (884 MB). The lab arms ran
// GGML_HEXAGON_MBUF=832 (one-copy REPORT, step 4 buffer census).
constexpr size_t k_onecopy_shared_mbuf_bytes = (size_t) 832 * 1024 * 1024;

enum class onecopy_load_result {
    loaded,
    // The two-model branch runs instead in this process - either refused
    // before anything was allocated, or torn down and restored after the
    // setter (F7).
    demoted,
    // The one-copy attempt failed AND the Hexagon globals could not be
    // restored (a shared buffer is still live): fail the load - never load
    // two-copy on top of a live shared buffer.
    failed,
};

// The one-copy three-leg load (FABLE (b), step 6): ONE model load per the
// engine's load contract, legs built through the one-model governor. Every
// refusal here demotes to today's loads - never to an NPU load without the
// OpenCL HOST leg (J(1)).
onecopy_load_result load_governor_one_model(
        llama_rn_context & owner,
        const llama_governor_params & governor_params,
        const llama_governor_thermo_profile & governor_thermo,
        const governor_load_options & load_options,
        const rn_leg_set & leg_set) {
    auto demote = [](const char * reason) {
        LOG_ERROR("KALSA_GOVERNOR_FALLBACK {stage:\"onecopy\", models_loaded:0, reason:\"%s\"}",
                  reason);
    };

    ggml_backend_dev_t npu = find_registered_device("HTP0");
    ggml_backend_dev_t gpu = find_registered_device("GPUOpenCL");
    if (npu == nullptr || gpu == nullptr) {
        demote(npu == nullptr ? "HTP0 not registered" : "GPUOpenCL not registered");
        return onecopy_load_result::demoted;
    }

    // The step-2 setter: offer the device's TILE32/HOST buffer types to the
    // weight placement (one weights copy for the HTP, CPU and GPU legs) and
    // cap the Hexagon buffer size. It takes effect only before any model
    // load or context creation - here, before anything reads the Hexagon
    // buffer types. Nonzero = refused (a shared buffer is still alive in
    // this process).
    auto set_shared_weights = hexagon_shared_weights_setter();
    if (set_shared_weights == nullptr) {
        demote("hexagon shared-weights setter missing");
        return onecopy_load_result::demoted;
    }
    // F3: the setter's globals are unsynchronized engine state; from here to
    // the governor's construction (and on every failure path, through the
    // teardown and restore) this load owns them under one process-wide lock.
    const std::lock_guard<std::mutex> onecopy_guard(onecopy_load_mutex());
    const ggml_hexagon_shared_weights shared_weights {
        true, k_onecopy_shared_mbuf_bytes };
    if (set_shared_weights(&shared_weights) != 0) {
        demote("shared weights refused (a shared buffer is alive)");
        return onecopy_load_result::demoted;
    }
    // F7: a failure after the setter is still recoverable - tear down, restore
    // the globals, and let the two-model branch load instead. Only a refused
    // restore escalates to failed: two-copy on top of a live shared buffer is
    // the one unrecoverable state.
    auto fail_onecopy = [&](const std::string & reason) -> onecopy_load_result {
        teardown_onecopy_unlocked(owner);
        if (!restore_hexagon_shared_weights()) {
            LOG_ERROR(
                "KALSA_GOVERNOR_FALLBACK {stage:\"onecopy\", models_loaded:0, "
                "reason:\"restore refused after '%s' (shared buffers live)\"}",
                reason.c_str());
            return onecopy_load_result::failed;
        }
        demote(reason.c_str());
        return onecopy_load_result::demoted;
    };

    // The load contract (llama-ext.h): devices {HTP0, GPUOpenCL} with every
    // layer on HTP0 via tensor_split {1,0}, no mmap (the shared buffers are
    // rpcmem, not file pages), no extra buffer types (no CPU_REPACK second
    // copy). The decode_repack knob has no decode model to repack here.
    if (!load_options.decode_repack) {
        LOG_INFO("KALSA_ONECOPY_NOTE {decode_repack:\"ignored (one model, no decode copy)\"}");
    }
    common_params params = owner.params;
    params.devices = { npu, gpu };
    params.tensor_split[0] = 1.0f;
    params.tensor_split[1] = 0.0f;
    params.n_gpu_layers = 99;
    params.n_parallel = 1;
    // No mmap: the shared buffers are rpcmem, not file pages - explicit, not
    // left to the loader's per-device AUTO demotion.
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    params.no_extra_bufts = true;
    // One KV decision for every leg: llama_kv_route_query::compatible
    // refuses mismatched cache types across leg contexts, the way the
    // two-model lane keeps its pair identical.
    const ggml_type requested_k = params.cache_type_k;
    const ggml_type requested_v = params.cache_type_v;
    const auto lane_kv = decide_governor_lane_kv(requested_k, requested_v, params.flash_attn_type);
    if (lane_kv.kv_on_device) {
        params.cache_type_k = lane_kv.type_k;
        params.cache_type_v = lane_kv.type_v;
        if (params.cache_type_k != requested_k || params.cache_type_v != requested_v) {
            LOG_INFO(
                "KALSA_KV_TYPE_OVERRIDE {requested_k:\"%s\", requested_v:\"%s\", "
                "effective_k:\"%s\", effective_v:\"%s\", reason:\"htp-prefill-kv\"}",
                ggml_type_name(requested_k), ggml_type_name(requested_v),
                ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v));
        }
    } else {
        params.no_kv_offload = true;
        LOG_INFO(
            "KALSA_KV_HOST_PIN {requested_k:\"%s\", requested_v:\"%s\", reason:\"flash-attn-off\"}",
            ggml_type_name(requested_k), ggml_type_name(requested_v));
    }

    try {
        owner.governor_onecopy_init = common_init_from_params(params, true);
    } catch (const std::exception & error) {
        return fail_onecopy(error.what());
    }
    if (owner.governor_onecopy_init == nullptr || owner.governor_onecopy_init->model() == nullptr) {
        return fail_onecopy("one-copy model load failed");
    }

    // The legs (owner decision J(1)): the NPU leg lists GPUOpenCL second, so
    // OpenCL takes what the HTP refuses - the tied Q6_K output read in
    // place. Host threads: the NPU and CPU legs take the caller's counts
    // (one host thread starved the J(1) NPU decode to 12-13 tok/s), the GPU
    // leg one (F1). The engine drops an absent leg; the governor ctor logs
    // one fallback line per dropped accelerator leg.
    llama_context_params leg_params = common_context_params_to_llama(params);
    llama_context_params gpu_leg_params = leg_params;
    gpu_leg_params.n_threads = 1;
    gpu_leg_params.n_threads_batch = 1;
    const ggml_backend_dev_t npu_leg_devices[] = { npu, gpu, nullptr };
    const ggml_backend_dev_t gpu_leg_devices[] = { gpu, nullptr };
    const llama_governor_leg legs[] = {
        { llama_governor_engine::NPU, npu_leg_devices, leg_params },
        { llama_governor_engine::GPU, gpu_leg_devices, gpu_leg_params },
        { llama_governor_engine::CPU, nullptr, leg_params },
    };
    // F4: the caller's governor params pass through, with the matched row's
    // decode-hop tuning filled in (owner decision 2026-10-07): the copy is
    // local to this branch, so the two-model path below keeps the caller's
    // params (tau 0, weight 0) whatever happens here. The one-copy
    // branch already required npu_lane_enabled (the kill switch), and the
    // readability facts are not synthesized because the engine's one-model
    // governor proves them by construction: its ctor sets npu_lane_capable
    // when the NPU leg exists, and the policy reads that flag BEFORE the
    // fit / htp_trunk_readable / htp_experts_readable facts (vendored
    // llama-governor-policy.cpp:230-243) - with the NPU leg absent those
    // caller facts decide as they always did.
    llama_governor_params onecopy_params = governor_params;
    onecopy_params.decode_headroom_tau_s = leg_set.decode_headroom_tau_s;
    onecopy_params.decode_heat_weight = leg_set.decode_heat_weight;
    onecopy_params.decode_heat_per_token_npu = leg_set.decode_heat_per_token_npu;
    onecopy_params.decode_heat_per_token_gpu = leg_set.decode_heat_per_token_gpu;
    onecopy_params.decode_heat_per_token_cpu = leg_set.decode_heat_per_token_cpu;
    try {
        owner.governor = std::make_unique<rn_governor>(
            owner.governor_onecopy_init->model(), legs, 3, onecopy_params);
    } catch (const std::exception & error) {
        return fail_onecopy(error.what());
    }

    if (!owner.governor->set_thermo_profile(governor_thermo)) {
        return fail_onecopy("thermo profile invalid");
    }

    owner.model = owner.governor_onecopy_init->model();
    owner.ctx = owner.governor->prefill_ctx();
    owner.params.n_gpu_layers = 99;
    // The app's KALSA_GOVERNOR_PLAN surface: the lane is the table's
    // resolution, HTP0 by the load contract, no fallback.
    owner.setGovernorNpuDevice("HTP0");
    owner.setGovernorNpuFallback(nullptr);
    owner.setGovernorKvCache(
        ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v),
        params.no_kv_offload ? "host" : "device");
    if (owner.model == nullptr || owner.ctx == nullptr) {
        return fail_onecopy("one-copy governor without an active context");
    }
    return onecopy_load_result::loaded;
}

bool load_governor_models(llama_rn_context & owner,
                          const llama_governor_params & governor_params,
                          const llama_governor_thermo_profile & governor_thermo,
                          const governor_load_options & load_options) {
    // The capability table is the only allow-list for the one-copy load:
    // exactly the validated devices get it, everything else - including any
    // fact that did not read - keeps today's loads byte for byte (R2).
    const rn_hw_facts hw_facts = rn_read_hw_facts();
    const rn_leg_set leg_set = rn_legs_for(hw_facts);
    LOG_INFO(
        "KALSA_LEGS_TABLE {soc:\"%s\", hexagon:\"%s\", gpu_name:\"%s\", gpu_version:\"%s\", "
        "gpu_driver:\"%s\", dotprod:%d, one_copy:%d, tau_s:%g, heat_w:%g, "
        "heat_tok_npu:%g, heat_tok_gpu:%g, heat_tok_cpu:%g}",
        hw_facts.soc_model.c_str(), hw_facts.hexagon_arch.c_str(), hw_facts.gpu_name.c_str(),
        hw_facts.gpu_version.c_str(), hw_facts.gpu_driver.c_str(),
        (int) hw_facts.dotprod, (int) leg_set.one_copy, leg_set.decode_headroom_tau_s,
        leg_set.decode_heat_weight, leg_set.decode_heat_per_token_npu,
        leg_set.decode_heat_per_token_gpu, leg_set.decode_heat_per_token_cpu);
    // F4: the caller's NPU kill switch gates the one-copy branch too - the
    // table proves the hardware, npu_lane_enabled says this caller wants the
    // NPU lane at all. With it off, today's path runs (two-model where its
    // own lane logic resolves, the plain load elsewhere).
    if (leg_set.one_copy && governor_params.npu_lane_enabled) {
        // A runtime HTP failure downgrades every later load of this process
        // (note_htp_runtime_fallback): the one-copy prefill IS the HTP lane,
        // so the downgrade applies here too and the two-model branch below
        // handles it.
        const char * htp_init_reason = std::getenv("KALSA_HTP_FALLBACK");
        if (htp_init_reason != nullptr && htp_init_reason[0] != '\0') {
            LOG_ERROR(
                "KALSA_GOVERNOR_FALLBACK {stage:\"onecopy\", models_loaded:0, reason:\"%s\"}",
                htp_init_reason);
        } else {
            switch (load_governor_one_model(owner, governor_params, governor_thermo,
                                            load_options, leg_set)) {
            case onecopy_load_result::loaded:
                return true;
            case onecopy_load_result::demoted:
                break; // fall through to the two-model load below
            case onecopy_load_result::failed:
                return false;
            }
        }
    }

    // Backend registration already happened in this SAME load step: the JSI
    // task called ensureBackendInitialized() (jsi/RNLlamaJSI.cpp:660) before
    // us, and the resolver never registers itself — the ggml registry is
    // only ever touched here. Java-side HTP failures ride KALSA_HTP_FALLBACK
    // (RNLlama.java noteHtpFallback) and outrank a device that resolves.
    common_params prefill_params = owner.params;
    common_params decode_params = owner.params;
    // The rn_governor policy is built from this copy, not from
    // governor_params: a lane that was asked for but did not resolve has
    // its flag cleared below, so prefill_engine() falls through to its own
    // GPU/CPU branches instead of claiming NPU for a context whose devices
    // are the default (unqualified) GPU. Reporting keeps reading
    // governor_params.
    llama_governor_params policy_params = governor_params;
    if (governor_params.npu_lane_enabled) {
        // Lane asked for: resolve the device and fill the plan fields. With
        // the lane off (today's default) the resolver is skipped and both
        // fields stay empty — a default plan must not claim htp-device-missing
        // for a lane nobody asked for.
        const char * htp_init_reason = std::getenv("KALSA_HTP_FALLBACK");
        const auto resolved = llama_governor_resolve_prefill_device();
        const auto device_plan = decide_governor_prefill_device(
            resolved.device != nullptr, resolved.npu_fallback, htp_init_reason);
        owner.setGovernorNpuDevice(device_plan.npu_device);
        // The accessor copies the reason bytes out of environ; the runtime
        // fallback's setenv (note_htp_runtime_fallback) is allowed to
        // reallocate it.
        owner.setGovernorNpuFallback(device_plan.npu_fallback);
        // A lane the resolver could not serve must not reach the engine
        // policy as enabled: prefill_engine() would route prefill to NPU
        // while this load's devices stayed the default GPU.
        policy_params.npu_lane_enabled = governor_lane_policy_enabled(
            governor_params.npu_lane_enabled, device_plan);
        if (device_plan.use_device) {
            prefill_params.devices = { resolved.device, nullptr };
            // One host thread for the HTP-pinned context: on the S23 an HTP
            // pp1024 runs at the same speed on 1 and 5 host threads (892 vs
            // 889 tok/s) with the CPU zones 5-10 C cooler (lab
            // s23-prefill-host-threads). It caps only graph nodes that fall
            // back to the CPU backend, not the DSP's own threads. The CPU
            // decode context keeps the caller's counts.
            prefill_params.cpuparams.n_threads = 1;
            prefill_params.cpuparams_batch.n_threads = 1;
            // The KV buffers follow the layer device (llama-kv-cache.cpp),
            // so with the lane resolved the prefill KV is HTP-resident —
            // host-addressable rpcmem, which keeps the prefill->decode
            // commit a host memcpy (MirrorAndCopy, llama-kv-commit.cpp).
            // The commit copies rows byte-for-byte and
            // llama_kv_commit_access::compatible refuses mismatched cache
            // types, so BOTH contexts must carry the same type; the catalog
            // stays the authority elsewhere and the override is announced
            // once per load on KALSA_KV_TYPE_OVERRIDE.
            const ggml_type requested_k = prefill_params.cache_type_k;
            const ggml_type requested_v = prefill_params.cache_type_v;
            const auto lane_kv = decide_governor_lane_kv(
                requested_k, requested_v, prefill_params.flash_attn_type);
            if (lane_kv.kv_on_device) {
                prefill_params.cache_type_k = lane_kv.type_k;
                decode_params.cache_type_k = lane_kv.type_k;
                prefill_params.cache_type_v = lane_kv.type_v;
                decode_params.cache_type_v = lane_kv.type_v;
                if (prefill_params.cache_type_k != requested_k ||
                    prefill_params.cache_type_v != requested_v) {
                    LOG_INFO(
                        "KALSA_KV_TYPE_OVERRIDE {requested_k:\"%s\", requested_v:\"%s\", "
                        "effective_k:\"%s\", effective_v:\"%s\", reason:\"htp-prefill-kv\"}",
                        ggml_type_name(requested_k), ggml_type_name(requested_v),
                        ggml_type_name(prefill_params.cache_type_k),
                        ggml_type_name(prefill_params.cache_type_v));
                }
            } else {
                // Flash attention explicitly off: device KV is off the
                // table (decide_governor_lane_kv), so the load reverts to
                // host-pinned KV with the caller's types.
                prefill_params.no_kv_offload = true;
                LOG_INFO(
                    "KALSA_KV_HOST_PIN {requested_k:\"%s\", requested_v:\"%s\", "
                    "reason:\"flash-attn-off\"}",
                    ggml_type_name(requested_k), ggml_type_name(requested_v));
            }
        }
#if defined(__ANDROID__)
    } else {
        // Lane off: what the CPU/OpenCL-only build did, byte for byte — HTP
        // joins no list, explicit or swept. An empty result stays an
        // explicit empty list ([nullptr]): the engine assigns zero GPU
        // layers instead of running its implicit sweep, which would include
        // HTP. Nothing in the binding or the app ever sets split_mode, so
        // the engine's tensor-split ">= 1 devices" refusal cannot fire from
        // this list; if a future caller did set it, this list fails that
        // load loudly (governor fallback) rather than offload to HTP.
        const auto lane_off = lane_off_devices(owner.params.devices);
        prefill_params.devices = lane_off;
        prefill_params.devices.push_back(nullptr);
        decode_params.devices = lane_off;
        decode_params.devices.push_back(nullptr);
#endif
    }
    // What this governor load built, for the getGovernorStats surface: the
    // cache types BOTH contexts run with (they only diverge from the
    // caller's params through the lane upgrade above) and where the prefill
    // KV buffers live — host when no_kv_offload is set (caller-requested,
    // or the flash-attn-off pin above), the layer device (HTP0 with the
    // lane resolved) otherwise.
    owner.setGovernorKvCache(
        ggml_type_name(prefill_params.cache_type_k),
        ggml_type_name(prefill_params.cache_type_v),
        prefill_params.no_kv_offload ? "host" : "device");
    prefill_params.n_gpu_layers = 99;
    decode_params.n_gpu_layers = 0;
    prefill_params.n_parallel = 1;
    decode_params.n_parallel = 1;
    // The CPU repack copy exists only where repackable tensors are placed on
    // CPU: that is the decode model (n_gpu_layers=0) — the load log shows its
    // CPU_REPACK ≈ W beside the prefill model's OpenCL copy, and no CPU_REPACK
    // line for prefill (its tensors are on OpenCL; its CPU residuals carry no
    // repack traits). The engine honours no_extra_bufts (use_extra_bufts =
    // !no_extra_bufts, common.cpp) — the same flag bmoe_stream raises. Only
    // the 8 GB S23 shape needs the second copy dropped (the lane does not fit
    // with repack); decode then pays the ~1.4x CPU decode cost the S23 repack
    // A/B measured, in a lane where GPU prefill covers the 2.6x. Everywhere
    // else repack stays on (governor.decode_repack, default true).
    if (!load_options.decode_repack) {
        decode_params.no_extra_bufts = true;
    }

    const bool profile_valid = governor_thermo_profile_is_valid(governor_thermo);
    auto cleanup = [&owner]() {
        teardown_governor_load(owner);
    };
    auto fail = [&](const char * stage, int models_loaded,
                    const std::string & reason) {
        log_governor_fallback(stage, models_loaded, reason,
                              governor_params.gpu_fit, profile_valid);
        cleanup();
        return false;
    };

    try {
        owner.governor_prefill_init = common_init_from_params(prefill_params, true);
    } catch (const std::exception & error) {
        return fail("accelerator_model_load", 0, error.what());
    }
    if (owner.governor_prefill_init == nullptr || owner.governor_prefill_init->model() == nullptr) {
        return fail("accelerator_model_load", 0, "accelerator model load failed");
    }

    try {
        owner.governor_decode_init = common_init_from_params(decode_params, true);
    } catch (const std::exception & error) {
        return fail("cpu_model_load", 1, error.what());
    }
    if (owner.governor_decode_init == nullptr || owner.governor_decode_init->model() == nullptr) {
        return fail("cpu_model_load", 1, "CPU model load failed");
    }

    bool runtime_profile_valid = profile_valid;
    try {
        const auto prefill_context_params = common_context_params_to_llama(prefill_params);
        const auto decode_context_params = common_context_params_to_llama(decode_params);
        owner.governor = std::make_unique<rn_governor>(
            owner.governor_prefill_init->model(), owner.governor_decode_init->model(),
            prefill_context_params, decode_context_params, policy_params);
        if (!owner.governor->set_thermo_profile(governor_thermo)) {
            runtime_profile_valid = false;
            throw std::runtime_error("governor: thermo profile invalid");
        }
    } catch (const std::exception & error) {
        const std::string reason = error.what();
        const bool route_rejected = reason.find("route") != std::string::npos ||
                                    reason.find("Route") != std::string::npos;
        log_governor_fallback(route_rejected ? "route_reject" : "init", 2, reason,
                              governor_params.gpu_fit, runtime_profile_valid);
        // CPU retry belongs to LlamaService/S4. Native owns only cleanup and
        // returns failure so that retry can recreate the single-context path.
        cleanup();
        return false;
    }

    owner.model = owner.governor_prefill_init->model();
    owner.ctx = owner.governor->prefill_ctx();
    owner.params.n_gpu_layers = 99;
    if (owner.model == nullptr || owner.ctx == nullptr) {
        log_governor_fallback("init", 2,
                              "governor initialized without an active context",
                              governor_params.gpu_fit, profile_valid);
        cleanup();
        return false;
    }
    return true;
}

} // namespace

size_t backend_dev_count() {
    return ggml_backend_dev_count();
}

ggml_backend_dev_t backend_dev_get(size_t index) {
    return ggml_backend_dev_get(index);
}

const char * backend_dev_name(ggml_backend_dev_t dev) {
    return dev ? ggml_backend_dev_name(dev) : nullptr;
}

enum ggml_backend_dev_type backend_dev_type(ggml_backend_dev_t dev) {
    return ggml_backend_dev_type(dev);
}

const char * backend_dev_reg_name(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg ? ggml_backend_reg_name(reg) : nullptr;
}

std::string backend_dev_device_id(ggml_backend_dev_t dev) {
    if (!dev) {
        return "";
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    return props.device_id ? props.device_id : "";
}

ggml_backend_buffer_type_t backend_cpu_buffer_type() {
    return ggml_backend_cpu_buffer_type();
}

bool read_gguf_file_info(const std::string & path, gguf_file_info & info) {
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };
    struct gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        return false;
    }
    info.version = gguf_get_version(ctx);
    info.alignment = gguf_get_alignment(ctx);
    info.data_offset = gguf_get_data_offset(ctx);
    const int n_kv = gguf_get_n_kv(ctx);
    info.kv.clear();
    info.kv.reserve(n_kv);
    for (int i = 0; i < n_kv; ++i) {
        info.kv.emplace_back(gguf_get_key(ctx, i), gguf_kv_to_str(ctx, i));
    }
    gguf_free(ctx);
    return true;
}

json get_backend_devices_info() {
    return backend_devices_info();
}

// ggml only prints its fatal assertion message to stderr, which logcat does not capture, so an
// Android crash report would carry just a SIGABRT backtrace. Forward the message to the platform
// log (and to the tombstone) before the process aborts.
static void ggml_abort_log_callback(const char *message) {
    log("ERROR", "ggml_abort", 0, "%s", message);
#if defined(__ANDROID__)
    android_set_abort_message(message);
#else
    fprintf(stderr, "%s\n", message);
#endif
}

void install_ggml_abort_handler() {
    ggml_set_abort_callback(ggml_abort_log_callback);
}

static const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

ggml_type kv_cache_type_from_str(const std::string & s) {
    if (s.empty()) {
        return GGML_TYPE_F16; // Default to F16 if empty string
    }

    for (const auto & type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }

    // Return default type instead of throwing to avoid crashes
    return GGML_TYPE_F16;
}

enum llama_flash_attn_type flash_attn_type_from_str(const std::string & s) {
    if (s == "on") {
        return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (s == "off") {
        return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    return LLAMA_FLASH_ATTN_TYPE_AUTO;
}


void log(const char *level, const char *function, int line,
                       const char *format, ...)
{
    va_list args;
    #if defined(__ANDROID__)
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "%s:%d %s", function, line, format);

        va_start(args, format);
        android_LogPriority priority;
        if (strcmp(level, "ERROR") == 0) {
            priority = ANDROID_LOG_ERROR;
        } else if (strcmp(level, "WARNING") == 0) {
            priority = ANDROID_LOG_WARN;
        } else if (strcmp(level, "INFO") == 0) {
            priority = ANDROID_LOG_INFO;
        } else {
            priority = ANDROID_LOG_DEBUG;
        }
        __android_log_vprint(priority, "RNLlama", prefix, args);
        va_end(args);
    #else
        printf("[%s] %s:%d ", level, function, line);
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    #endif
}

std::string token_piece_to_output_string(const std::string & piece)
{
    std::string out = piece;
    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }
    else if (!utf8_is_well_formed(out))
    {
        out = utf8_sanitize(out);
    }
    return out;
}

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    return token_piece_to_output_string(token == -1 ? "" : common_token_to_piece(ctx, token));
}

std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end)
{
    std::string ret;
    for (auto it = begin; it != end; ++it)
    {
        ret += common_token_to_piece(ctx, *it);
    }
    return ret;
}

bool model_uses_mrope(const llama_model *model) {
    if (model == nullptr) {
        return false;
    }
    const enum llama_rope_type rope = llama_model_rope_type(model);
    return rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE;
}

static std::string state_meta_path(const std::string &state_path) {
    return state_path + ".meta";
}

void write_state_meta(const std::string &state_path, const std::vector<std::string> &bitmap_hashes) {
    const std::string meta_path = state_meta_path(state_path);
    if (bitmap_hashes.empty()) {
        std::remove(meta_path.c_str());
        return;
    }
    // Write-through-temp + rename so an interrupted write can never leave a
    // stale sidecar describing a different state file
    const std::string tmp_path = meta_path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::trunc);
        if (!out.is_open()) {
            LOG_WARNING("Failed to write state metadata: %s", tmp_path.c_str());
            std::remove(meta_path.c_str()); // fail closed: no metadata beats wrong metadata
            return;
        }
        out << "rnllama-state-meta v1\n" << bitmap_hashes.size() << "\n";
        for (const auto &hash : bitmap_hashes) {
            out << hash << "\n";
        }
        out.flush();
        if (!out.good()) {
            LOG_WARNING("Failed to write state metadata: %s", tmp_path.c_str());
            out.close();
            std::remove(tmp_path.c_str());
            std::remove(meta_path.c_str());
            return;
        }
    }
    if (std::rename(tmp_path.c_str(), meta_path.c_str()) != 0) {
        LOG_WARNING("Failed to replace state metadata: %s", meta_path.c_str());
        std::remove(tmp_path.c_str());
        std::remove(meta_path.c_str());
    }
}

std::vector<std::string> read_state_meta(const std::string &state_path) {
    std::vector<std::string> hashes;
    std::ifstream in(state_meta_path(state_path));
    if (!in.is_open()) {
        return hashes;
    }
    std::string line;
    if (!std::getline(in, line) || line != "rnllama-state-meta v1") {
        return hashes;
    }
    size_t count = 0;
    if (!std::getline(in, line)) {
        return hashes;
    }
    try {
        count = std::stoul(line);
    } catch (const std::exception &) {
        return hashes;
    }
    for (size_t i = 0; i < count && std::getline(in, line); i++) {
        hashes.push_back(line);
    }
    if (hashes.size() != count) {
        hashes.clear(); // truncated sidecar - treat as absent
    }
    return hashes;
}

// Well-formed UTF-8 lead bytes and the range of their first continuation
// byte (Unicode 15.0, table 3-7). The restricted first-continuation ranges
// exclude overlong encodings, UTF-16 surrogates and values > U+10FFFF.
// Returns the sequence length, or 0 for anything that cannot start a
// multi-byte sequence (ASCII, continuation bytes, invalid leads).
static size_t utf8_lead_info(unsigned char c, unsigned char & lo, unsigned char & hi)
{
    lo = 0x80; hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) { return 2; }
    if (c == 0xE0)              { lo = 0xA0; return 3; }
    if (c >= 0xE1 && c <= 0xEC) { return 3; }
    if (c == 0xED)              { hi = 0x9F; return 3; }
    if (c >= 0xEE && c <= 0xEF) { return 3; }
    if (c == 0xF0)              { lo = 0x90; return 4; }
    if (c >= 0xF1 && c <= 0xF3) { return 4; }
    if (c == 0xF4)              { hi = 0x8F; return 4; }
    return 0;
}

size_t utf8_incomplete_suffix_length(const std::string & text)
{
    const size_t n = text.size();

    // the lead of an incomplete sequence can be at most 3 bytes from the end
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        const unsigned char c = text[n - back];
        if ((c & 0xC0) == 0x80) {
            continue;
        }
        unsigned char lo, hi;
        const size_t seq_len = utf8_lead_info(c, lo, hi);
        if (seq_len == 0 || back >= seq_len) {
            return 0; // not a lead, or the sequence is already complete/dead
        }
        // the continuations seen so far must be in range for this lead
        for (size_t k = 1; k < back; ++k) {
            const unsigned char cc = text[n - back + k];
            if (cc < (k == 1 ? lo : 0x80) || cc > (k == 1 ? hi : 0xBF)) {
                return 0;
            }
        }
        return back;
    }

    return 0; // lone continuation bytes at the end are dead, not incomplete
}

bool utf8_is_well_formed(const std::string & text)
{
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = text[i];
        if (c < 0x80) {
            i++;
            continue;
        }
        unsigned char lo, hi;
        const size_t seq_len = utf8_lead_info(c, lo, hi);
        if (seq_len == 0 || i + seq_len > n) {
            return false;
        }
        for (size_t k = 1; k < seq_len; ++k) {
            const unsigned char cc = text[i + k];
            if (cc < (k == 1 ? lo : 0x80) || cc > (k == 1 ? hi : 0xBF)) {
                return false;
            }
        }
        i += seq_len;
    }
    return true;
}

std::string utf8_sanitize(const std::string & text)
{
    std::string out;
    out.reserve(text.size());

    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = text[i];
        if (c < 0x80) {
            out += (char) c;
            i++;
            continue;
        }

        unsigned char lo, hi;
        const size_t seq_len = utf8_lead_info(c, lo, hi);
        // seq_len == 0: stray continuation byte or invalid lead (0xC0/0xC1/0xF5..0xFF)

        if (seq_len == 0) {
            out += "\xEF\xBF\xBD"; // U+FFFD replacement character
            i++;
            continue;
        }

        // consume the maximal valid subpart of the sequence
        size_t k = 1;
        while (k < seq_len && i + k < n) {
            const unsigned char cc = text[i + k];
            if (cc < (k == 1 ? lo : 0x80) || cc > (k == 1 ? hi : 0xBF)) {
                break;
            }
            k++;
        }

        if (k == seq_len) {
            out.append(text, i, seq_len);
        } else {
            out += "\xEF\xBF\xBD";
        }
        i += k;
    }

    return out;
}

std::string utf8_stream_gate::feed(const std::string & piece)
{
    pending += piece;

    const size_t hold = utf8_incomplete_suffix_length(pending);
    const size_t ready_len = pending.size() - hold;
    std::string ready = utf8_sanitize(pending.substr(0, ready_len));
    pending.erase(0, ready_len);
    return ready;
}

std::string utf8_stream_gate::finish()
{
    if (pending.empty()) {
        return {};
    }
    std::string tail = utf8_sanitize(pending);
    pending.clear();
    return tail;
}


void llama_rn_context::cleanupThreadpools() {
    if (ctx != nullptr && (threadpool != nullptr || threadpool_batch != nullptr)) {
        llama_detach_threadpool(ctx);
    }

    if (threadpool_batch != nullptr) {
        ggml_threadpool_free(threadpool_batch);
        threadpool_batch = nullptr;
    }

    if (threadpool != nullptr) {
        ggml_threadpool_free(threadpool);
        threadpool = nullptr;
    }
}

bool llama_rn_context::attachThreadpoolsIfAvailable() {
    if (governor) {
        return false;
    }
    if (ctx == nullptr) {
        return false;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        LOG_WARNING("No CPU backend available; skipping threadpool attachment");
        return false;
    }

    cleanupThreadpools();

    ggml_threadpool_params tpp =
        ggml_threadpool_params_from_cpu_params(params.cpuparams);
    ggml_threadpool_params tpp_batch =
        ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);

    if (tpp.n_threads <= 0) {
        LOG_WARNING("Skipping threadpool attachment (n_threads = %d)", tpp.n_threads);
        return false;
    }

    bool need_batch_pool =
        !ggml_threadpool_params_match(&tpp, &tpp_batch) && tpp_batch.n_threads > 0;

    ggml_threadpool *new_batch = nullptr;
    if (need_batch_pool) {
        new_batch = ggml_threadpool_new(&tpp_batch);
        if (new_batch == nullptr) {
            LOG_WARNING("Failed to create batch threadpool (n_threads=%d)", tpp_batch.n_threads);
            return false;
        }
        tpp.paused = true;
    }

    ggml_threadpool *new_threadpool = ggml_threadpool_new(&tpp);
    if (new_threadpool == nullptr) {
        LOG_WARNING("Failed to create threadpool (n_threads=%d)", tpp.n_threads);
        if (new_batch != nullptr) {
            ggml_threadpool_free(new_batch);
        }
        return false;
    }

    llama_attach_threadpool(ctx, new_threadpool, new_batch);
    threadpool = new_threadpool;
    threadpool_batch = new_batch;
    LOG_INFO("Attached ggml threadpool (n_threads=%d, n_threads_batch=%d)",
             tpp.n_threads,
             threadpool_batch ? tpp_batch.n_threads : tpp.n_threads);
    return true;
}

llama_rn_context::llama_rn_context() = default;

llama_rn_context::~llama_rn_context() {
    if (moe_stream) moe_stream->shutdown();

    // Disable parallel mode first (cleans up slot_manager)
    disableParallelMode();

    if (governor) {
        lora.clear();
        clear_init_lora_ownership(llama_init);
        owned_lora.clear();
        // Explicit: release_opencl_imports must run between the governor's
        // leg contexts freeing and the model freeing, which member
        // destruction cannot order.
        teardown_governor_load(*this);
    } else {
        removeLoraAdapters();
    }
    cleanupThreadpools();

    if (completion != nullptr) {
        delete completion;
        completion = nullptr;
    }

    releaseMultimodal();
    releaseVocoder();
}

llama_context * llama_rn_context::active_ctx() const {
    return governor ? governor->active_ctx() : ctx;
}

int32_t llama_rn_context::decode(llama_batch batch) {
    governor_pause_ = nullptr;
    if (!governor) {
        return llama_decode(ctx, batch);
    }

    const bool was_failed = governor->failed();
    const int32_t result = governor->decode(batch);
    if (governor_decode_failed(result, governor->engine_failed()) && !was_failed) {
        const std::string & reason = governor->failure_reason();
        // Only a compute failure of a batch the HTP device ran kills the
        // lane — the HTP-routed prefill, or an NPU decode hop (a 1-token
        // batch on ctx_prefill); a KV-commit (host/OpenCL) or CPU-routed
        // failure must not.
        const llama_governor_stats failure_stats = governor->stats();
        if (htp_runtime_failure(
                batch.n_tokens, failure_stats.prefill_engine,
                failure_stats.decode_engine, result, reason.c_str())) {
            note_htp_runtime_fallback();
        }
        const bool route_rejected = reason.find("route Reject") != std::string::npos;
        // This line is the one surface that fires at failure time: the
        // end-of-turn KALSA_GOVERNOR telemetry is skipped on the throwing
        // turn and the retry's reload suppresses the plan line, so without
        // the field here the recorded constant is invisible to the lab.
        // Snapshot under the mutex: the recorder may be writing this very
        // reason on another thread's failed decode.
        const std::string npu_fallback = governorNpuFallback();
        LOG_ERROR(
            "KALSA_GOVERNOR_FALLBACK {stage:\"%s\", models_loaded:2, reason:\"%s\", npu_fallback:\"%s\", gpu_fit:%d, profile_valid:%d}",
            route_rejected ? "route_reject" : "decode",
            reason.c_str(), npu_fallback.c_str(),
            (int) governor->gpu_fit(), (int) governor->profile_valid());
    }
    if (result == -2 && !governor->failed()) {
        // Flow-control -2s share one rc (rn-governor.h); the labels use only
        // what the governor publishes in the stats of THIS decode. Prefill:
        // Unknown state is the profile wait; last_router_rule == 0 means the
        // admission returned Wait without naming an engine — its heat refusal
        // (the warn-line ceiling, or a non-finite reading), and the only
        // signal here that the governor itself says thermal (every admit/chunk
        // return stamps rule {2,3,9} in llama-governor-policy.cpp). A -2 that
        // follows an admitted engine is a partition refusal the binding cannot
        // name — the count<=1 guard is unreachable while the policy admits
        // row+1 whole (6581c7a7b), yet logged, not asserted, there — so it is
        // "unexplained", never "thermal": no host may cool-and-resume a retry
        // that would repeat the same partition. Decode: reload-required when
        // the stats say so, otherwise the profile wait.
        const llama_governor_stats stats = governor->stats();
        if (batch.n_tokens > 1) {
            if (stats.thermal_state == llama_governor_thermal_state::Unknown) {
                governor_pause_ = "profile";
            } else if (stats.last_router_rule == 0) {
                governor_pause_ = "thermal";
            } else {
                governor_pause_ = "unexplained";
            }
        } else {
            governor_pause_ = stats.decode_requires_reload ? "reload" : "profile";
        }
    }
    return result;
}

int32_t llama_rn_context::decode(llama_batch_ext * batch) {
    // The ext form arrives only from the speculative/MTP paths (common_batch),
    // which governor mode refuses at load. Raw rc: the caller throws its own
    // error with the batch context in it.
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
}

bool llama_rn_context::hasGovernor() const {
    return governor != nullptr;
}

bool llama_rn_context::governorFailed() const {
    return governor != nullptr && governor->failed();
}

std::string llama_rn_context::governorFailureReason() const {
    return governor == nullptr ? "" : governor->failure_reason();
}

void llama_rn_context::resetGovernorPrefillStats() {
    if (governor != nullptr) {
        governor->reset_prefill_stats();
    }
}

bool llama_rn_context::setThermoProfile(const llama_governor_thermo_profile & profile) {
    return governor != nullptr && governor->set_thermo_profile(profile);
}

bool llama_rn_context::setPrefillOverride(int mode) {
    return governor != nullptr && governor->set_prefill_override(mode);
}

void llama_rn_context::setGovernorNpuDevice(const char * device) {
    std::lock_guard<std::mutex> lock(npu_fallback_mutex_);
    governor_npu_device_ = device == nullptr ? "" : device;
}

std::string llama_rn_context::governorNpuDevice() const {
    std::lock_guard<std::mutex> lock(npu_fallback_mutex_);
    return governor_npu_device_;
}

void llama_rn_context::setGovernorNpuFallback(const char * reason) {
    std::lock_guard<std::mutex> lock(npu_fallback_mutex_);
    governor_npu_fallback_ = reason == nullptr ? "" : reason;
}

std::string llama_rn_context::governorNpuFallback() const {
    std::lock_guard<std::mutex> lock(npu_fallback_mutex_);
    return governor_npu_fallback_;
}

void llama_rn_context::setGovernorKvCache(
        const char * type_k, const char * type_v, const char * prefill_placement) {
    std::lock_guard<std::mutex> lock(governor_kv_mutex_);
    governor_kv_type_k_ = type_k == nullptr ? "" : type_k;
    governor_kv_type_v_ = type_v == nullptr ? "" : type_v;
    governor_kv_placement_ = prefill_placement == nullptr ? "" : prefill_placement;
}

std::string llama_rn_context::governorKvTypeK() const {
    std::lock_guard<std::mutex> lock(governor_kv_mutex_);
    return governor_kv_type_k_;
}

std::string llama_rn_context::governorKvTypeV() const {
    std::lock_guard<std::mutex> lock(governor_kv_mutex_);
    return governor_kv_type_v_;
}

std::string llama_rn_context::governorKvPlacement() const {
    std::lock_guard<std::mutex> lock(governor_kv_mutex_);
    return governor_kv_placement_;
}

void llama_rn_context::note_htp_runtime_fallback() {
    // The decode wrapper has already attributed this failure to a batch the
    // HTP device ran (htp_runtime_failure); the recorder enforces
    // what it owns: the lane resolved (governorNpuDevice() — without one,
    // prefill already runs on the non-HTP path) and no reason recorded yet —
    // the first reason written in this context's life wins, and this
    // recorder never displaces the plan reason the load step published
    // (every load is a fresh context, so that publish is its first write).
    // HTP errors before the lane opened ride the env instead (Java
    // noteHtpFallback / the resolver's own reason).
    if (governorNpuDevice().empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(npu_fallback_mutex_);
        if (!governor_npu_fallback_.empty()) {
            return;
        }
        governor_npu_fallback_ = KALSA_HTP_RUNTIME_FALLBACK;
    }
    // The engine governor is sticky-failed, so no in-place retry exists on
    // this context pair; the failed turn fails cleanly to the caller (the
    // completion throws "Governor decode failed: ..."). The app's retry
    // reload runs CPU-only by design (governor off), so it never consults
    // this env; the env governs every LATER load that re-arms the governor:
    // decide_governor_prefill_device reads it fresh per load and keeps the
    // lane off even when the device resolves. The degrade sticks for the
    // process because Java only reaches its own KALSA_HTP_FALLBACK clear
    // (RNLlama.java extractHtpLibrariesFromAssets) inside loadNative, which
    // short-circuits on libsLoaded once set - nothing re-runs that clear.
    if (setenv("KALSA_HTP_FALLBACK", KALSA_HTP_RUNTIME_FALLBACK, /*overwrite=*/1) != 0) {
        // Plain message on purpose: the FALLBACK marker is a load-retry
        // signal for the app's log scan, and a failed setenv is not one.
        // The recorded field still degrades the stats surface.
        LOG_ERROR("kalsa: setenv KALSA_HTP_FALLBACK failed; the next load of this process would keep HTP");
    }
}

llama_governor_stats llama_rn_context::governorStats() const {
    return governor == nullptr ? llama_governor_stats{} : governor->stats();
}

bool llama_rn_context::loadModel(
    common_params &params_,
    const llama_governor_params * governor_params,
    const llama_governor_thermo_profile * governor_thermo,
    const governor_load_options & load_options)
{
    const bool governor_enabled = governor_params != nullptr;
    // Do not return after the two-model load. Both modes converge here so the
    // context has templates, n_ctx, adapter metadata, n_seq logging, completion,
    // and the same ctx_shift setup before the first completion.
    if (governor_enabled) {
        if (governor_thermo == nullptr || !governor_thermo_profile_is_valid(*governor_thermo)) {
            throw std::runtime_error("governor: thermo profile invalid");
        }
        if (params_.n_parallel > 1) {
            throw std::runtime_error("Governor mode supports sequence 0 only; parallel slots are disabled");
        }
        if (has_speculative_mode(params_)) {
            log_governor_fallback(
                "params", 0,
                "Governor mode does not support speculative or MTP decoding",
                governor_params->gpu_fit,
                governor_thermo != nullptr &&
                    governor_thermo_profile_is_valid(*governor_thermo));
            throw std::runtime_error("Governor mode does not support speculative or MTP decoding");
        }
        if (params_.kalsa_moe.enabled) {
            throw std::runtime_error("Governor mode does not support moe_stream");
        }
        if (!params_.lora_adapters.empty()) {
            throw std::runtime_error("Governor mode does not support LoRA adapters");
        }
    }

    if (!governor) {
        removeLoraAdapters();
    }
    draft_model.reset();
    params = params_;

    // common_init_from_params() now creates its threadpools directly, so CPU
    // defaults must be resolved just as the upstream CLI parser resolves them.
    // In particular, n_threads_batch=-1 means "inherit n_threads"; passing -1
    // to ggml_threadpool_new underflows its worker count.
    postprocess_cpu_params(params.cpuparams);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    // Ensure n_parallel is set to a reasonable default for parallel decoding support
    // This sets n_seq_max in the context, which cannot be changed later
    if (governor_enabled) {
        params.n_parallel = 1;
    } else if (params.n_parallel < 1) {
        params.n_parallel = 8; // Default to support up to 8 parallel slots
        LOG_INFO("Setting n_parallel to default: %d (enables up to %d parallel slots)", params.n_parallel, params.n_parallel);
    } else {
        LOG_INFO("Using n_parallel: %d (enables up to %d parallel slots)", params.n_parallel, params.n_parallel);
    }

    // Source buffers back the expert tensors: shut the streamer, then drop the old
    // context, then the hook. Inverse of this order dangling-cb_eval or UAF on reload.
    if (moe_stream) moe_stream->shutdown();
    moe_stream.reset();
    teardown_governor_load(*this);

    // The CLI layer this fork does not ship is what resolves the cpu params
    // (kalsallama common/arg.cpp:891-892); nothing else does, and this pin's
    // common_threadpools::init builds one ggml threadpool per cpuparams pair
    // from the values as given. It drops two things:
    //  - an unresolved -1 (the cpuparams_batch default, meaning "use cpuparams"
    //    per common.cpp:1746) reached ggml raw: ggml-cpu.c:4204 computed
    //    workers_size = 544 * -1, ggml_aligned_malloc failed and the memset
    //    at :4207 of (size_t)-544 faulted on the first model load
    //    (e2e run 35307242869, batch pool first);
    //  - an explicit 0 survives postprocess_cpu_params, which rescues only
    //    n_threads < 0 (common.cpp:293), and then faults at ggml-cpu.c:4238
    //    (workers[0] on the NULL array of a zero-worker pool).
    // Upstream normalizes the zero where the CLI parses the flag -- arg.cpp:
    // 1522-1525 for -t, :1532-1535 for -tb, "<= 0 ->
    // std::thread::hardware_concurrency()", no warning. We deviate on the
    // source of the default, on purpose: hardware_concurrency() may return 0
    // ("not computable or well defined"), which is exactly the corner that
    // makes this reachable -- a device reporting 0 cores -- and a CLI that gets
    // that wrong prints an error and exits, while an app takes a SIGSEGV in
    // front of a user. So the default is the 0-safe accessor
    // postprocess_cpu_params itself uses for its < 0 case (common.cpp:298),
    // whose Android path counts /sys cores and whose last resort is "... : 4"
    // (common.cpp:148). -1 semantics stay untouched: resolving that is
    // postprocess_cpu_params's job, it copies the role model.
    // Normalizing here and not at the JS parse site: loadModel is the single
    // choke point every load goes through (main and both governor copies),
    // whatever built the params.
    const int default_threads = (int) common_cpu_get_num_math();
    if (params.cpuparams.n_threads       == 0) { params.cpuparams.n_threads       = default_threads; }
    if (params.cpuparams_batch.n_threads == 0) { params.cpuparams_batch.n_threads = default_threads; }
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    if (governor_enabled) {
        if (!load_governor_models(*this, *governor_params, *governor_thermo, load_options)) {
            return false;
        }
    } else {
        moe_stream = std::make_unique<kalsa::MoeStream>();
        {
            std::string moe_err;
            if (!moe_stream->arm(params, moe_err)) {
                LOG_INFO("kalsa moe stream: %s", moe_err.c_str());
            }
        }

        llama_init = common_init_from_params(params);
        model = llama_init != nullptr ? llama_init->model() : nullptr;
        ctx = llama_init != nullptr ? llama_init->context() : nullptr;

        // common_init_from_params() can fail after loading the model but before
        // constructing the context, so both pointers must be validated here.
        if (model == nullptr || ctx == nullptr) {
            if (model == nullptr) {
                LOG_ERROR("unable to load model: %s", params_.model.path.c_str());
            } else {
                LOG_ERROR("unable to initialize context for model: %s", params_.model.path.c_str());
            }
            return false;
        }

        if (moe_stream->armed()) {
            std::string moe_err;
            if (!moe_stream->bind(ctx, moe_err)) {
                LOG_INFO("kalsa moe stream: %s", moe_err.c_str());
            }
        }

        // Kalsa patch: DFlash needs the standalone draft model too — upstream gated
        // the loader on MTP only, so a pure ["draft-dflash"] config silently ran
        // without a draft (common_speculative_init requires ctx_dft for DFLASH).
        if (params.speculative.has_dft() &&
            (has_speculative_type(params.speculative, COMMON_SPECULATIVE_TYPE_DRAFT_MTP) ||
             has_speculative_type(params.speculative, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH))) {
            const auto & draft_params = params.speculative.draft;
            common_params params_dft = params;
            params_dft.devices = draft_params.devices;
            params_dft.model = draft_params.mparams;
            params_dft.n_gpu_layers = draft_params.n_gpu_layers;
            params_dft.cache_type_k = draft_params.cache_type_k;
            params_dft.cache_type_v = draft_params.cache_type_v;
            params_dft.tensor_buft_overrides = draft_params.tensor_buft_overrides;

            if (draft_params.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads = draft_params.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = draft_params.cpuparams_batch.n_threads;
            }

            auto mparams_dft = common_model_params_to_llama(params_dft);
            LOG_INFO("Loading MTP draft model: %s", params_dft.model.path.c_str());
            draft_model.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (draft_model == nullptr) {
                LOG_ERROR("unable to load MTP draft model: %s", params_dft.model.path.c_str());
                return false;
            }
        }
    }

    if (governor_enabled) {
        // Both governor forms carry the same GGUF; the two-model inits are
        // the historical source and keep precedence.
        llama_model * governor_model = governor_decode_init != nullptr
            ? governor_decode_init->model()
            : (governor_prefill_init != nullptr ? governor_prefill_init->model()
                                                : governor_onecopy_init->model());
        templates = common_chat_templates_init(governor_model, params.chat_template);
        n_ctx = llama_n_ctx(active_ctx());
    } else {
        templates = common_chat_templates_init(model, params.chat_template);
        n_ctx = llama_n_ctx(ctx);
    }

    // Init-time adapters are already loaded and applied by common_init_from_params().
    // Mirror the resulting adapter metadata so getLoadedLoraAdapters() reflects reality
    // without forcing a second load/apply pass in the JSI layer.
    lora = params.lora_adapters;
    owned_lora.clear();
    for (auto &la : lora) {
        populate_lora_metadata(la);
    }

    // Log the actual n_seq_max that was set
    uint32_t n_seq_max = governor_enabled
        ? llama_n_seq_max(active_ctx())
        : llama_n_seq_max(ctx);
    LOG_INFO("Context initialized with n_seq_max = %u", n_seq_max);

    // Initialize completion context
    if (completion != nullptr) {
        delete completion;
    }
    completion = new llama_rn_context_completion(this);

    // Initialize context shift flag
    LOG_INFO("ctx_shift: %s", params.ctx_shift ? "enabled" : "disabled");

    // We can uncomment for debugging or after this fix: https://github.com/ggerganov/llama.cpp/pull/11101
    // LOG_INFO("%s\n", common_params_get_system_info(params).c_str());

    return true;
}

bool llama_rn_context::hasDraftModel() const {
    return draft_model != nullptr;
}

llama_model * llama_rn_context::getMTPDraftModel() const {
    return draft_model != nullptr ? draft_model.get() : model;
}

llama_context * llama_rn_context::createMTPDraftContext(const common_params &params_for_context) const {
    if (hasGovernor()) {
        throw std::runtime_error(
            "Governor mode does not support MTP or speculative decoding");
    }
    llama_model * model_dft = getMTPDraftModel();
    if (model_dft == nullptr || ctx == nullptr) {
        return nullptr;
    }

    auto cparams = common_context_params_to_llama(params_for_context);
    // Kalsa patch: a MTP-type context on a model without nextn layers returns
    // nullptr (llama-context.cpp init guard). Standalone drafts (DFlash) may
    // lack them — fall back to a default context for those.
    cparams.ctx_type = llama_model_n_layer_nextn(model_dft) > 0
        ? LLAMA_CONTEXT_TYPE_MTP
        : LLAMA_CONTEXT_TYPE_DEFAULT;
    cparams.n_rs_seq = 0;
    cparams.type_k = params_for_context.speculative.draft.cache_type_k;
    cparams.type_v = params_for_context.speculative.draft.cache_type_v;
    cparams.ctx_other = ctx;

    return llama_init_from_model(model_dft, cparams);
}


bool llama_rn_context::validateModelChatTemplate(bool use_jinja, const char *name) const {
    const char * tmpl = llama_model_chat_template(model, name);
    if (tmpl == nullptr) {
      return false;
    }
    return common_chat_verify_template(tmpl, use_jinja);
}

common_chat_params llama_rn_context::getFormattedChatWithJinja(
        const std::string& messages,
        const std::string& chat_template,
        const std::string& json_schema,
        const std::string& tools,
        const bool& parallel_tool_calls,
        const std::string& tool_choice,
        const bool& enable_thinking,
        const std::string& reasoning_format,
        const bool& add_generation_prompt,
        const std::string& now_str,
        const std::map<std::string, std::string>& chat_template_kwargs,
        const bool& force_pure_content
) const {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(messages));
    auto useTools = !tools.empty();
    if (useTools) {
        inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(tools));
    }
    inputs.parallel_tool_calls = parallel_tool_calls;
    if (!tool_choice.empty()) {
        inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
    }
    if (!json_schema.empty()) {
        inputs.json_schema = json_schema;
    }
    inputs.enable_thinking = enable_thinking;
    inputs.reasoning_format = common_reasoning_format_from_name(reasoning_format);
    inputs.add_generation_prompt = add_generation_prompt;

    // Handle now parameter - parse timestamp or use current time
    if (!now_str.empty()) {
        try {
            // Try to parse as timestamp (seconds since epoch)
            auto timestamp = std::stoll(now_str);
            inputs.now = std::chrono::system_clock::from_time_t(timestamp);
        } catch (...) {
            // If parsing fails, use current time
            inputs.now = std::chrono::system_clock::now();
        }
    }

    inputs.chat_template_kwargs = chat_template_kwargs;
    inputs.force_pure_content = force_pure_content;

    // If chat_template is provided, create new one and use it (probably slow)
    if (!chat_template.empty()) {
        auto tmps = common_chat_templates_init(model, chat_template);
        return common_chat_templates_apply(tmps.get(), inputs);
    } else {
        return common_chat_templates_apply(templates.get(), inputs);
    }
}

std::string llama_rn_context::getFormattedChat(
  const std::string &messages,
  const std::string &chat_template
) const {
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(messages));
    inputs.use_jinja = false;

    // If chat_template is provided, create new one and use it (probably slow)
    if (!chat_template.empty()) {
        auto tmps = common_chat_templates_init(model, chat_template);
        return common_chat_templates_apply(tmps.get(), inputs).prompt;
    } else {
        return common_chat_templates_apply(templates.get(), inputs).prompt;
    }
}

llama_rn_tokenize_result llama_rn_context::tokenize(const std::string &text, const std::vector<std::string> &media_paths) {
  if (media_paths.size() > 0) {
      if (!isMultimodalEnabled()) {
          throw std::runtime_error("Multimodal is not enabled but media paths are provided");
      }
      auto result = tokenizeWithMedia(mtmd_wrapper, text, media_paths);
      mtmd_input_chunks_free(result.chunks);
      llama_rn_tokenize_result tokenize_result;
      tokenize_result.tokens = result.tokens;
      tokenize_result.has_media = true;
      tokenize_result.bitmap_hashes = result.bitmap_hashes;
      tokenize_result.chunk_pos = result.chunk_pos;
      tokenize_result.chunk_pos_media = result.chunk_pos_media;
      return tokenize_result;
  }
  std::vector<llama_token> text_tokens;
  text_tokens = common_tokenize(active_ctx(), text, /* add_special= */ false, /* parse_special= */ true);
  llama_rn_tokenize_result tokenize_result;
  tokenize_result.tokens = text_tokens;
  tokenize_result.has_media = false;
  tokenize_result.bitmap_hashes = {};
  tokenize_result.chunk_pos = {};
  tokenize_result.chunk_pos_media = {};
  return tokenize_result;
}

void llama_rn_context::applyLoraAdapters(std::vector<common_adapter_lora_info> lora) {
    if (hasGovernor()) {
        throw std::runtime_error(
            "Governor mode does not support runtime LoRA adapters");
    }
    if (model == nullptr || ctx == nullptr) {
        throw std::runtime_error("Cannot apply LoRA adapters: context is not initialized");
    }

    std::vector<llama_adapter_lora_ptr> loaded_adapters;
    loaded_adapters.reserve(lora.size());

    for (auto &la : lora) {
        llama_adapter_lora_ptr adapter(llama_adapter_lora_init(model, la.path.c_str()));
        if (adapter == nullptr) {
            throw std::runtime_error(
                "Failed to apply LoRA adapter '" + la.path +
                "'. Check native logs for the detailed loader error. The adapter may not match the loaded base model."
            );
        }

        la.ptr = adapter.get();
        populate_lora_metadata(la);
        loaded_adapters.emplace_back(std::move(adapter));
    }

    common_set_adapter_lora(ctx, lora);
    this->lora = std::move(lora);
    clear_init_lora_ownership(llama_init);
    owned_lora = std::move(loaded_adapters);
}

void llama_rn_context::removeLoraAdapters() {
    if (hasGovernor()) {
        throw std::runtime_error(
            "Governor mode does not support runtime LoRA adapters");
    }
    if (ctx != nullptr) {
        std::vector<common_adapter_lora_info> empty_lora;
        common_set_adapter_lora(ctx, empty_lora); // apply empty list
    }

    this->lora.clear();
    clear_init_lora_ownership(llama_init);
    owned_lora.clear();
}

std::vector<common_adapter_lora_info> llama_rn_context::getLoadedLoraAdapters() {
    return this->lora;
}

bool llama_rn_context::initMultimodal(const std::string &mmproj_path, bool use_gpu, int image_min_tokens, int image_max_tokens) {
    if (hasGovernor()) {
        throw std::runtime_error(
            "Governor mode does not support multimodal initialization");
    }
    try {
        mtmd_wrapper = new llama_rn_context_mtmd(mmproj_path, use_gpu, model, ctx, params, has_multimodal, params, image_min_tokens, image_max_tokens);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("[DEBUG] Failed to initialize multimodal: %s", e.what());
        return false;
    }
}

bool llama_rn_context::isMultimodalEnabled() const {
    return mtmd_wrapper != nullptr && mtmd_wrapper->isEnabled(has_multimodal);
}

bool llama_rn_context::isMultimodalSupportVision() const {
    return isMultimodalEnabled() && mtmd_wrapper->supportVision();
}

bool llama_rn_context::isMultimodalSupportAudio() const {
    return isMultimodalEnabled() && mtmd_wrapper->supportAudio();
}

void llama_rn_context::releaseMultimodal() {
    if (mtmd_wrapper != nullptr) {
        delete mtmd_wrapper;
        mtmd_wrapper = nullptr;
        has_multimodal = false;
    }
}

std::vector<std::string> llama_rn_context::getMediaHashes() const {
    if (mtmd_wrapper == nullptr) {
        return {};
    }
    return mtmd_wrapper->bitmap_past_hashes;
}

void llama_rn_context::setMediaHashes(const std::vector<std::string> &hashes) {
    if (mtmd_wrapper != nullptr) {
        mtmd_wrapper->bitmap_past_hashes = hashes;
    }
}

bool llama_rn_context::initVocoder(const std::string &vocoder_model_path, int batch_size, bool use_gpu) {
    try {
        tts_wrapper = new llama_rn_context_tts(vocoder_model_path, batch_size, use_gpu);
        has_vocoder = true;
        return true;
    } catch (const std::exception& e) {
        has_vocoder = false;
        return false;
    }
}

bool llama_rn_context::isVocoderEnabled() const {
    return has_vocoder && tts_wrapper != nullptr;
}

void llama_rn_context::releaseVocoder() {
    if (tts_wrapper != nullptr) {
        delete tts_wrapper;
        tts_wrapper = nullptr;
    }
    has_vocoder = false;
}

// Enable parallel decoding mode
void llama_rn_context::enableParallelMode(int32_t n_parallel, int32_t n_batch) {
    if (governor != nullptr) {
        LOG_ERROR("Governor mode does not support parallel slots");
        throw std::runtime_error("Governor mode does not support parallel slots");
    }
    if (ctx == nullptr) {
        LOG_ERROR("Cannot enable parallel mode: context not initialized");
        throw std::runtime_error("Cannot enable parallel mode: context not initialized");
    }

    // Verify n_seq_max is sufficient for requested parallel slots
    uint32_t n_seq_max = llama_n_seq_max(ctx);
    if (n_seq_max < (uint32_t)n_parallel) {
        LOG_ERROR("Context n_seq_max (%u) is less than requested parallel slots (%d). Context was initialized with n_parallel=%d",
                  n_seq_max, n_parallel, params.n_parallel);
        LOG_ERROR("To use %d parallel slots, reinitialize the context with n_parallel >= %d", n_parallel, n_parallel);

        char error_msg[512];
        snprintf(error_msg, sizeof(error_msg),
                "Failed to enable parallel mode with %d slots. Context n_seq_max (%u) is less than requested. "
                "Context was initialized with n_parallel=%d. To use %d parallel slots, reinitialize the context with n_parallel >= %d",
                n_parallel, n_seq_max, params.n_parallel, n_parallel, n_parallel);
        throw std::runtime_error(error_msg);
    }

    // If parallel mode is already enabled, reconfigure it
    if (parallel_mode_enabled) {
        LOG_INFO("Reconfiguring parallel mode to %d slots, batch size %d", n_parallel, n_batch);
        // Clean up existing slot manager
        if (slot_manager != nullptr) {
            delete slot_manager;
            slot_manager = nullptr;
        }
    } else {
        LOG_INFO("Enabling parallel mode with %d slots, batch size %d (n_seq_max=%u)", n_parallel, n_batch, n_seq_max);
    }

    // Create slot manager
    slot_manager = new llama_rn_slot_manager(this);
    if (!slot_manager->init(n_parallel, n_batch, n_ctx)) {
        LOG_ERROR("Failed to initialize slot manager");
        delete slot_manager;
        slot_manager = nullptr;

        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg),
                "Failed to initialize slot manager with %d slots and batch size %d",
                n_parallel, n_batch);
        throw std::runtime_error(error_msg);
    }

    parallel_mode_enabled = true;

    LOG_INFO("Parallel mode enabled successfully with %d slots", n_parallel);
}

// Disable parallel decoding mode
void llama_rn_context::disableParallelMode() {
    if (!parallel_mode_enabled) {
        return;
    }

    LOG_INFO("Disabling parallel mode");

    if (slot_manager != nullptr) {
        delete slot_manager;
        slot_manager = nullptr;
    }

    parallel_mode_enabled = false;

    LOG_INFO("Parallel mode disabled");
}

/**
 * Empty the live KV. The one snapshot that survives is the stable prefix.
 *
 * Callers are all in JS (the only binding is llamaClearCache), and what JS is
 * clearing is a CONVERSATION: a window slide, a reconcile, a chat switch. The
 * system prompt it also destroys is byte-identical in whatever comes next, so
 * throwing that snapshot away only buys a cold prefill of it -- 1832 tokens,
 * 40 s on an S23. Keeping it is ~12 MB of RAM against that.
 *
 * JS is not told, and must not be. It decides WHAT prompt to send; the native
 * side decides how much of it still has to be computed, and only after
 * verifying the snapshot's tokens really are a prefix of that prompt
 * (findStateCheckpoint) and that the memory really came back where the label
 * says (recoverStateCheckpoint's pos_max + 1 == k). A JS flag asserting the
 * state of the native KV is the desync this project has hit three times.
 *
 * A model switch destroys the whole context, and with it the snapshot, so a
 * checkpoint can never outlive the tokenizer that produced its token ids.
 */
void llama_rn_context::clearCache(bool clear_data) {
    if (hasGovernor()) {
        governor->clear_cache(clear_data);
        if (completion != nullptr) {
            completion->embd.clear();
            completion->n_past = 0;
            completion->keepOnlyStablePrefixCheckpoint();
        }
        LOG_INFO("Governor caches cleared and completion state reset (clear_data=%s)",
                 clear_data ? "true" : "false");
        return;
    }
    if (ctx == nullptr) {
        LOG_WARNING("Cannot clear cache: context not initialized");
        return;
    }

    auto * kv = llama_get_memory(ctx);
    if (kv == nullptr) {
        LOG_WARNING("Cannot clear cache: memory not available");
        return;
    }

    llama_memory_clear(kv, clear_data);

    if (completion != nullptr) {
        completion->embd.clear();
        completion->n_past = 0;
        completion->keepOnlyStablePrefixCheckpoint();
        LOG_INFO("Cache cleared and completion state reset (clear_data=%s)", clear_data ? "true" : "false");
    } else {
        LOG_INFO("Cache cleared (clear_data=%s)", clear_data ? "true" : "false");
    }
}

}
