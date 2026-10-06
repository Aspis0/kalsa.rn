// The one-model governor's legs (FABLE Amendment F, F1): construction of the
// leg contexts over one shared llama_model and the context/side lookup both
// governor forms share. The decode/commit path itself lives in
// llama-governor.cpp and never branches on the form beyond these helpers.

#include "llama-governor.h"

#include "ggml-hexagon.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <stdexcept>
#include <string>

llama_governor::llama_governor(llama_model * model, const llama_governor_leg * leg_specs,
                               uint32_t n_legs, llama_governor_params governor_params)
    : policy_(governor_params), policy_enabled_(true) {
    init_forced_rotation(governor_params);
    if (!model) {
        throw std::runtime_error("one-model governor requires a model");
    }
    // F-01: three engines means at most three legs; duplicates or a null
    // spec list would corrupt the fixed three-entry side arrays downstream.
    if (!leg_specs && n_legs > 0) {
        throw std::runtime_error("one-model governor requires a leg array");
    }
    if (n_legs > 3) {
        throw std::runtime_error("one-model governor takes at most three legs (CPU, GPU, NPU)");
    }
    for (uint32_t i = 0; i < n_legs; ++i) {
        if (leg_specs[i].engine != llama_governor_engine::CPU &&
            leg_specs[i].engine != llama_governor_engine::GPU &&
            leg_specs[i].engine != llama_governor_engine::NPU) {
            throw std::runtime_error("GPU_COOLMODE is a decode state, not a governor leg");
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (leg_specs[j].engine == leg_specs[i].engine) {
                throw std::runtime_error("one-model governor takes each engine at most once");
            }
        }
        // F-04: Amendment F fences shared KV out of step 5 - a leg aliasing
        // another context's memory would break the per-leg watermark model.
        if (leg_specs[i].params.ctx_other != nullptr) {
            throw std::runtime_error("governor legs must have independent KV caches");
        }
        // F-02: context init rewrites the shared model's n_ctx_train on a
        // custom YaRN rescale (llama_init_from_model_with_legs), so every leg
        // must agree on the rope/YaRN params that feed that write - the
        // first leg is the reference.
        if (i > 0 &&
            (leg_specs[i].params.rope_scaling_type != leg_specs[0].params.rope_scaling_type ||
             leg_specs[i].params.rope_freq_scale   != leg_specs[0].params.rope_freq_scale ||
             leg_specs[i].params.yarn_orig_ctx     != leg_specs[0].params.yarn_orig_ctx)) {
            throw std::runtime_error("governor legs must share identical rope/YaRN params");
        }
    }

    // A throw anywhere below must not leak the legs built so far: the
    // destructor does not run for a partially constructed object. The
    // reserve also makes push_back non-throwing, so a leg cannot be orphaned
    // between creation and bookkeeping.
    legs.reserve(n_legs);
    try {
        for (uint32_t i = 0; i < n_legs; ++i) {
            const auto & spec = leg_specs[i];
            // The CPU leg is ALWAYS the empty list: passing the model's
            // devices would give it an accelerator backend set, and the
            // always-added CPU backend makes any CPU entry in the list
            // invalid anyway (llama_init_from_model_with_legs refuses it).
            ggml_backend_dev_t cpu_only[] = { nullptr };
            const ggml_backend_dev_t * devices = spec.devices;
            if (spec.engine == llama_governor_engine::CPU) {
                devices = cpu_only;
            } else if (!devices) {
                LLAMA_LOG_INFO("%s: leg %d is absent (no devices given)\n", __func__, (int) spec.engine);
                continue;
            } else if (devices[0]) {
                // a first device the model does not list makes the leg absent
                // (same membership rule llama_init_from_model_with_legs
                // enforces); an EMPTY list is a present, CPU-only leg - what
                // every leg is on a host build
                bool listed = false;
                for (const auto & model_dev : model->devices) {
                    if (model_dev.dev == devices[0]) {
                        listed = true;
                        break;
                    }
                }
                if (!listed) {
                    LLAMA_LOG_INFO("%s: leg %d is absent (%s is not one of the model's devices)\n",
                                   __func__, (int) spec.engine, ggml_backend_dev_name(devices[0]));
                    continue;
                }
            }
            llama_context * ctx = llama_init_from_model_with_legs(model, spec.params, devices);
            if (!ctx) {
                throw std::runtime_error("failed to create governor leg " +
                                         std::to_string(static_cast<int>(spec.engine)));
            }
            legs.push_back({ spec.engine, ctx, side_state{} });
        }
        if (leg_ctx(llama_governor_engine::CPU) == nullptr) {
            throw std::runtime_error("one-model governor requires the CPU leg");
        }
        // Same contract as the two-model form: every leg pair must have a
        // usable KV route or the commits between them cannot work.
        for (size_t a = 0; a < legs.size(); ++a) {
            for (size_t b = a + 1; b < legs.size(); ++b) {
                if (llama_kv_route_query(legs[b].ctx, legs[a].ctx) == llama_kv_route::Reject) {
                    throw std::runtime_error("governor legs have incompatible KV routes");
                }
            }
        }
        prefill_layers_device_ = context_layers_device(prefill_ctx(false));
        decode_layers_device_ = context_layers_device(decode_ctx(llama_governor_engine::CPU));
    } catch (...) {
        for (auto & l : legs) {
            llama_free(l.ctx);
        }
        legs.clear();
        throw;
    }

    stats_.npu_fit = policy_.npu_fit();
    stats_.prefill_token_cap = policy_.prefill_token_cap();
    stats_.expert_substitution_lambda = governor_params.expert_substitution_lambda;
    stats_.cache_budget_warning = policy_.cache_budget_warning();
    stats_.prefill_ctx_ngl = model->n_gpu_layers();
    if (stats_.cache_budget_warning) {
        LLAMA_LOG_WARN("%s: cache budget is smaller than one expert cycle; streaming will thrash\n", __func__);
    }

    // F1/F6b: the Hexagon admission word is device-global (llama-ext.h
    // contract: every context on the device re-reserves at its next decode
    // when it changes) - usable here only because the NPU leg is this
    // governor's sole HTP context. Set it once, right after the leg exists,
    // and never rewrite it, so the generation re-reserve fires once. The
    // values are the one-copy admission (B2: offload_min 1 token, every op
    // kind, no gate). False on hosts without a Hexagon backend - expected.
    if (llama_context * npu = leg_ctx(llama_governor_engine::NPU)) {
        (void) llama_set_backend_admission(npu, 1, GGML_HEXAGON_ADMIT_OPS_ALL, 0);
        // The NPU lane is live exactly when the NPU leg exists and its
        // admission was set: constructing the leg over this model IS the
        // proof the LaunchConfig's fit / readable-streams facts try to
        // express. The binding's npu_lane_enabled platform switch stays the
        // master gate (npu_lane_live), and the safety states still win.
        policy_.set_npu_lane_capable(true);
    }

    // rule v2's leg mask starts as the leg table (no samples yet, so no
    // floor can have fired); decode_impl refreshes it every turn latch
    refresh_decode_legs();
}

llama_context * llama_governor::leg_ctx(llama_governor_engine engine) const {
    for (const auto & l : legs) {
        if (l.engine == engine) {
            return l.ctx;
        }
    }
    return nullptr;
}

llama_governor_engine llama_governor::leg_engine_of(llama_context * ctx) const {
    for (const auto & l : legs) {
        if (l.ctx == ctx) {
            return l.engine;
        }
    }
    return llama_governor_engine::CPU;
}

// The engine a decode TARGET runs on: a leg's engine in the one-model form;
// the two-model form's ctx_prefill plays the NPU role and ctx_decode the CPU
// one (only decode targets are asked).
llama_governor_engine llama_governor::engine_of_ctx(llama_context * ctx) const {
    if (legs.empty()) {
        return ctx == ctx_prefill ? llama_governor_engine::NPU : llama_governor_engine::CPU;
    }
    return leg_engine_of(ctx);
}

// stats_.decode_hop_pairs index for an unordered engine pair
size_t llama_governor::hop_pair_index(llama_governor_engine a, llama_governor_engine b) {
    const auto has = [a, b](llama_governor_engine e) { return a == e || b == e; };
    if (has(llama_governor_engine::NPU)) {
        return has(llama_governor_engine::GPU) ? 0 : 1; // npu<->gpu : npu<->cpu
    }
    return 2; // gpu<->cpu
}

// Rule v2's availability mask (F3). H-04/H-05 rules:
//   - evidence is TURN-LOCAL: the per-leg deltas since the last latch, so a
//     slow cold start cannot strand a leg forever;
//   - every refresh re-admits the present legs first - a demoted leg runs
//     again after one turn latch, and stays admitted unless THIS turn's
//     evidence demotes it again;
//   - evidence shorter than two windows demotes nothing (the first window
//     after a prefill carries warm-up cost; hop commits never pollute the
//     rate - decode_impl times llama_decode only, the commit happens before
//     the clock starts);
//   - the CPU leg is never excludable: it is the fallback that always
//     exists, whatever its floor says.
// Floors default off. Only the one-model form calls this; the two-model
// form keeps the policy's default NPU+CPU mask, so v1 decisions stand.
void llama_governor::refresh_decode_legs() {
    const auto & p = policy_.params();
    const uint32_t hop = p.decode_hop_tokens;
    // indexed like llama_governor_engine (CPU, GPU, NPU)
    const uint64_t tokens_now[3] = { stats_.decode_tokens_cpu, stats_.decode_tokens_gpu, stats_.decode_tokens_npu };
    const uint64_t us_now[3] = { stats_.decode_us_cpu, stats_.decode_us_gpu, stats_.decode_us_npu };
    const float floors[3] = { p.decode_floor_tps_cpu, p.decode_floor_tps_gpu, p.decode_floor_tps_npu };
    const auto demoted = [&](int i) {
        if (floors[i] <= 0.0f || hop == 0) {
            return false; // floor off, or no windows to count evidence in
        }
        const uint64_t tokens = tokens_now[i] - floor_mark_tokens_[i];
        const uint64_t us = us_now[i] - floor_mark_us_[i];
        if (tokens < 2ull * hop) {
            return false; // under two windows of evidence: warm-up noise
        }
        return static_cast<float>(tokens) * 1000000.0f / static_cast<float>(us) < floors[i];
    };
    // decide BEFORE moving the marks: the deltas are this turn's evidence
    const bool npu_ok = leg_ctx(llama_governor_engine::NPU) != nullptr && !demoted(2);
    const bool gpu_ok = leg_ctx(llama_governor_engine::GPU) != nullptr && !demoted(1);
    for (int i = 0; i < 3; ++i) {
        floor_mark_tokens_[i] = tokens_now[i];
        floor_mark_us_[i] = us_now[i];
    }
    set_decode_legs(npu_ok, gpu_ok,
                    true); // H-05: the CPU leg is never excludable
}

llama_context * llama_governor::prefill_ctx(bool cpu_route) const {
    if (legs.empty()) {
        return cpu_route ? ctx_decode : ctx_prefill;
    }
    if (cpu_route) {
        return leg_ctx(llama_governor_engine::CPU);
    }
    // today's ctx_prefill role: the accelerator context, NPU first
    if (llama_context * npu = leg_ctx(llama_governor_engine::NPU)) {
        return npu;
    }
    if (llama_context * gpu = leg_ctx(llama_governor_engine::GPU)) {
        return gpu;
    }
    return leg_ctx(llama_governor_engine::CPU);
}

llama_context * llama_governor::decode_ctx(llama_governor_engine engine) const {
    if (legs.empty()) {
        return engine == llama_governor_engine::NPU ? ctx_prefill : ctx_decode;
    }
    if (llama_context * ctx = leg_ctx(engine)) {
        return ctx;
    }
    return leg_ctx(llama_governor_engine::CPU);
}

auto llama_governor::side_of(llama_context * ctx) -> side_state * {
    if (!legs.empty()) {
        for (auto & l : legs) {
            if (l.ctx == ctx) {
                return &l.state;
            }
        }
        return nullptr;
    }
    return ctx == ctx_prefill ? &prefill_state : &decode_state;
}

auto llama_governor::collect_sides(llama_context * ctxs[3], side_state * states[3]) -> size_t {
    size_t n = 0;
    if (!legs.empty()) {
        for (auto & l : legs) {
            ctxs[n] = l.ctx;
            states[n] = &l.state;
            ++n;
        }
        return n;
    }
    if (ctx_prefill) {
        ctxs[n] = ctx_prefill;
        states[n] = &prefill_state;
        ++n;
    }
    if (ctx_decode) {
        ctxs[n] = ctx_decode;
        states[n] = &decode_state;
        ++n;
    }
    return n;
}
