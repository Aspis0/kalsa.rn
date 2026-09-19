#include "bmoe_stream.h"

#include "bmoe/config.h"
#include "bmoe/recipe.h"
#include "bmoe/expert_stream_source.h"
#include "bmoe/gguf_offsets.h"
#include "bmoe/router_hook.h"

#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Routed expert weights: blk.N.ffn_{up,down,gate,gate_up}_{,ch}exps — not *_shexp, not the
// router. Matches llama.cpp's LLM_FFN_EXPS_REGEX; mirrored from dense_weights.cpp so a capture
// miss cannot hand ffn_*_exps to a host-copy dense mode (that path would rebind the streamer's
// live mmap).
bool is_routed_expert_name(const std::string & name) {
    if (name.find("shexp") != std::string::npos) return false;
    if (name.find(".ffn_") == std::string::npos) return false;
    return name.find("exps") != std::string::npos;
}

std::unordered_set<std::string> expert_tensor_names(const std::vector<bmoe::LayerExperts> & layers) {
    std::unordered_set<std::string> names;
    for (const bmoe::LayerExperts & L : layers) {
        if (!L.bound) continue;
        for (int p = 0; p < bmoe::MoeRecipe::max_exps; ++p)
            if (L.proj[p].tensor) names.insert(L.proj[p].tensor->name);
    }
    return names;
}

bool parse_dense_weights(const char * s, bmoe::DenseWeightsMode & out) {
    using M = bmoe::DenseWeightsMode;
    const M modes[] = {M::Mmap, M::Warmed, M::Anonymous, M::Pinned, M::AnonGpu};
    if (!s || !s[0]) {
        out = M::Anonymous;
        return true;
    }
    for (M m : modes) {
        if (std::strcmp(s, bmoe::dense_weights_flag(m)) == 0) {
            out = m;
            return true;
        }
    }
    return false;
}

} // namespace

namespace kalsa {

struct MoeStream::State {
    const bmoe::MoeRecipe * recipe = nullptr;
    bmoe::MoeStreamConfig cfg;
    bmoe::GgufMeta meta;
    std::unique_ptr<bmoe::RouterHook> hook;
    std::unique_ptr<bmoe::ExpertStreamSource> source;

    // False except while the streamer is actually driving a graph. See eval_trampoline.
    bool hook_live = false;

    // Every tensor ExpertStreamSource::init may repoint, with the mmap pointer it had before.
    // init() rebinds layer by layer and returns false in the middle without unwinding, so a
    // failure there leaves earlier layers pointing at reserved-but-never-filled memory: the model
    // would keep running and compute on zeros, silently. The engine's own caller can ignore this
    // because it aborts the whole session; ours declines and lets the app carry on, so ours has
    // to put the pointers back.
    std::vector<std::pair<ggml_tensor *, void *>> rebound;
};

bool MoeStream::eval_trampoline(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<MoeStream::State *>(user_data);
    if (!st || !st->hook_live) return false;
    return bmoe::RouterHook::c_eval(t, ask, st->hook.get());
}

MoeStream::MoeStream() : st_(std::make_unique<State>()) {}

MoeStream::~MoeStream() { shutdown(); }

void MoeStream::shutdown() {
    if (st_) {
        st_->hook_live = false; // first: no further callback can run against what we tear down
        // Back onto the mmap before the buffers are freed. The mapping is still there — streaming
        // stops reading from it, it never unmaps it — so this leaves a model that still decodes
        // rather than one holding pointers into freed memory.
        for (const auto & r : st_->rebound) r.first->data = r.second;
        st_->rebound.clear();
        if (st_->source) st_->source->shutdown();
        st_->source.reset();
        if (st_->hook) st_->hook->set_source(nullptr);
    }
    armed_ = false;
    active_ = false;
}

void MoeStream::set_cache_budget(size_t bytes) {
    if (st_ && st_->source) st_->source->set_cache_budget(bytes);
}

bool MoeStream::arm(common_params & params, std::string & err) {
    shutdown();
    if (!params.kalsa_moe.enabled) {
        err = "kalsa_moe disabled";
        return false;
    }

    st_->meta = bmoe::read_gguf_meta(params.model.path.c_str());
    if (!st_->meta.ok) {
        err = "cannot read gguf metadata: " + params.model.path;
        return false;
    }
    const bmoe::GgufModelInfo & info = st_->meta.info;
    if (info.n_expert == 0) {
        err = "model is dense (n_expert=0)";
        return false;
    }
    st_->recipe = bmoe::find_moe_recipe(info.arch.c_str());
    if (!st_->recipe) {
        err = "no MoE recipe for architecture '" + info.arch + "'";
        return false;
    }

    const auto & in = params.kalsa_moe;
    bmoe::MoeStreamConfig cfg;
    cfg.enabled = true;
    cfg.cache_mb = in.cache_mb;
    cfg.cache_auto = in.cache_auto;
    cfg.cache_floor_mb = in.cache_floor_mb;
    cfg.cache_ceil_mb = in.cache_ceil_mb;
    cfg.io_threads = in.io_threads;
    cfg.overlap = in.overlap;
    cfg.drop_cold_frac = in.drop_cold_frac;
    cfg.drop_renorm = !in.drop_no_renorm;
    if (!parse_dense_weights(in.dense_weights, cfg.dense_weights)) {
        err = std::string("unknown dense_weights '") + in.dense_weights + "'";
        return false;
    }
    if (cfg.cache_mb >= 1 && cfg.cache_mb < bmoe::MoeStreamConfig::cache_min_mb) {
        err = "cache_mb " + std::to_string(cfg.cache_mb) +
              " is below MoeStreamConfig::cache_min_mb; a budget under one token's working set "
              "is slower than no cache (use 0 or >=" +
              std::to_string(bmoe::MoeStreamConfig::cache_min_mb) + ")";
        return false;
    }
    if (cfg.cache_mb < 0) {
        err = "cache_mb must be >= 0";
        return false;
    }
    if (cfg.cache_auto && cfg.cache_mb > 0) {
        err = "cache_auto and an explicit cache_mb are mutually exclusive: choose auto-sizing "
              "(cache_mb 0 + cache_auto) or a fixed budget (cache_mb > 0)";
        return false;
    }
    if (!(cfg.drop_cold_frac >= 0.0f && cfg.drop_cold_frac <= 1.0f)) {
        err = "drop_cold_frac must be in [0, 1] (0 = off)";
        return false;
    }
    // Dropping decides from live cache state: with no cache every routed expert is a miss, so the
    // policy would discard on a signal that means nothing. The engine's validator refuses it and
    // so does this one — silently accepting it would degrade output for no I/O saved.
    if (cfg.drop_cold_frac > 0.0f && cfg.cache_mb == 0 && !cfg.cache_auto) {
        err = "drop_cold_frac requires the LRU cache (cache_mb > 0 or cache_auto)";
        return false;
    }
    // Parsed end to end and applied by nobody: the top-k override lives on the engine's RunConfig,
    // not on MoeStreamConfig, and reaches llama.cpp as a model KV override the glue does not build.
    // Refusing beats ignoring — a lossy knob that silently does nothing is a wrong measurement
    // waiting to be published.
    if (in.n_expert_used != 0) {
        err = "n_expert_used is not implemented in the app path (engine applies it via a model KV "
              "override); leave it 0";
        return false;
    }
    if (cfg.io_threads < 1) cfg.io_threads = 1;
    if (cfg.io_threads > bmoe::MoeStreamConfig::io_threads_max)
        cfg.io_threads = bmoe::MoeStreamConfig::io_threads_max;

    st_->cfg = cfg;
    st_->hook = std::make_unique<bmoe::RouterHook>(*st_->recipe, info.n_layer);
    st_->hook->set_drop_policy(cfg.drop_cold_frac, cfg.drop_renorm, cfg.drop_prefill);
    params.cb_eval = &MoeStream::eval_trampoline;
    params.cb_eval_user_data = st_.get();

    // Repack is load-bearing, not a tuning knob: the streamer rebinds tensor->data to buffers
    // filled from the file's NATIVE byte layout, and repack would change that layout so the
    // file offsets would no longer describe what the matmul reads. The engine honours
    // no_extra_bufts (→ use_extra_bufts = !no_extra_bufts), and takes mmap through the
    // load_mode enum since pin 134a35cf2 replaced the use_mmap/use_mlock pair: the stream
    // needs mmap on, so raise it while preserving whatever mlock the incoming mode carries.
    params.no_extra_bufts = true;
    params.load_mode = params.load_mode == LLAMA_LOAD_MODE_MLOCK ||
                               params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK
                           ? LLAMA_LOAD_MODE_MMAP_MLOCK
                           : LLAMA_LOAD_MODE_MMAP;

    armed_ = true;
    return true;
}

bool MoeStream::bind(llama_context * ctx, std::string & err) {
    if (!armed_ || !st_ || !st_->hook || !st_->recipe) {
        err = "not armed";
        return false;
    }
    if (!ctx) {
        err = "null context";
        return false;
    }

    // Every early return from here on goes through fail(), which is what puts the rebound
    // tensors back and makes the callback inert again. Declared before the capture so the
    // warm-up's own failure path cannot skip it — leaving the hook mid-capture would strand it
    // there for the life of the context, harvesting tensors on every decode and isolating none.
    auto fail = [&](std::string m) -> bool {
        err = std::move(m);
        st_->hook->end_capture(); // idempotent; the warm-up's own failure path lands here too
        st_->hook_live = false;
        for (const auto & r : st_->rebound) r.first->data = r.second;
        st_->rebound.clear();
        if (st_->source) st_->source->shutdown();
        st_->source.reset(); // frees the buffers only after nothing points at them
        st_->hook->set_source(nullptr);
        if (llama_memory_t mem = llama_get_memory(ctx)) llama_memory_clear(mem, true);
        return false;
    };

    // One mmap-resident BOS decode: the eval-callback harvests expert tensor pointers. KV is
    // wiped afterwards so this token never reaches the app.
    st_->hook_live = true;
    st_->hook->begin_capture();
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = model ? llama_model_get_vocab(model) : nullptr;
    llama_token warm_tok = vocab ? llama_vocab_bos(vocab) : 0;
    if (warm_tok < 0) warm_tok = 0;
    llama_batch warm = llama_batch_get_one(&warm_tok, 1);
    if (llama_decode(ctx, warm) != 0) return fail("capture warm-up decode failed");
    st_->hook->end_capture();

    const bmoe::GgufOffsets & offs = st_->meta.offsets;
    if (!offs.ok) return fail("cannot read gguf offsets");

    std::vector<bmoe::LayerExperts> layers = st_->hook->captured();
    int n_expert = 0;
    int n_bound = 0;
    for (bmoe::LayerExperts & L : layers) {
        if (!L.bound) continue;
        ++n_bound;
        for (int p = 0; p < bmoe::MoeRecipe::max_exps; ++p) {
            if (!st_->recipe->exps_suffix[p]) continue;
            ggml_tensor * t = L.proj[p].tensor;
            if (!t)
                return fail(std::string("captured MoE layer is missing expert tensor '") +
                            st_->recipe->exps_suffix[p] + "'");
            auto it = offs.off_by_name.find(t->name);
            auto fi = offs.file_by_name.find(t->name);
            if (it == offs.off_by_name.end() || fi == offs.file_by_name.end())
                return fail(std::string("no gguf offset for tensor ") + t->name);
            L.proj[p].file_off = it->second;
            L.proj[p].file_idx = fi->second;
            const int ne2 = (int) t->ne[2];
            if (n_expert == 0) n_expert = ne2;
            else if (ne2 != n_expert)
                return fail(std::string("inconsistent expert count: tensor ") + t->name);
        }
    }
    if (n_bound == 0) return fail("no MoE expert tensors captured — is this a MoE model?");

    st_->source = std::make_unique<bmoe::ExpertStreamSource>();
    if (bmoe::dense_mode_host_copy(st_->cfg.dense_weights)) {
        const std::unordered_set<std::string> expert_names = expert_tensor_names(layers);
        std::vector<bmoe::DenseTensorRef> dense;
        for (const auto & kv : st_->hook->captured_weights()) {
            const std::string & name = kv.first;
            if (expert_names.count(name) || is_routed_expert_name(name)) continue;
            auto off = offs.off_by_name.find(name);
            auto sz = offs.size_by_name.find(name);
            auto fi = offs.file_by_name.find(name);
            if (off == offs.off_by_name.end() || sz == offs.size_by_name.end() ||
                fi == offs.file_by_name.end())
                continue;
            dense.push_back({kv.second, off->second, sz->second, fi->second});
        }
        for (const bmoe::DenseTensorRef & d : dense)
            if (d.tensor) st_->rebound.emplace_back(d.tensor, d.tensor->data);
        st_->source->set_dense_tensors(std::move(dense));
    }
    for (const bmoe::LayerExperts & L : layers) {
        if (!L.bound) continue;
        for (int p = 0; p < bmoe::MoeRecipe::max_exps; ++p)
            if (L.proj[p].tensor) st_->rebound.emplace_back(L.proj[p].tensor, L.proj[p].tensor->data);
    }

    if (!st_->source->init(offs.shard_paths, n_expert, std::move(layers), st_->cfg))
        return fail("expert stream source init failed");
    st_->hook->set_source(st_->source.get());

    if (st_->cfg.overlap) st_->source->enable_overlap_hook();

    llama_set_abort_callback(
        ctx,
        [](void * ud) -> bool { return static_cast<bmoe::ExpertStreamSource *>(ud)->fatal(); },
        st_->source.get());

    if (llama_memory_t mem = llama_get_memory(ctx)) llama_memory_clear(mem, true);
    active_ = true;
    return true;
}

} // namespace kalsa
