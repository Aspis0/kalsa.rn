// MoE expert streaming for llama.rn: the whole integration, in one seam.
//
// The streaming engine itself is vendored unmodified under native/bmoe/{include,src}; this
// file is the only Kalsa-written part, and it exists so rn-llama.cpp gains ten lines rather
// than a hundred and fifty. Two calls bracket llama.rn's own model load:
//
//   arm()  before common_init_from_params — installs the eval callback and forces the load
//          layout the streamer requires (mmap on, repack off, no mlock).
//   bind() after the context exists — runs one warm-up decode to harvest the expert tensors
//          from the graph, resolves their file offsets, and goes live.
//
// Failing either one is not fatal to the app: the model still loads and runs mmap-resident,
// exactly as it did before this file existed. Streaming is a strictly additive fast path for
// a model that would otherwise not fit, so it declines rather than refuses.
#pragma once

#include "common.h"
#include "llama.h"

#include <memory>
#include <string>

namespace bmoe {
class RouterHook;
class ExpertStreamSource;
} // namespace bmoe

namespace kalsa {

// One streaming session, owned by one llama_rn_context. Not copyable: it holds the
// process-global expert-ready hook while live.
class MoeStream {
public:
    MoeStream();
    ~MoeStream();

    MoeStream(const MoeStream &) = delete;
    MoeStream & operator=(const MoeStream &) = delete;

    // Reads params.kalsa_moe and the gguf metadata, decides whether this model can stream,
    // and if so points params.cb_eval at the router hook and fixes the load layout. False +
    // `err` when streaming is off, the model is dense, its architecture has no recipe, or the
    // file cannot be read.
    bool arm(common_params & params, std::string & err);

    // Warm-up capture, offset resolution, buffer binding. Must be called after arm() and
    // after the context was created from those same params. False + `err` leaves the model
    // fully mmap-resident and usable.
    bool bind(llama_context * ctx, std::string & err);

    // Stop streaming: unregister the hook, join the I/O lanes, release the cache. Safe to
    // call twice, and called by the destructor. MUST run before the llama_context dies —
    // the streamer's buffers are what the model's expert tensors point at.
    void shutdown();

    bool armed() const { return armed_; }
    bool active() const { return active_; }

    // Drop the expert cache to `bytes` and evict down to it now. For Android onTrimMemory.
    // Precondition: no decode in flight.
    void set_cache_budget(size_t bytes);

private:
    struct State;

    // llama.cpp's eval callback, installed instead of RouterHook::c_eval itself. The indirection
    // buys one thing: a bind() that failed or declined must cost NOTHING afterwards. The router
    // hook isolates every ffn_moe_topk node whenever it is installed — a graph split and a full
    // compute-thread sync per MoE layer per token — and it does that whether or not a stream
    // source was ever attached, because llama.cpp exposes no way to remove cb_eval once the
    // context exists. This trampoline returns false while the stream is not live, which is the
    // "grouped with its neighbours, no callback" answer, i.e. the graph llama.rn would have run
    // if none of this were here.
    static bool eval_trampoline(ggml_tensor * t, bool ask, void * user_data);

    std::unique_ptr<State> st_;
    bool armed_ = false;
    bool active_ = false;
};

} // namespace kalsa
