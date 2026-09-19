#include "llama-governor-metrics.h"

namespace {

uint64_t subtract_counter(uint64_t post, uint64_t pre) {
    return post >= pre ? post - pre : 0;
}

} // namespace

void llama_governor_prefill_tally::begin(const llama_governor_counter_sample & sample) {
    result_ = {};
    result_.pre = sample;
    active_ = true;
}

void llama_governor_prefill_tally::end(const llama_governor_counter_sample & sample) {
    if (!active_) {
        return;
    }
    result_.post = sample;
    result_.delta = subtract(sample, result_.pre);
    result_.closed = true;
    active_ = false;
}

llama_governor_tally_result llama_governor_prefill_tally::snapshot(
        const llama_governor_counter_sample & sample) const {
    if (!active_) {
        return result_;
    }
    auto current = result_;
    current.post = sample;
    current.delta = subtract(sample, result_.pre);
    return current;
}

bool llama_governor_prefill_tally::active() const {
    return active_;
}

llama_governor_counter_sample llama_governor_prefill_tally::subtract(
        const llama_governor_counter_sample & post,
        const llama_governor_counter_sample & pre) {
    return {
        subtract_counter(post.cpu_us, pre.cpu_us),
        subtract_counter(post.read_bytes, pre.read_bytes),
        subtract_counter(post.read_us, pre.read_us),
        subtract_counter(post.stall_us, pre.stall_us),
        subtract_counter(post.management_us, pre.management_us),
    };
}

void llama_governor_stall_union_state::enter(uint64_t now_us) {
    if (depth_++ == 0) {
        open_us_ = now_us;
    }
}

void llama_governor_stall_union_state::exit(uint64_t now_us) {
    if (depth_ == 0) {
        return;
    }
    if (--depth_ == 0) {
        total_us_ += now_us >= open_us_ ? now_us - open_us_ : 0;
    }
}

uint64_t llama_governor_stall_union_state::total(uint64_t now_us) const {
    if (depth_ == 0) {
        return total_us_;
    }
    return total_us_ + (now_us >= open_us_ ? now_us - open_us_ : 0);
}

void llama_governor_stall_union::enter(uint64_t now_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.enter(now_us);
}

void llama_governor_stall_union::exit(uint64_t now_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.exit(now_us);
}

uint64_t llama_governor_stall_union::total(uint64_t now_us) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_.total(now_us);
}
