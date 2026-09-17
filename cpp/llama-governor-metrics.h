#pragma once

#include <cstdint>
#include <mutex>

struct llama_governor_counter_sample {
    uint64_t cpu_us = 0;
    uint64_t read_bytes = 0;
    uint64_t read_us = 0;
    uint64_t stall_us = 0;
    uint64_t management_us = 0;
};

struct llama_governor_tally_result {
    llama_governor_counter_sample pre;
    llama_governor_counter_sample post;
    llama_governor_counter_sample delta;
    bool closed = false;
};

class llama_governor_prefill_tally {
public:
    void begin(const llama_governor_counter_sample & sample);
    void end(const llama_governor_counter_sample & sample);
    llama_governor_tally_result snapshot(const llama_governor_counter_sample & sample) const;
    bool active() const;

private:
    static llama_governor_counter_sample subtract(
            const llama_governor_counter_sample & post,
            const llama_governor_counter_sample & pre);

    llama_governor_tally_result result_;
    bool active_ = false;
};

class llama_governor_stall_union_state {
public:
    void enter(uint64_t now_us);
    void exit(uint64_t now_us);
    uint64_t total(uint64_t now_us) const;

private:
    uint32_t depth_ = 0;
    uint64_t open_us_ = 0;
    uint64_t total_us_ = 0;
};

class llama_governor_stall_union {
public:
    void enter(uint64_t now_us);
    void exit(uint64_t now_us);
    uint64_t total(uint64_t now_us) const;

private:
    mutable std::mutex mutex_;
    llama_governor_stall_union_state state_;
};
