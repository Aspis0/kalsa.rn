#pragma once

#include "rn-battery-sysfs.h"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace rnllama {

// One decode-hop window's Stage 0 observation. NaN and the -1 codes mean
// unknown; every such cell crosses the JSI bridge as null. The three ages
// are APPROXIMATE: each stamp is independently atomic, so one sample can mix
// generations, and no Stage 0 check treats an age as exact.
struct rn_window_sample {
    // Tokens decoded since the last prefill when the window opened.
    uint32_t tokens = 0;
    // The leg that ran the step before this window: 0 cpu, 1 gpu, 2 npu;
    // -1 none (the completion's first window has no predecessor in it).
    int8_t leg_prev = -1;
    // The platform forecast and platform status at the window boundary, with
    // the ages of their last reads.
    float headroom = std::numeric_limits<float>::quiet_NaN();
    int32_t headroom_age_ms = -1;
    int32_t status = -1;
    int32_t status_age_ms = -1;
    int32_t app_age_ms = -1;
    // Hottest zone temp per leg (max over the zones, unlike the headroom's
    // min); NaN when the leg is unknown (no zones, or one zone went dark).
    float cpu_hot_c = std::numeric_limits<float>::quiet_NaN();
    float npu_hot_c = std::numeric_limits<float>::quiet_NaN();
    float gpu_hot_c = std::numeric_limits<float>::quiet_NaN();
    // The forecast and status the engine APPLIED for the step whose realized
    // duty is duty_last - the previous window's inputs, not this boundary's.
    float applied_headroom = std::numeric_limits<float>::quiet_NaN();
    int32_t applied_status = -1;
    // Realized duty of the previous single-token step; -1 when no step of
    // this completion has run yet (the engine's value is a leftover).
    float duty_last = -1.0f;
};

// Passive-trip range per leg in C (milli-C / 1000 at the reader); NaN when
// the leg has no usable zone.
struct rn_leg_trips {
    float cpu_min_c = std::numeric_limits<float>::quiet_NaN();
    float cpu_max_c = std::numeric_limits<float>::quiet_NaN();
    float npu_min_c = std::numeric_limits<float>::quiet_NaN();
    float npu_max_c = std::numeric_limits<float>::quiet_NaN();
    float gpu_min_c = std::numeric_limits<float>::quiet_NaN();
    float gpu_max_c = std::numeric_limits<float>::quiet_NaN();
};

// The per-completion observation record of the decode hop: first/last step
// stamps, the first step's snapshot and battery probe, and a 256-entry ring
// of window samples with an overflow flag (route_chunks precedent). Decode
// thread only, under rn_governor::engine_mutex_ - the JSI-side copy takes
// that same lock in rn_governor::samples(); no internal locking.
//
// Header-only for the same reason as rn-battery-sysfs.h: the HTP skel
// fingerprint covers every cmake list a .cpp would have to be registered in,
// and this stage must not move it (scripts/htp-skels-common.sh
// HTP_SRC_PATHS). Every build that compiles rn-governor.cpp already sees it
// through rn-governor.h.
class rn_governor_samples {
public:
    static constexpr uint32_t k_ring_capacity = 256;

    // Once per governor, at init_hop_reader.
    void set_trips(const rn_leg_trips & trips) { trips_ = trips; }

    // Completion start (reset_prefill_stats): ring empty, stamps 0, the
    // first-step latch open. Trips survive: they are per governor, not per
    // completion.
    void reset() {
        first_ = {};
        battery_ = {};
        first_step_ms_ = 0;
        last_step_ms_ = 0;
        first_step_done_ = false;
        overflow_ = false;
        count_ = 0;
        // ring_ stays uncleared: count_ bounds what samples() exposes, so
        // stale cells past count_ are never read.
    }

    // The completion's first 1-token decode batch with its boundary snapshot
    // and one battery probe; later calls are ignored.
    void first_step(int64_t now_ms, const rn_window_sample & snapshot,
                    const rn_battery_sysfs & battery) {
        if (first_step_done_) {
            return;
        }
        first_step_done_ = true;
        first_ = snapshot;
        battery_ = battery;
        first_step_ms_ = now_ms;
    }

    // Every 1-token decode batch, the first included.
    void last_step(int64_t now_ms) { last_step_ms_ = now_ms; }

    // One window boundary sample; once the ring is full, further samples are
    // dropped and the overflow flag stays set so truncation is observable.
    void record(const rn_window_sample & sample) {
        if (count_ >= k_ring_capacity) {
            overflow_ = true;
            return;
        }
        ring_[count_] = sample;
        ++count_;
    }

    const rn_leg_trips & trips() const { return trips_; }
    int64_t first_step_ms() const { return first_step_ms_; }
    int64_t last_step_ms() const { return last_step_ms_; }
    const rn_window_sample & first() const { return first_; }
    const rn_battery_sysfs & battery() const { return battery_; }
    uint32_t count() const { return count_; }
    bool overflow() const { return overflow_; }

    // The recorded samples, oldest first.
    std::vector<rn_window_sample> samples() const {
        return {ring_.begin(), ring_.begin() + static_cast<std::ptrdiff_t>(count_)};
    }

private:
    rn_leg_trips trips_{};
    rn_window_sample first_{};
    rn_battery_sysfs battery_{};
    int64_t first_step_ms_ = 0;
    int64_t last_step_ms_ = 0;
    bool first_step_done_ = false;
    bool overflow_ = false;
    uint32_t count_ = 0;
    std::array<rn_window_sample, k_ring_capacity> ring_{};
};

} // namespace rnllama
