#include "llama-governor.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace {

// Bound a slow token's added latency so an unexpectedly long engine step
// cannot turn sustained COOLMODE into an unresponsive session.
constexpr uint64_t k_decode_pacing_cap_us = 250000;

} // namespace

void llama_governor::pace_decode(uint32_t n_tokens, uint64_t compute_us) {
    if (n_tokens != 1 || policy_.thermal_state() != llama_governor_thermal_state::COOLMODE) {
        return;
    }
    // COOLMODE (battery heat, or a platform status of 2+ over FAST/WARM) wants
    // a sustainable slow rate; pausing remains the emergency brake. Below the
    // cap, duty is engine decode time / (decode time + pacing idle), so the idle
    // delay is compute_us * (1 / duty - 1); commit, sampling and callbacks
    // are outside this window.
    const double requested_us = static_cast<double>(compute_us) *
                               (1.0 / policy_.decode_coolmode_duty() - 1.0);
    if (requested_us <= 0.0) {
        return;
    }
    const bool capped = requested_us >= static_cast<double>(k_decode_pacing_cap_us);
    const uint64_t delay_us = static_cast<uint64_t>(
            std::min(requested_us, static_cast<double>(k_decode_pacing_cap_us)));
    if (delay_us == 0) {
        return;
    }
    if (decode_sleep_fn_ != nullptr) {
        decode_sleep_fn_(delay_us);
    } else {
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }
    decode_paced_us_ += delay_us;
    stats_.decode_paced_ms = decode_paced_us_ / 1000;
    ++stats_.decode_paced_tokens;
    if (capped) {
        ++stats_.decode_paced_capped;
    }
}
