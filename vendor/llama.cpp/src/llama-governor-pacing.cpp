#include "llama-governor-pacing-duty.h"
#include "llama-governor.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace {

// Bound a slow token's added latency so an unexpectedly long engine step
// cannot turn a paced decode into an unresponsive session.
constexpr uint64_t k_decode_pacing_cap_us = 250000;

} // namespace

void llama_governor::pace_decode(uint32_t n_tokens, uint64_t compute_us) {
    if (n_tokens != 1) {
        return;
    }
    // The duty comes from the platform thermal headroom forecast wherever
    // decode runs (FAST, WARM, COOLMODE, LOWBAT), with the battery temperature
    // ramping underneath it, so margin means full speed and the heat the
    // forecast or the battery reports is the only slowdown. The two ramps are
    // the graduated brake; the CRITICAL abort is the hard one that stops a
    // decode that must not run at all. duty is engine decode time /
    // (decode time + pacing idle), so the idle delay below is
    // compute_us * (1 / duty - 1); commit, sampling and callbacks are outside
    // this window.
    const auto thermal = policy_.thermal_snapshot();
    const float duty = llama_governor_decode_pacing_duty(
            policy_.platform_thermal_headroom(), policy_.current_temperature_c(),
            policy_.decode_battery_lines(), thermal.state, thermal.platform_status,
            policy_.params());
    const double requested_us = static_cast<double>(compute_us) * (1.0 / duty - 1.0);
    if (requested_us <= 0.0) {
        stats_.decode_pacing_duty_last = 1.0f; // this step ran unpaced
        return;
    }
    const bool capped = requested_us >= static_cast<double>(k_decode_pacing_cap_us);
    const uint64_t delay_us = static_cast<uint64_t>(
            std::min(requested_us, static_cast<double>(k_decode_pacing_cap_us)));
    if (delay_us == 0) {
        stats_.decode_pacing_duty_last = 1.0f; // this step ran unpaced
        return;
    }
    // Recorded only once the delay is known, and as realized: a capped step
    // sleeps less than the curve asked for, so it ran at a higher duty.
    stats_.decode_pacing_duty_last = static_cast<float>(
            static_cast<double>(compute_us) / static_cast<double>(compute_us + delay_us));
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
