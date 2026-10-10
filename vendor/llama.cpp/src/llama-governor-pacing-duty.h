#pragma once

#include "llama-ext.h"

// The two battery-temperature lines the duty's second ramp reads: duty 1.0 at
// or below cool_enter_c and decode_pacing_min_duty at or above warn_c. The
// policy fills them from the live profile, so a plugged phone ramps on its own
// cool entry instead of the unplugged one.
struct llama_governor_pacing_battery_lines {
    float cool_enter_c = 0.0f;
    float warn_c = 0.0f;
};

// The decode pacing duty: the fraction of wall-clock time the engine may spend
// decoding (decode time / (decode time + pacing idle)). headroom is the
// platform thermal forecast, NaN when the platform does not answer and +inf
// when it reports maximally hot. battery_temp_c is the battery temperature the
// thermal classifier reads. platform_status is the Android thermal status
// (-1 absent) and caps the duty whatever the forecast says. Pure - no governor
// state - so the unit test drives it directly, and llama_governor::pace_decode
// stays the only caller that sleeps.
float llama_governor_decode_pacing_duty(
        float headroom,
        float battery_temp_c,
        const llama_governor_pacing_battery_lines & battery_lines,
        llama_governor_thermal_state state,
        int32_t platform_status,
        const llama_governor_params & params);
