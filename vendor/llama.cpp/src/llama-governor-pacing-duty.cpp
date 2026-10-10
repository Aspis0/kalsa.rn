#include "llama-governor-pacing-duty.h"

#include <algorithm>
#include <cmath>

namespace {

// The forecast ramp alone: full speed up to the start line, the floor duty
// from the full line, linear between. A reading past the full line (+inf
// included) is the floor; a negative reading is clamped to the start line
// rather than read as cooler than cool.
float headroom_duty(float headroom, const llama_governor_params & params) {
    if (headroom < 0.0f) {
        headroom = 0.0f;
    }
    if (headroom <= params.decode_pacing_headroom_start) {
        return 1.0f;
    }
    if (headroom >= params.decode_pacing_headroom_full) {
        return params.decode_pacing_min_duty;
    }
    // start < full and min_duty in (0, 1] are sanitizer invariants, so the
    // window is positive and the fraction lands in (0, 1).
    const float window = params.decode_pacing_headroom_full - params.decode_pacing_headroom_start;
    const float fraction = (headroom - params.decode_pacing_headroom_start) / window;
    return 1.0f + fraction * (params.decode_pacing_min_duty - 1.0f);
}

// The battery ramp alone: full speed at or below the cool entry line, the
// floor duty at or above the pause line, linear between. The policy hands
// over the profile's cool entry, which the fixed pause line always stays
// above, so the window is positive.
float battery_duty(float battery_temp_c, const llama_governor_pacing_battery_lines & lines,
                   const llama_governor_params & params) {
    if (battery_temp_c <= lines.cool_enter_c) {
        return 1.0f;
    }
    if (battery_temp_c >= lines.warn_c) {
        return params.decode_pacing_min_duty;
    }
    const float window = lines.warn_c - lines.cool_enter_c;
    const float fraction = (battery_temp_c - lines.cool_enter_c) / window;
    return 1.0f + fraction * (params.decode_pacing_min_duty - 1.0f);
}

// The platform status caps the duty whatever the forecast says. A cool forecast
// must not hand a COOLMODE decode full speed while the platform reports heat:
// SEVERE and above hold the floor, MODERATE holds the flat COOLMODE duty, and
// absent or LIGHT or lower adds no cap.
float status_duty(int32_t platform_status, const llama_governor_params & params) {
    if (platform_status >= 3) {
        return params.decode_pacing_min_duty;
    }
    if (platform_status == 2) {
        return params.decode_coolmode_duty;
    }
    return 1.0f;
}

} // namespace

// A phone that answers the thermal forecast gets a graduated pace wherever
// decode runs: full speed while there is margin, and only as much slowdown as
// the forecast calls for. The battery temperature ramps underneath it, so a
// hot battery slows decode even when the forecast is cool or absent, and the
// tighter of the three wins. The flat COOLMODE duty stays the fallback for a
// platform that sends no forecast at all.
float llama_governor_decode_pacing_duty(
        float headroom,
        float battery_temp_c,
        const llama_governor_pacing_battery_lines & battery_lines,
        llama_governor_thermal_state state,
        int32_t platform_status,
        const llama_governor_params & params) {
    const float battery = battery_duty(battery_temp_c, battery_lines, params);
    const float cap = status_duty(platform_status, params);
    if (std::isnan(headroom)) {
        // NaN is the only absent forecast. +inf never lands here: it is a
        // reading (maximally hot) and reading it as absent would hand COOLMODE
        // its flat duty, which is faster than the floor.
        const float no_forecast = state == llama_governor_thermal_state::COOLMODE
                                        ? params.decode_coolmode_duty
                                        : 1.0f;
        return std::min({ no_forecast, battery, cap });
    }
    return std::min({ headroom_duty(headroom, params), battery, cap });
}
