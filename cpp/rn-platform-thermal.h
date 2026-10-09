#pragma once

#include <cstdint>

#if defined(__ANDROID__)
#include <android/thermal.h>
#endif

namespace rnllama {

/** The engine's platform_thermal_status scale: -1 absent, 0..6 = Android's
 *  AThermalStatus ladder NONE..SHUTDOWN. The field is declared in llama-ext.h
 *  and given its meaning in llama-governor-policy.cpp apply_platform_floor,
 *  where every status 2..6 floors the thermal state at COOLMODE. */
constexpr int32_t k_platform_thermal_absent = -1;

/** The top of that scale: NDK ATHERMAL_STATUS_SHUTDOWN (android/thermal.h). */
constexpr int32_t k_platform_thermal_shutdown = 6;

/** One native read per this many microseconds of wall time: the decode path
 *  is the only reader, and a read is a round trip to the thermal HAL service
 *  that a decode step should not pay more than once a second. */
constexpr int64_t k_platform_thermal_interval_us = 1000000;

/** AThermal's ladder mapped 1:1 onto the engine's. ATHERMAL_STATUS_ERROR and
 *  any value a newer platform adds above SHUTDOWN are refused to "absent":
 *  a number the platform never sent must not reach the >= 2 floor, which is
 *  the clamp the engine's own ingress already applies (normalize_platform_status).
 *  Pure, so a host build and the test can drive it without Android. */
int32_t rn_map_platform_thermal_status(int32_t athermal_status);

/** Android's current thermal status, read natively (android/thermal.h).
 *
 *  The app can only report the platform status once per completion, so a turn
 *  long enough to heat the phone finishes on the status the app handed over
 *  at load: the engine never sees that the phone went SEVERE in the middle.
 *  The manager is acquired once for this object's lifetime
 *  (AThermal_acquireManager is a binder to the thermal HAL service, not a
 *  call to repeat per read) and status() is called from the decode thread only.
 *
 *  Off Android nothing answers and status() reports "absent", so the same
 *  translation unit links into every build. Never throws. */
class rn_platform_thermal {
public:
    rn_platform_thermal();
    ~rn_platform_thermal();

    rn_platform_thermal(const rn_platform_thermal &) = delete;
    rn_platform_thermal & operator=(const rn_platform_thermal &) = delete;

    /** The current status mapped onto the engine's scale, or
     *  k_platform_thermal_absent: no manager acquired (not Android, a
     *  pre-API-30 libandroid that exports none of the three entry points, or
     *  an acquire that returned null) or the platform answered
     *  ATHERMAL_STATUS_ERROR. */
    int32_t status() const;

private:
#if defined(__ANDROID__)
    // An AThermalManager handle, opaque on both sides of the .cpp boundary;
    // null is "never acquired" and is the answer "absent".
    AThermalManager * manager_ = nullptr;
#endif
};

/** Whether a freshly read status must reach the engine, plus the clock that
 *  throttles the read. Pure: the caller owns last_read_us and the test drives
 *  it with literals.
 *
 *  True only for a read that is due (at most one native read per
 *  k_platform_thermal_interval_us of wall time, ggml_time_us is monotonic),
 *  and that differs from remembered_status, the status the last profile
 *  carried. A read of "absent" is never true: it is the platform declining to
 *  vote, and it must not overwrite the status the app already handed over.
 *  last_read_us advances whenever a read was taken, so an unchanged status
 *  still pays its second of wall time. */
bool rn_platform_thermal_should_send(int64_t now_us, int64_t & last_read_us,
                                     int32_t native_status, int32_t remembered_status);

} // namespace rnllama
