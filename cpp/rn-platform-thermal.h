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

/** Minimum wall time between native reads. */
constexpr int64_t k_platform_thermal_interval_us = 1000000;
constexpr int64_t k_platform_thermal_escalation_debounce_us = 5000000;

struct rn_platform_thermal_send_state {
    int32_t pending_status = k_platform_thermal_absent;
    int64_t pending_since_us = 0;
    int64_t last_seen_us = 0;
};

struct rn_platform_thermal_send_decision {
    bool send = false;
    bool pending_started = false;
    rn_platform_thermal_send_state next_state;
};

/** AThermal's ladder mapped 1:1 onto the engine's. ATHERMAL_STATUS_ERROR and
 *  any value a newer platform adds above SHUTDOWN are refused to "absent":
 *  a number the platform never sent must not reach the >= 2 floor, which is
 *  the clamp the engine's own ingress already applies (normalize_platform_status).
 *  Pure, so a host build and the test can drive it without Android. */
int32_t rn_map_platform_thermal_status(int32_t athermal_status);

/** Reads Android's thermal status through the API 30 NDK interface when
 *  available. The manager is acquired once per object. Off Android, status()
 *  reports absent so this translation unit remains portable. */
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
    const char * unavailable_reason() const;

private:
#if defined(__ANDROID__)
    // An AThermalManager handle, opaque on both sides of the .cpp boundary;
    // null is "never acquired" and is the answer "absent".
    AThermalManager * manager_ = nullptr;
#endif
};

/** Advances last_read_us and returns true only when a native read is due. */
bool rn_platform_thermal_should_read(int64_t now_us, int64_t & last_read_us);

/** Pure send decision. Escalations to status >= 2 need five seconds of
 *  consecutive reads at or above the pending level; lower statuses apply
 *  immediately. Two rules govern the gaps: an absent read sends nothing and
 *  leaves the pending state unchanged (a real read gap of more than
 *  2 * k_platform_thermal_interval_us restarts a pending escalation instead),
 *  while a read equal to the remembered status resets the pending escalation
 *  without sending. Absent never replaces the remembered status. */
rn_platform_thermal_send_decision rn_platform_thermal_should_send(
    int64_t now_us, int32_t native_status, int32_t remembered_status,
    rn_platform_thermal_send_state state);

} // namespace rnllama
