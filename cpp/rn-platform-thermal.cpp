// Android's current thermal status, read natively.
//
// One responsibility: answer "what does the platform say now" on the scale
// the engine's thermo profile carries. The policy about when an answer
// reaches the engine lives in cpp/rn-governor.cpp (refresh_platform_thermal),
// which this TU deliberately knows nothing about.

#include "rn-platform-thermal.h"

#if defined(__ANDROID__)
#include <dlfcn.h>
#endif

namespace rnllama {

#if defined(__ANDROID__)
// The mapping is a compile-time fact of the NDK header, not a run-time
// convention: if AThermalStatus ever stops being the engine's ladder, this
// breaks the build instead of silently mistranslating a floor.
static_assert(k_platform_thermal_shutdown == ATHERMAL_STATUS_SHUTDOWN,
              "AThermalStatus is no longer 0..SHUTDOWN; the mapping must be revisited");
#endif

int32_t rn_map_platform_thermal_status(int32_t athermal_status) {
    if (athermal_status < 0 || athermal_status > k_platform_thermal_shutdown) {
        return k_platform_thermal_absent;
    }
    return athermal_status;
}

bool rn_platform_thermal_should_send(int64_t now_us, int64_t & last_read_us,
                                     int32_t native_status, int32_t remembered_status) {
    if (now_us - last_read_us < k_platform_thermal_interval_us) {
        return false;
    }
    last_read_us = now_us;
    return native_status != k_platform_thermal_absent && native_status != remembered_status;
}

#if !defined(__ANDROID__)

// Not Android: there is no platform status to read, and these definitions keep
// the governor that owns the reader portable to every build.
rn_platform_thermal::rn_platform_thermal() = default;
rn_platform_thermal::~rn_platform_thermal() = default;

int32_t rn_platform_thermal::status() const {
    return k_platform_thermal_absent;
}

#else

namespace {

using rn_thermal_acquire_fn = AThermalManager * (*)();
using rn_thermal_current_fn = AThermalStatus (*)(AThermalManager * manager);
using rn_thermal_release_fn = void (*)(AThermalManager * manager);

struct rn_thermal_api {
    rn_thermal_acquire_fn acquire = nullptr;
    rn_thermal_current_fn current = nullptr;
    rn_thermal_release_fn release = nullptr;

    bool complete() const {
        return acquire != nullptr && current != nullptr && release != nullptr;
    }
};

// API 30's thermal API, resolved once per process out of the resident
// libandroid.so instead of being linked. The library's minSdk is 23
// (android/gradle.properties RNLlama_minSdkVersion): a pre-30 libandroid
// exports none of the three entry points, so a linked reference would fail
// the load of librnllama_v8_*.so there. libandroid is a system library that
// is never unloaded in an app process, which is also why the handle is not
// closed - the table has to outlive every manager taken from it.
const rn_thermal_api & thermal_api() {
    static const rn_thermal_api api = [] {
        rn_thermal_api table;
        void * lib = ::dlopen("libandroid.so", RTLD_NOW | RTLD_NODELETE);
        if (lib == nullptr) {
            return table;
        }
        table.acquire = reinterpret_cast<rn_thermal_acquire_fn>(
            ::dlsym(lib, "AThermal_acquireManager"));
        table.current = reinterpret_cast<rn_thermal_current_fn>(
            ::dlsym(lib, "AThermal_getCurrentThermalStatus"));
        table.release = reinterpret_cast<rn_thermal_release_fn>(
            ::dlsym(lib, "AThermal_releaseManager"));
        return table.complete() ? table : rn_thermal_api{};
    }();
    return api;
}

} // namespace

rn_platform_thermal::rn_platform_thermal() {
    const rn_thermal_api & api = thermal_api();
    manager_ = api.complete() ? api.acquire() : nullptr;
}

rn_platform_thermal::~rn_platform_thermal() {
    const rn_thermal_api & api = thermal_api();
    if (manager_ != nullptr && api.complete()) {
        api.release(manager_);
    }
}

int32_t rn_platform_thermal::status() const {
    const rn_thermal_api & api = thermal_api();
    if (manager_ == nullptr || !api.complete()) {
        return k_platform_thermal_absent;
    }
    return rn_map_platform_thermal_status(api.current(manager_));
}

#endif // __ANDROID__

} // namespace rnllama
