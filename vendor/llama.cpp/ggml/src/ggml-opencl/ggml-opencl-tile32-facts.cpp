#include "ggml-opencl-tile32-facts.h"

#include "ggml-impl.h"

#include <atomic>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

// The model number has to be a whole token: "Adreno 7401" and "Adreno 1740" are other parts,
// and a substring match would send the leg to a device it was never measured on. Boundaries
// are any non-alphanumeric byte, so "Adreno(TM) 740" matches and "7401" does not.
static bool has_model_token(const char * s, const char * model) {
    const size_t n = strlen(model);
    for (const char * p = strstr(s, model); p != NULL; p = strstr(p + 1, model)) {
        if ((p == s || !isalnum((unsigned char) p[-1])) &&
            (p[n] == '\0' || !isalnum((unsigned char) p[n]))) {
            return true;
        }
    }
    return false;
}

// The driver clause is "E031.41." with the trailing dot on purpose: it accepts the
// E031.41.x patch line the S23 ships and rejects E031.45 (Adreno 619), whose compiler
// this leg has never been measured on.
bool ggml_opencl_tile32_facts_ok(const struct ggml_opencl_tile32_facts * facts, size_t need_bytes,
                                 bool unmeasured_ok, const char ** reason) {
    if (reason) {
        *reason = "";
    }
    if (!facts || !facts->device_name || !facts->device_version || !facts->driver_version) {
        if (reason) {
            *reason = "missing device facts";
        }
        return false;
    }
    if (!facts->adreno) {
        if (reason) {
            *reason = "not an Adreno GPU";
        }
        return false;
    }
    const bool model_ok  = has_model_token(facts->device_name, "740") ||
                           has_model_token(facts->device_version, "740");
    const bool driver_ok = strstr(facts->driver_version, "E031.41.") != NULL;
    if (!unmeasured_ok && !model_ok) {
        if (reason) {
            *reason = "CL_DEVICE_NAME/CL_DEVICE_VERSION has no 740 model token";
        }
        return false;
    }
    if (!unmeasured_ok && !driver_ok) {
        if (reason) {
            *reason = "CL_DRIVER_VERSION is not on the E031.41. line";
        }
        return false;
    }
    // Only reachable when unmeasured_ok waived the two measured-on clauses above; everything
    // from here on refuses regardless of the override.
    const bool unmeasured_admit = !model_ok || !driver_ok;
    if (!facts->ext_dmabuf_host_ptr || !facts->ext_host_ptr) {
        if (reason) {
            *reason = "missing cl_qcom_dmabuf_host_ptr / cl_qcom_ext_host_ptr";
        }
        return false;
    }
    if (need_bytes > facts->max_alloc_size) {
        if (reason) {
            *reason = "TILE32 buffer above CL_DEVICE_MAX_MEM_ALLOC_SIZE";
        }
        return false;
    }
    if (unmeasured_admit) {
        // Only reachable because unmeasured_ok waived one of the two measured-on clauses
        // above: say what happened loudly, once per process.
        static std::atomic<bool> warned{false};
        bool expected = false;
        if (warned.compare_exchange_strong(expected, true)) {
            GGML_LOG_WARN("TILE32 gate: unmeasured-device waiver (lab build) admits %s, %s, %s\n",
                          facts->device_name, facts->device_version, facts->driver_version);
        }
    }
    return true;
}

// The env lives here, next to the gate it feeds, and only the callers read it so the pure
// facts check stays a function of its arguments the host test can drive without touching
// process env. The getenv branch exists only in lab builds: the shipped binding builds
// with GGML_OPENCL_TILE32_LAB off, so R2 - every leg off on unmeasured phones - cannot be
// bypassed by an env var.
bool ggml_opencl_tile32_unmeasured_override(void) {
#ifdef GGML_OPENCL_TILE32_LAB
    const char * v = getenv("GGML_OPENCL_TILE32_UNMEASURED");
    return v != NULL && strcmp(v, "1") == 0;
#else
    return false;
#endif
}
