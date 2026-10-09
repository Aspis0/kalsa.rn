#include "ggml-opencl-tile32-facts.h"

#include "ggml-impl.h"

#include <atomic>
#include <ctype.h>
#include <stdio.h>
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

struct measured_pair {
    const char * model;
    const char * driver_line;
};

// These are the same-device pairs measured for this leg. Xiaomi 14 cell 74 passed every
// one-copy leg (E5 4000 GEMV calls, worst 2.218e-05 x RMS); QDC SM8650 cell 72 reported
// identical KL (~1e-5).
// Evidence: scratchpad/agents/xiaomi-merge-b11514-legs-74/REPORT.md,
// scratchpad/agents/qdc-merge-b11514-legs-72/REPORT.md, and
// scratchpad/agents/xiaomi-legheat-76/raw/legheat_gpu_log.grep (Adreno 750,
// Compiler E031.45.02.25). Each trailing dot admits that line's patch versions while
// keeping E031.41. distinct from E031.410 and E031.45. distinct from E031.450.
static const measured_pair MEASURED_PAIRS[] = {
    { "740", "E031.41." },
    { "750", "E031.45." },
};

static const char * find_adreno_model_token(const char * s, size_t * length) {
    for (const char * match = strstr(s, "Adreno"); match != NULL; match = strstr(match + 6, "Adreno")) {
        const char * p = match + 6;
        while (*p && !isdigit((unsigned char) *p)) {
            p++;
        }
        if (*p) {
            const char * end = p;
            while (isdigit((unsigned char) *end)) {
                end++;
            }
            *length = (size_t) (end - p);
            return p;
        }
    }
    return NULL;
}

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
    bool measured_ok = false;
    for (const measured_pair & pair : MEASURED_PAIRS) {
        const bool model_ok = has_model_token(facts->device_name, pair.model) ||
                              has_model_token(facts->device_version, pair.model);
        if (model_ok && strstr(facts->driver_version, pair.driver_line) != NULL) {
            measured_ok = true;
            break;
        }
    }
    if (!unmeasured_ok && !measured_ok) {
        if (reason) {
            size_t model_length = 0;
            const char * model = find_adreno_model_token(facts->device_name, &model_length);
            if (!model) {
                model = find_adreno_model_token(facts->device_version, &model_length);
            }
            if (!model) {
                model = "unknown";
                model_length = strlen(model);
            }
            static thread_local char detail[256];
            snprintf(detail, sizeof(detail), "no measured TILE32 pair for Adreno model token %.*s "
                     "on driver line %s", (int) model_length, model,
                     facts->driver_version);
            *reason = detail;
        }
        return false;
    }
    // Only reachable when unmeasured_ok waived the two measured-on clauses above; everything
    // from here on refuses regardless of the override.
    const bool unmeasured_admit = !measured_ok;
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
