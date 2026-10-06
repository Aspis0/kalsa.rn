// The one-copy capability table (FABLE (c)): the pure row match over device
// facts, plus the in-process fact readers that feed it. The table is the only
// allow-list for the one-model three-leg load; everything not on it keeps
// today's loads byte for byte.

#include "rn-legs-table.h"

#include <ggml-backend.h>

#include <cctype>
#include <cstring>
#include <vector>

#if defined(__ANDROID__)
#include <sys/auxv.h>
#include <sys/hwcap.h>
#include <sys/system_properties.h>

#if defined(GGML_USE_OPENCL)
#include <CL/cl.h>
#endif
#endif

namespace rnllama {

namespace {

// The model number has to be a whole token ("Adreno 7401" is another part):
// boundaries are any non-alphanumeric byte, so "Adreno(TM) 740" matches and
// "1740" does not. Same rule as the engine's gate
// (ggml-opencl-tile32-facts.cpp has_model_token).
bool has_model_token(const std::string & s, const char * model) {
    const size_t n = std::strlen(model);
    for (size_t pos = s.find(model); pos != std::string::npos; pos = s.find(model, pos + 1)) {
        const bool left_ok = pos == 0 || !std::isalnum((unsigned char) s[pos - 1]);
        const bool right_ok = pos + n >= s.size() || !std::isalnum((unsigned char) s[pos + n]);
        if (left_ok && right_ok) {
            return true;
        }
    }
    return false;
}

// One validated row: exact SoC and Hexagon arch, the Adreno model as a whole
// token in the CL name/version strings, and the GPU driver compiler's
// major.minor with the trailing dot (accepts the E031.41.x patch line,
// rejects E031.45, and "E031.41." cannot match "E031.411"). A firmware
// update off this compiler line must demote to two-copy, because J(1) makes
// the NPU decode depend on the OpenCL HOST leg that gate validates.
struct rn_legs_row {
    const char * soc_model;
    const char * hexagon_arch;
    const char * gpu_model_token;
    const char * gpu_compiler;
};

const rn_legs_row k_rows[] = {
    { "SM8550", "Hexagon v73", "740", "E031.41." },
};

bool row_matches(const rn_legs_row & row, const rn_hw_facts & facts) {
    return facts.soc_model == row.soc_model &&
           facts.hexagon_arch == row.hexagon_arch &&
           (has_model_token(facts.gpu_name, row.gpu_model_token) ||
            has_model_token(facts.gpu_version, row.gpu_model_token)) &&
           facts.gpu_driver.find(row.gpu_compiler) != std::string::npos &&
           facts.dotprod;
}

} // namespace

rn_leg_set rn_legs_for(const rn_hw_facts & facts) {
    for (const rn_legs_row & row : k_rows) {
        if (row_matches(row, facts)) {
            // All three legs or nothing (J(1)/R2): the NPU decode on the
            // shared copy reads the tied Q6_K output through the OpenCL HOST
            // leg, so one-copy without a validated GPU leg is not a state
            // this table can produce.
            return { true, true, true, true };
        }
    }
    return {};
}

rn_hw_facts rn_read_hw_facts() {
    rn_hw_facts facts;
#if defined(__ANDROID__)
    // SoC: unreadable stays empty, which never matches a row.
    char prop[92] = { 0 };
    if (__system_property_get("ro.soc.model", prop) > 0) {
        facts.soc_model = prop;
    }
    // The tile reader's sdot; also the CPU feature the shipped dotprod .so
    // was picked for.
    facts.dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;

    // Hexagon arch from the registered HTP device; the device registers only
    // when its skel loaded, and the description is the arch line
    // (ggml-hexagon.cpp "Hexagon v73").
    if (ggml_backend_reg_t htp = ggml_backend_reg_by_name("HTP")) {
        if (ggml_backend_reg_dev_count(htp) > 0) {
            facts.hexagon_arch = ggml_backend_dev_description(
                ggml_backend_reg_dev_get(htp, 0));
        }
    }

#if defined(GGML_USE_OPENCL)
    // The OpenCL strings, straight from the ICD the backend itself loads.
    // A build without the OpenCL backend cannot one-copy (the GPU leg IS an
    // OpenCL leg), so its facts stay empty.
    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) == CL_SUCCESS && n_platforms > 0) {
        std::vector<cl_platform_id> platforms(n_platforms);
        if (clGetPlatformIDs(n_platforms, platforms.data(), nullptr) == CL_SUCCESS) {
            for (cl_platform_id platform : platforms) {
                cl_uint n_devices = 0;
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr,
                                   &n_devices) != CL_SUCCESS || n_devices == 0) {
                    continue;
                }
                cl_device_id device = nullptr;
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device,
                                   nullptr) != CL_SUCCESS) {
                    continue;
                }
                const auto read_str = [&](cl_device_info what) {
                    size_t size = 0;
                    if (clGetDeviceInfo(device, what, 0, nullptr, &size) != CL_SUCCESS ||
                        size == 0) {
                        return std::string();
                    }
                    std::string value(size - 1, '\0');
                    if (clGetDeviceInfo(device, what, size, value.data(), nullptr) !=
                        CL_SUCCESS) {
                        return std::string();
                    }
                    return value;
                };
                facts.gpu_name = read_str(CL_DEVICE_NAME);
                facts.gpu_version = read_str(CL_DEVICE_VERSION);
                facts.gpu_driver = read_str(CL_DRIVER_VERSION);
                break;
            }
        }
    }
#endif // GGML_USE_OPENCL
#endif // __ANDROID__
    return facts;
}

} // namespace rnllama
