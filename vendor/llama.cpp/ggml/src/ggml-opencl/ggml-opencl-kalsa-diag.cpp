// Match the OpenCL version setup of ggml-opencl.cpp and cl-program-cache.cpp, so
// every OpenCL translation unit in this backend sees the same API level. This
// must come before any include that reaches CL/cl.h: cl_version.h is
// include-guarded and only the first inclusion in a TU wins, so a later define
// would be ignored (or worse, silently overridden).
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "ggml-opencl-kalsa-diag.h"

// The relocated helpers need only GGML_LOG_INFO from the backend headers; the
// ggml_tensor / ggml_nbytes / ggml_type_name / ggml_type stack they were
// originally surrounded by is not used here. OpenCL declarations come from
// CL/cl.h, which the header above already includes.
#include "ggml-impl.h"

#include <CL/cl.h>
#include <string.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <set>

// KALSA production smoke (GGML_OPENCL_LOG_KERNELS=1): log each unique dispatched
// kernel name once (a set, not per-call spam) to stderr via clGetKernelInfo. Called
// in enqueue_ndrange_kernel (compute) AND at the trans4 convert/restore kernel
// picks (set_tensor/get_tensor dispatch bypasses enqueue_ndrange_kernel) so the
// real engine's full dispatch set -- including the convert/restore path -- can be
// audited for trans4 paths.
void ggml_opencl_log_kernel_once(cl_kernel kernel) {
    if (ggml_opencl_env_value_enabled("GGML_OPENCL_LOG_KERNELS")) {
        static std::set<std::string> seen_kernels;
        char kname[512] = {0};
        if (clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kname), kname, nullptr) == CL_SUCCESS) {
            if (seen_kernels.insert(std::string(kname)).second) {
                fprintf(stderr, "GGML_OPENCL_LOG_KERNELS: %s\n", kname);
                fflush(stderr);
            }
        }
    }
}

// Value-parsed env flags: "1" on; "0" / "" / "false" (any case) off; any other
// non-empty string on. Sibling hooks GGML_OPENCL_HOST_REPACK / GGML_OPENCL_DUMP_Q
// remain presence-only (see F4) and are not routed through this helper.
bool ggml_opencl_env_value_enabled(const char * name) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return false;
    }
    if (e[0] == '0' && e[1] == '\0') {
        return false;
    }
    char buf[8];
    memset(buf, 0, sizeof(buf));
    for (size_t i = 0; i < sizeof(buf) - 1 && e[i]; ++i) {
        const unsigned char c = (unsigned char) e[i];
        buf[i] = (char) ((c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c);
    }
    if (strcmp(buf, "false") == 0) {
        return false;
    }
    return true;
}

bool ggml_opencl_oracle_enabled(void) {
    return ggml_opencl_env_value_enabled("GGML_OPENCL_ORACLE");
}

// GGML_OPENCL_LOG_ALLOC=1 — value-parsed. Print every GPU alloc to stderr and
// fflush immediately so a driver reboot still leaves the last line on the
// PC-side adb stream. Running total is created-bytes (not live); staging
// that is later released still counts, which is what we want for a cliff.
bool ggml_opencl_log_alloc_enabled(void) {
    return ggml_opencl_env_value_enabled("GGML_OPENCL_LOG_ALLOC");
}

static uint64_t g_opencl_alloc_sum = 0;

void ggml_opencl_log_alloc(const char * kind, size_t bytes, cl_int err) {
    if (!ggml_opencl_log_alloc_enabled()) {
        return;
    }
    g_opencl_alloc_sum += (uint64_t) bytes;
    fprintf(stderr,
            "OPENCL_ALLOC{kind=%s,bytes=%zu,mib=%.2f,sum_mib=%.2f,err=%d}\n",
            kind, bytes, bytes / 1048576.0, g_opencl_alloc_sum / 1048576.0, (int) err);
    fflush(stderr);
}

void ggml_opencl_log_alloc_note(const char * msg) {
    if (!ggml_opencl_log_alloc_enabled()) {
        return;
    }
    fprintf(stderr, "OPENCL_ALLOC{note=%s,sum_mib=%.2f}\n", msg, g_opencl_alloc_sum / 1048576.0);
    fflush(stderr);
}

bool ggml_opencl_gemv_audit_enabled(void) {
    return ggml_opencl_env_value_enabled("GGML_OPENCL_GEMV_AUDIT");
}

int ggml_opencl_env_int_clamped(const char * name, int def, int lo, int hi) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return def;
    }
    int v = def;
    if (sscanf(e, "%d", &v) != 1) {
        return def;
    }
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

size_t ggml_opencl_gemv_audit_stash_cap_bytes(void) {
    const int mb = ggml_opencl_env_int_clamped("GGML_OPENCL_GEMV_AUDIT_STASH_MB", 512, 0, 4096);
    return (size_t) mb * 1024ull * 1024ull;
}

static int  g_gemv_audit_n_overlap_pairs = 0;
static bool g_gemv_audit_overlap_logged  = false;

uint64_t ggml_opencl_gemv_fnv1a64(const void * p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) b[i];
        h *= 1099511628211ull;
    }
    return h;
}

void ggml_opencl_gemv_ends4(const float * f, size_t n_elem, float first4[4], float last4[4]) {
    for (int i = 0; i < 4; ++i) {
        first4[i] = (n_elem > (size_t) i) ? f[i] : 0.f;
        last4[i]  = 0.f;
    }
    if (n_elem >= 4) {
        memcpy(last4, f + (n_elem - 4), 4 * sizeof(float));
    } else if (n_elem > 0) {
        memcpy(last4, f, n_elem * sizeof(float));
    }
}

bool ggml_opencl_gemv_range_overlap(size_t a0, size_t a1, size_t b0, size_t b1) {
    return a0 < b1 && b0 < a1;
}

void ggml_opencl_gemv_emit_overlap_map(void) {
    if (g_gemv_audit_overlap_logged || !ggml_opencl_gemv_audit_enabled()) {
        return;
    }
    if (g_gemv_audit_dst_snaps.empty()) {
        return;
    }
    g_gemv_audit_overlap_logged = true;
    int n_pairs = 0;
    int n_live = 0;
    const size_t n = g_gemv_audit_dst_snaps.size();
    for (size_t i = 0; i < n; ++i) {
        if (!g_gemv_audit_dst_snaps[i].consumed) {
            n_live += 1;
        }
        for (size_t j = i + 1; j < n; ++j) {
            const ggml_opencl_gemv_dst_snap & a = g_gemv_audit_dst_snaps[i];
            const ggml_opencl_gemv_dst_snap & b = g_gemv_audit_dst_snaps[j];
            if (a.consumed || b.consumed) {
                continue;
            }
            if (a.mem != b.mem) {
                continue;
            }
            if (!ggml_opencl_gemv_range_overlap(a.off, a.off + a.size, b.off, b.off + b.size)) {
                continue;
            }
            n_pairs += 1;
            if (n_pairs <= 16) {
                GGML_LOG_INFO(
                    "OVERLAP_PAIR{case_a=%d, case_b=%d, off_a=%zu, size_a=%zu, off_b=%zu, size_b=%zu}\n",
                    a.case_id, b.case_id, a.off, a.size, b.off, b.size);
            }
        }
    }
    GGML_LOG_INFO(
        "OVERLAP_MAP{n_overlaps=%d, n_snaps=%zu, n_live=%d, n_post_enq=%d, n_post_bind=%d}\n",
        n_pairs, n, n_live, g_gemv_audit_n_post_enq, g_gemv_audit_n_post_bind);
}

void ggml_opencl_gemv_register_dst_snap(
        int case_id, cl_mem mem, size_t off, size_t size,
        const float * got, size_t n_elem) {
    if (!ggml_opencl_gemv_audit_enabled() || mem == nullptr || got == nullptr || size == 0) {
        return;
    }
    if (g_gemv_audit_dst_snaps.size() >= 64) {
        return;
    }
    for (size_t i = 0; i < g_gemv_audit_dst_snaps.size(); ++i) {
        const ggml_opencl_gemv_dst_snap & e = g_gemv_audit_dst_snaps[i];
        if (e.consumed) {
            continue;
        }
        if (e.mem == mem &&
            ggml_opencl_gemv_range_overlap(e.off, e.off + e.size, off, off + size)) {
            g_gemv_audit_n_overlap_pairs += 1;
        }
    }
    ggml_opencl_gemv_dst_snap s;
    s.mem = mem;
    s.off = off;
    s.size = size;
    s.case_id = case_id;
    s.ck = ggml_opencl_gemv_fnv1a64(got, size);
    ggml_opencl_gemv_ends4(got, n_elem, s.first4, s.last4);
    g_gemv_audit_dst_snaps.push_back(s);
}

// DST_READBACK outcomes when snap offset == get offset (this vehicle:
// offsetd == extra->offset + view_offs + 0). Both the production get and the
// hook re-read hit the same bytes, so (raw_match=1, returned_match=0) is
// STRUCTURALLY UNREACHABLE. Reachable:
//   (1,1) memory intact and caller received the audited bytes
//   (0,0) memory changed between the audit snapshot and get_tensor
//   (0,1) changed in the microsecond window between get's read and the re-read
// Offset / size divergence is not a match pair: it shows as get_off/snap_off
// (and get_size/snap_size) with returned_match=-2 when the get range does not
// contain the snap. first4/last4 fingerprint checksum disagreements.
void ggml_opencl_gemv_check_dst_readback(
        cl_command_queue queue, cl_mem mem, size_t get_off, size_t get_size,
        const void * returned, size_t returned_size) {
    if (!ggml_opencl_gemv_audit_enabled() || mem == nullptr || queue == nullptr) {
        return;
    }
    {
        int n_live = 0;
        for (size_t i = 0; i < g_gemv_audit_dst_snaps.size(); ++i) {
            if (!g_gemv_audit_dst_snaps[i].consumed) {
                n_live += 1;
            }
        }
        if (n_live >= 2) {
            ggml_opencl_gemv_emit_overlap_map();
        }
    }
    for (size_t si = 0; si < g_gemv_audit_dst_snaps.size(); ++si) {
        ggml_opencl_gemv_dst_snap & s = g_gemv_audit_dst_snaps[si];
        if (s.consumed) {
            continue;
        }
        if (s.mem != mem) {
            continue;
        }
        if (!ggml_opencl_gemv_range_overlap(s.off, s.off + s.size, get_off, get_off + get_size)) {
            continue;
        }
        std::vector<uint8_t> now(s.size);
        const cl_int err = clEnqueueReadBuffer(queue, mem, CL_TRUE, s.off, s.size,
                                               now.data(), 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            GGML_LOG_INFO(
                "DST_READBACK{case=%d, raw_match=-1, returned_match=-1, skipped=read_err, err=%d, "
                "snap_off=%zu, snap_size=%zu}\n",
                s.case_id, err, s.off, s.size);
            s.consumed = true;
            continue;
        }
        const uint64_t raw_now_ck = ggml_opencl_gemv_fnv1a64(now.data(), s.size);
        const int raw_match = (raw_now_ck == s.ck) ? 1 : 0;

        int returned_match = -2;
        uint64_t returned_ck = 0;
        const uint8_t * rslice = nullptr;
        if (returned != nullptr && returned_size > 0) {
            const uint8_t * r = (const uint8_t *) returned;
            if (get_off == s.off && get_size == s.size && returned_size == s.size) {
                rslice = r;
                returned_ck = ggml_opencl_gemv_fnv1a64(r, s.size);
                returned_match = (returned_ck == s.ck) ? 1 : 0;
            } else if (get_off <= s.off &&
                       (get_off + get_size) >= (s.off + s.size) &&
                       returned_size >= (s.off - get_off) + s.size) {
                rslice = r + (s.off - get_off);
                returned_ck = ggml_opencl_gemv_fnv1a64(rslice, s.size);
                returned_match = (returned_ck == s.ck) ? 1 : 0;
            } else {
                returned_ck = ggml_opencl_gemv_fnv1a64(r, returned_size);
                returned_match = -2;
            }
        }

        float raw_first[4], raw_last[4], ret_first[4], ret_last[4];
        ggml_opencl_gemv_ends4((const float *) now.data(), s.size / sizeof(float), raw_first, raw_last);
        if (rslice != nullptr) {
            ggml_opencl_gemv_ends4((const float *) rslice, s.size / sizeof(float), ret_first, ret_last);
        } else {
            memset(ret_first, 0, sizeof(ret_first));
            memset(ret_last, 0, sizeof(ret_last));
        }

        GGML_LOG_INFO(
            "DST_READBACK{case=%d, raw_match=%d, returned_match=%d, "
            "snap_ck=%016llx, raw_now_ck=%016llx, returned_ck=%016llx, "
            "get_off=%zu, get_size=%zu, snap_off=%zu, snap_size=%zu, "
            "n_snaps=%zu, n_overlap_pairs=%d, n_post_enq=%d, n_post_bind=%d, "
            "first4_snap=%.8g:%.8g:%.8g:%.8g, first4_raw=%.8g:%.8g:%.8g:%.8g, "
            "first4_ret=%.8g:%.8g:%.8g:%.8g, "
            "last4_snap=%.8g:%.8g:%.8g:%.8g, last4_raw=%.8g:%.8g:%.8g:%.8g, "
            "last4_ret=%.8g:%.8g:%.8g:%.8g}\n",
            s.case_id, raw_match, returned_match,
            (unsigned long long) s.ck, (unsigned long long) raw_now_ck, (unsigned long long) returned_ck,
            get_off, get_size, s.off, s.size,
            g_gemv_audit_dst_snaps.size(), g_gemv_audit_n_overlap_pairs,
            g_gemv_audit_n_post_enq, g_gemv_audit_n_post_bind,
            s.first4[0], s.first4[1], s.first4[2], s.first4[3],
            raw_first[0], raw_first[1], raw_first[2], raw_first[3],
            ret_first[0], ret_first[1], ret_first[2], ret_first[3],
            s.last4[0], s.last4[1], s.last4[2], s.last4[3],
            raw_last[0], raw_last[1], raw_last[2], raw_last[3],
            ret_last[0], ret_last[1], ret_last[2], ret_last[3]);
        s.consumed = true;
    }
}

bool ggml_opencl_qk23_oracle_enabled(void) {
    return ggml_opencl_env_value_enabled("GGML_OPENCL_QK23_ORACLE");
}

bool ggml_opencl_gemv_audit_ratio_near(double r, double v) {
    const double den = std::fabs(v) > 1e-12 ? std::fabs(v) : 1.0;
    return std::fabs(r - v) / den < 0.08;
}
