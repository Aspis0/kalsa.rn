#pragma once

#include <stddef.h>

// The R2 device facts for the OpenCL TILE32 weight leg (Amendment E4): an Adreno 740 on
// the E031.41 driver line exposing both QCOM ext-host-ptr extensions, with a
// CL_DEVICE_MAX_MEM_ALLOC_SIZE large enough for the buffer asked about. Deliberately free
// of OpenCL types so the host build (GGML_OPENCL=OFF) can test the gate on fake facts;
// the device-side caller fills the struct from the backend context.
struct ggml_opencl_tile32_facts {
    bool         adreno;              // backend_ctx->gpu_family == ADRENO
    const char * device_name;         // CL_DEVICE_NAME
    const char * device_version;      // CL_DEVICE_VERSION
    const char * driver_version;      // CL_DRIVER_VERSION, must contain "E031.41."
    bool         ext_dmabuf_host_ptr; // cl_qcom_dmabuf_host_ptr in CL_DEVICE_EXTENSIONS
    bool         ext_host_ptr;        // cl_qcom_ext_host_ptr in CL_DEVICE_EXTENSIONS
    size_t       max_alloc_size;      // CL_DEVICE_MAX_MEM_ALLOC_SIZE, bytes
};

// The model number may live in either string: the S23 answers CL_DEVICE_NAME
// "QUALCOMM Adreno(TM)" and puts the 740 in CL_DEVICE_VERSION ("OpenCL 3.0 Adreno(TM) 740").
//
// Returns true when every fact passes and max_alloc_size covers need_bytes; on a miss
// returns false and sets *reason to the failing clause. need_bytes is 0 before any TILE32
// buffer exists; the import re-asks with the dma-buf block it would have to place.
bool ggml_opencl_tile32_facts_ok(const struct ggml_opencl_tile32_facts * facts, size_t need_bytes, const char ** reason);
