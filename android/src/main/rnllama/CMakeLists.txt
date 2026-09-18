cmake_minimum_required(VERSION 3.12)

project(rnllama)

option(HEXAGON_SDK_ROOT "Hexagon SDK root directory" OFF)
option(HEXAGON_TOOLS_ROOT "Hexagon SDK toolchain directory" OFF)

find_package(Python3 REQUIRED)
set(CMAKE_CXX_STANDARD 17)
# Use CMAKE_CURRENT_SOURCE_DIR to work correctly both standalone and as subdirectory
set(RNLLAMA_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cpp)

include_directories(
    ${RNLLAMA_LIB_DIR}
    ${RNLLAMA_LIB_DIR}/common
    ${RNLLAMA_LIB_DIR}/common/jinja
    ${RNLLAMA_LIB_DIR}/ggml-cpu
    # vendored sha256.c includes "rotate-bits/rotate-bits.h" from the hash root
    ${RNLLAMA_LIB_DIR}/hash
    ${RNLLAMA_LIB_DIR}/tools/mtmd
)

# Collect source files using glob patterns
file(GLOB MODEL_FILES CONFIGURE_DEPENDS ${RNLLAMA_LIB_DIR}/models/*.cpp)
file(GLOB GGML_CPU_C_FILES CONFIGURE_DEPENDS ${RNLLAMA_LIB_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES CONFIGURE_DEPENDS ${RNLLAMA_LIB_DIR}/ggml-cpu/*.cpp)
file(GLOB GGML_CPU_AMX_FILES CONFIGURE_DEPENDS ${RNLLAMA_LIB_DIR}/ggml-cpu/amx/*.cpp)
file(GLOB LLAMA_FILES CONFIGURE_DEPENDS ${RNLLAMA_LIB_DIR}/llama*.cpp)
file(GLOB MTMD_FILES CONFIGURE_DEPENDS ${RNLLAMA_LIB_DIR}/tools/mtmd/*.cpp)
file(GLOB MTMD_MODEL_FILES CONFIGURE_DEPENDS ${RNLLAMA_LIB_DIR}/tools/mtmd/models/*.cpp)
file(GLOB COMMON_FILES CONFIGURE_DEPENDS ${RNLLAMA_LIB_DIR}/common/*.cpp)
file(GLOB JINJA_FILES CONFIGURE_DEPENDS ${RNLLAMA_LIB_DIR}/common/jinja/*.cpp)

# Kalsa's MoE streamer is optional so llama.rn can still be consumed from npm alone.
get_filename_component(BMOE_ROOT "${RNLLAMA_LIB_DIR}/../../../native/bmoe" ABSOLUTE)
set(BMOE_SOURCE_FILES)
if (EXISTS "${BMOE_ROOT}")
    file(GLOB BMOE_IO_FILES CONFIGURE_DEPENDS "${BMOE_ROOT}/src/io/*.cpp")
    file(GLOB BMOE_MOE_FILES CONFIGURE_DEPENDS "${BMOE_ROOT}/src/moe/*.cpp")
    file(GLOB BMOE_RN_FILES CONFIGURE_DEPENDS "${BMOE_ROOT}/rn/bmoe_stream.cpp")
    set(BMOE_SOURCE_FILES
        ${BMOE_ROOT}/src/config.cpp
        ${BMOE_IO_FILES}
        ${BMOE_MOE_FILES}
        ${BMOE_RN_FILES}
    )

    # The vendored engine uses upstream ggml names while llama.rn exports lm_* symbols.
    set(BMOE_COMPAT_HEADER "${BMOE_ROOT}/bmoe_lmggml_compat.h")
    set_source_files_properties(
        ${BMOE_SOURCE_FILES}
        PROPERTIES
        COMPILE_OPTIONS "-include;${BMOE_COMPAT_HEADER}"
        COMPILE_DEFINITIONS BMOE_HAVE_EXPERT_READY_HOOK
    )
endif()

# Source files for rnllama library (excluding jni.cpp and jni-utils)
set(
    RNLLAMA_SOURCE_FILES
    # Core GGML files
    ${RNLLAMA_LIB_DIR}/ggml.c
    ${RNLLAMA_LIB_DIR}/ggml-alloc.c
    ${RNLLAMA_LIB_DIR}/ggml-backend.cpp
    ${RNLLAMA_LIB_DIR}/ggml-backend-dl.cpp
    ${RNLLAMA_LIB_DIR}/ggml-backend-meta.cpp
    ${RNLLAMA_LIB_DIR}/ggml-backend-reg.cpp
    ${RNLLAMA_LIB_DIR}/ggml-opt.cpp
    ${RNLLAMA_LIB_DIR}/ggml-threading.cpp
    ${RNLLAMA_LIB_DIR}/ggml-quants.c
    ${RNLLAMA_LIB_DIR}/gguf.cpp

    # GGML CPU files (globbed)
    ${GGML_CPU_C_FILES}
    ${GGML_CPU_CPP_FILES}
    ${GGML_CPU_AMX_FILES}

    # Llama files (globbed)
    ${LLAMA_FILES}
    ${RNLLAMA_LIB_DIR}/unicode-data.cpp
    ${RNLLAMA_LIB_DIR}/unicode.cpp

    # Common utilities
    ${COMMON_FILES}

    ${RNLLAMA_LIB_DIR}/anyascii.c

    # mtmd-helper.cpp keys its bitmap cache with hash_sha256_hex
    # (hash/hash.cpp), which calls sha256_hash (hash/sha256/sha256.c); neither
    # was in the target, so the arm64 link ended in "undefined symbol:
    # hash_sha256_hex". The rest of the vendored hash tree (sha1, xxhash)
    # ships with no caller among the compiled sources and stays out.
    ${RNLLAMA_LIB_DIR}/hash/hash.cpp
    ${RNLLAMA_LIB_DIR}/hash/sha256/sha256.c

    # Jinja template engine
    ${JINJA_FILES}

    # Multimodal support (globbed)
    ${MTMD_MODEL_FILES}
    ${MTMD_FILES}

    # React Native llama APIs
    ${RNLLAMA_LIB_DIR}/rn-llama.cpp
    ${RNLLAMA_LIB_DIR}/rn-governor.cpp
    ${RNLLAMA_LIB_DIR}/rn-governor-params.cpp
    ${RNLLAMA_LIB_DIR}/rn-completion.cpp
    ${RNLLAMA_LIB_DIR}/rn-tts.cpp
    ${RNLLAMA_LIB_DIR}/rn-slot.cpp
    ${RNLLAMA_LIB_DIR}/rn-slot-manager.cpp

    # Model implementations (globbed)
    ${MODEL_FILES}

    # Kalsa MoE expert streaming (optional, outside node_modules)
    ${BMOE_SOURCE_FILES}
)

find_library(LOG_LIB log)

# Check if building standalone (for prebuilts) or as subdirectory (for source build)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(RNLLAMA_STANDALONE_BUILD ON)
    message(STATUS "Building rnllama as standalone (for prebuilt distribution)")
else()
    set(RNLLAMA_STANDALONE_BUILD OFF)
    message(STATUS "Building rnllama as subdirectory (will be linked to JNI wrapper)")
endif()

function(build_rnllama_library target_name arch cpu_flags)
    set(ENABLE_OPENCL OFF)
    if (${target_name} MATCHES ".*_opencl$")
        set(ENABLE_OPENCL ON)
    endif ()

    set(ENABLE_HEXAGON OFF)
    if (${target_name} MATCHES ".*_hexagon.*")
        set(ENABLE_HEXAGON ON)
    endif ()

    if (NOT ${arch} STREQUAL "generic")
        set(SOURCE_FILES_ARCH
            ${RNLLAMA_LIB_DIR}/ggml-cpu/arch/${arch}/quants.c
            ${RNLLAMA_LIB_DIR}/ggml-cpu/arch/${arch}/repack.cpp
        )
    endif ()

    # Build as shared library for prebuilt distribution
    add_library(
        ${target_name}
        SHARED
        ${RNLLAMA_SOURCE_FILES}
        ${SOURCE_FILES_ARCH}
    )

    if (BMOE_SOURCE_FILES)
        target_include_directories(${target_name} PRIVATE
            ${BMOE_ROOT}/include
            ${BMOE_ROOT}/src
            ${BMOE_ROOT}/rn
        )
    endif()

    target_link_libraries(${target_name} PRIVATE ${LOG_LIB} android)

    if (${arch} STREQUAL "generic")
        target_compile_options(${target_name} PRIVATE -DLM_GGML_CPU_GENERIC)
    endif ()

    target_compile_options(${target_name} PRIVATE -DLM_GGML_USE_CPU -DLM_GGML_USE_CPU_REPACK -pthread ${cpu_flags} -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE)

    if (CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DRNLLAMA_ANDROID_ENABLE_LOGGING)
    endif ()

    # Hexagon backend support
    # Note: Hexagon requires building DSP libraries with Hexagon SDK toolchain
    # This is a complex process that needs to be done separately
    if (ENABLE_HEXAGON)
        # Hexagon SDK is optional — if not present, build the target without Hexagon support
        set(HEXAGON_SDK_AVAILABLE OFF)
        if (HEXAGON_SDK_ROOT AND HEXAGON_TOOLS_ROOT)
            # Verify libcdsprpc.so exists
            set(CDSPRPC_LIB "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so")
            if (EXISTS ${CDSPRPC_LIB})
                set(HEXAGON_SDK_AVAILABLE ON)
                message(STATUS "Hexagon SDK verified and available")
            else()
                message(WARNING
                    "Hexagon SDK found but libcdsprpc.so is missing at ${CDSPRPC_LIB}. "
                    "Building ${target_name} without Hexagon support.")
            endif()
        else()
            message(STATUS "Hexagon SDK not found — building ${target_name} without Hexagon support.")
        endif()

        if (HEXAGON_SDK_AVAILABLE)
            # Add host-side hexagon sources
            target_sources(${target_name} PRIVATE
                ${RNLLAMA_LIB_DIR}/ggml-hexagon/ggml-hexagon.cpp
                ${RNLLAMA_LIB_DIR}/ggml-hexagon/htp-drv.cpp
            )

            # The Android host build consumes generated QAIC artifacts copied by
            # scripts/build-hexagon-htp.sh. Keep the failure mode explicit when
            # they have not been generated yet.
            set(HEXAGON_PREBUILT_HTP_DIR "${RNLLAMA_LIB_DIR}/ggml-hexagon/htp/v73/")
            set(HEXAGON_HOST_STUB "${HEXAGON_PREBUILT_HTP_DIR}/htp_iface_stub.c")
            set(HEXAGON_HOST_HEADER "${HEXAGON_PREBUILT_HTP_DIR}/htp_iface.h")

            if (NOT EXISTS ${HEXAGON_HOST_STUB} OR NOT EXISTS ${HEXAGON_HOST_HEADER})
                message(FATAL_ERROR
                    "Hexagon host QAIC artifacts are missing. "
                    "Run scripts/build-hexagon-htp.sh to generate ${HEXAGON_PREBUILT_HTP_DIR}.")
            endif()

            target_sources(${target_name} PRIVATE
                ${HEXAGON_HOST_STUB}
            )

            target_include_directories(${target_name} PRIVATE
                ${HEXAGON_SDK_ROOT}/incs
                ${HEXAGON_SDK_ROOT}/incs/stddef
                ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
                ${HEXAGON_SDK_ROOT}/utils/examples
                ${RNLLAMA_LIB_DIR}/ggml-hexagon
                ${RNLLAMA_LIB_DIR}/ggml-hexagon/htp
                ${HEXAGON_PREBUILT_HTP_DIR}
                ${CMAKE_CURRENT_BINARY_DIR}
            )

            # Link Hexagon SDK libraries
            # These are stub libraries that communicate with the DSP
            # libcdsprpc.so contains all needed symbols: remote_handle64_*, dspqueue_*, rpcmem_*
            # Note: CDSPRPC_LIB is already verified to exist above
            target_link_libraries(${target_name} PRIVATE ${CDSPRPC_LIB})
            message(STATUS "Linking Hexagon cdsprpc library: ${CDSPRPC_LIB}")

            target_compile_options(${target_name} PRIVATE
                -DLM_GGML_USE_HEXAGON
            )

            message(STATUS "Hexagon backend enabled for ${target_name}")
            if (NOT RNLLAMA_STANDALONE_BUILD)
                message(STATUS "  Note: DSP libraries (libggml-htp-*.so) must be built separately")
                message(STATUS "  Run: scripts/build-hexagon-htp.sh to build DSP libraries")
            endif()
        else()
            # Only show warning when building from source
            if (NOT RNLLAMA_STANDALONE_BUILD)
                message(WARNING
                    "Hexagon SDK not found. Hexagon backend will not be built.")
            endif()
        endif()
    endif ()

    # OpenCL only for the special target
    if (ENABLE_OPENCL)
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/OpenCL-Headers)

        # Exactly the set of kernels the fork ships, by construction: file(GLOB)
        # over cpp/ggml-opencl/kernels/*.cl, embedded via embed_kernel.py into the
        # <name>.cl.h files ggml-opencl.cpp #includes (181 .cl files at this pin;
        # its 183 #include lines hit two of them -- gemm_moe_q4_0_f32_ns,
        # gemv_moe_q4_0_f32_ns -- twice). The previous hand-written list named 156
        # and drifted 25 names behind the overlay, so the arm64 OpenCL target
        # failed at build time; the glob cannot drift, and a .cl.h include with no
        # matching .cl file is the include scan's failure, not configure's.
        file(GLOB GGML_OPENCL_KERNELS CONFIGURE_DEPENDS
            ${RNLLAMA_LIB_DIR}/ggml-opencl/kernels/*.cl)
        set(GGML_OPENCL_KERNEL_HEADERS "")

        foreach(kernel_path ${GGML_OPENCL_KERNELS})
            get_filename_component(kernel ${kernel_path} NAME_WE)
            set(input_file ${kernel_path})
            set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${kernel}.cl.h)
            message(STATUS "opencl: embedding kernel ${kernel}")
            add_custom_command(
                OUTPUT ${output_file}
                COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/..
                        ${Python3_EXECUTABLE} ${RNLLAMA_LIB_DIR}/ggml-opencl/kernels/embed_kernel.py
                        ${input_file} ${output_file}
                DEPENDS ${input_file} ${RNLLAMA_LIB_DIR}/ggml-opencl/kernels/embed_kernel.py
                COMMENT "Embedding OpenCL kernel: ${kernel}.cl"
            )

            list(APPEND GGML_OPENCL_KERNEL_HEADERS ${output_file})
        endforeach()

        set(OPENCL_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../bin/${ANDROID_ABI})
        set(OPENCL_STUB     ${OPENCL_STUB_DIR}/libOpenCL.so)

        if (EXISTS ${OPENCL_STUB})
            # Keep using -lOpenCL so we don't create a packagable .so target.
            target_link_directories(${target_name} PRIVATE ${OPENCL_STUB_DIR})
            target_link_libraries(${target_name} PRIVATE OpenCL)
        else()
            message(WARNING
                "OpenCL library not found. Please build with ./scripts/build-opencl.sh")
        endif()

        target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        target_sources(${target_name} PRIVATE
            ${RNLLAMA_LIB_DIR}/ggml-opencl/ggml-opencl.cpp
            # cl-program-cache.cpp backs the program cache ggml-opencl.cpp uses
            # (cl_program_cache_state program_cache, cl_program_cache_init /
            # try_load / try_save); leaving it out is a link-time failure.
            ${RNLLAMA_LIB_DIR}/ggml-opencl/cl-program-cache.cpp
            ${GGML_OPENCL_KERNEL_HEADERS}
        )
        target_compile_options(${target_name} PRIVATE
            -DLM_GGML_USE_OPENCL
            -DLM_GGML_OPENCL_USE_ADRENO_KERNELS
            -DLM_GGML_OPENCL_EMBED_KERNELS
            -DLM_GGML_OPENCL_SOA_Q
            -DLM_GGML_OPENCL_TARGET_VERSION=300
        )
    endif ()

    # Optimize for size and performance
    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${target_name} PRIVATE -flto)
endfunction()

# Default target (no specific CPU features)
build_rnllama_library("rnllama" "generic" "")

if (ANDROID_ABI AND ANDROID_ABI STREQUAL "arm64-v8a")
    # ARM64 targets
    build_rnllama_library("rnllama_v8" "arm" "-march=armv8-a")
    build_rnllama_library("rnllama_v8_2" "arm" "-march=armv8.2-a")
    build_rnllama_library("rnllama_v8_2_dotprod" "arm" "-march=armv8.2-a+dotprod")
    build_rnllama_library("rnllama_v8_2_i8mm" "arm" "-march=armv8.2-a+i8mm")
    build_rnllama_library("rnllama_v8_2_dotprod_i8mm" "arm" "-march=armv8.2-a+dotprod+i8mm")
    build_rnllama_library("rnllama_v8_2_dotprod_i8mm_hexagon_opencl" "arm" "-march=armv8.2-a+dotprod+i8mm")

elseif (ANDROID_ABI AND ANDROID_ABI STREQUAL "x86_64")
    # x86_64 target
    build_rnllama_library("rnllama_x86_64" "x86" "-march=x86-64;-mtune=generic;-msse4.2;-mpopcnt")

endif ()
