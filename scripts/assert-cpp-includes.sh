#!/usr/bin/env bash
# Every local #include in the assembled engine must resolve to a file on disk,
# and every C++ TU the build compiles must parse.
#
# The sync copies the kalsallama tree through explicit copy lists
# (copy_llama_api / copy_common / copy_mtmd in scripts/sync-kalsallama.sh).
# Those lists do not learn: when the fork adds a file and an existing source
# starts including it, the file is simply absent and the NDK fails ~5 minutes
# into the APK build, far from the commit that caused it. That is how
#   common/peg-parser.h:3:10: fatal error: 'json.h' file not found
# reached CI on 2026-09-17 -- json.h/json.cpp, plus llama-kv-cache-dsa-iswa,
# llama-kv-cache-msa and llama-memory-hybrid-idx, all existed in the fork and
# none were on the lists.
#
# The scan below resolves every local `#include "..."` in seconds; the syntax
# pass then parses the C++ TUs the build actually compiles, with the NDK clang
# the APK build uses -- -fsyntax-only resolves the declarations that drift
# (e.g. common_context_seq_rm going static). The host clang accepts constructs
# the NDK rejects (JSIParams.cpp parsed on host and failed the arm64 NDK build
# on 2026-09-17), so the host compiler is only a fallback and the OK line
# names which one ran.
# Conditional backend includes (CUDA, Vulkan, ...) and build-generated headers
# are listed as known-absent; anything else is a missing copy-list entry.
# The rn-owned sources need the app repo's bmoe headers: without
# KALSA_BMOE_DIR they are skipped and the gate only passes as PARTIAL
# (KALSA_ALLOW_PARTIAL_GATE=1) -- verify refuses half coverage otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CPP="${1:-$ROOT/cpp}"
[ -d "$CPP" ] || { echo "[includes] missing $CPP -- run pin first" >&2; exit 1; }

python3 - "$CPP" <<'PY'
import os, re, sys, glob

cpp = sys.argv[1]
os.chdir(cpp)

# Absent for a reason: guarded by #ifdef for a backend we never build,
# emitted by the build, or living outside this repo.
KNOWN_ABSENT = {
    "ggml-cann.h", "ggml-cuda.h", "ggml-et.h", "ggml-openvino.h", "ggml-rpc.h",
    "ggml-sycl.h", "ggml-virtgpu.h", "ggml-vulkan.h", "ggml-webgpu.h",
    "ggml-zdnn.h", "ggml-zendnn.h",
    "bmoe_stream.h",            # lives in the app repo (native/bmoe/rn);
                                # included unconditionally by rn-llama.cpp
    "sheredom/subprocess.h",    # mtmd-helper.cpp, behind #ifdef MTMD_VIDEO
    "windows.h",                # ggml-cpu.c, behind #if defined(_WIN32)
    "unistd.h",                 # quoted POSIX header, ggml-hexagon/htp-drv.h
    "htp_iface.h",              # generated from the Hexagon IDL
    "kleidiai/kleidiai.h",      # optional CPU backend, build-flag guarded
    "llamafile/sgemm.h",        # optional CPU backend, build-flag guarded
    "spacemit/ime.h",           # optional CPU backend, build-flag guarded
    "half.hpp",                 # ggml-opencl.cpp:14135, inside an #if 0 block
    "HAP_farf.h",               # Hexagon SDK (DSP-side htp/*.c, built by
    "hexagon_protos.h",         # scripts/build-hexagon-htp.sh, never by the
    "hexagon_types.h",          # NDK or host clang)
    "hvx_hexagon_protos.h",     #
}
# The kernel headers that the Android build generates from kernels/*.cl
# (kernels/embed_kernel.py); a *different* x.cl.h include is a real failure.
GENERATED_KERNEL_HEADERS = {
    os.path.basename(k) + ".h" for k in glob.glob(os.path.join(cpp, "ggml-opencl/kernels/*.cl"))
}
SEARCH = ["", "common", "common/jinja", "ggml-cpu", "ggml-hexagon",
          "ggml-hexagon/htp", "tools/mtmd", "nlohmann"]

files = []
for pat in ("*.h", "*.cpp", "*.c",
            "common/*.h", "common/*.cpp",
            "common/jinja/*.h", "common/jinja/*.cpp",
            "models/*.cpp", "models/*.h",
            "ggml-cpu/*.c", "ggml-cpu/*.cpp", "ggml-cpu/*.h", "ggml-cpu/*/*.cpp",
            "ggml-cpu/arch/*/*.c",
            "ggml-metal/*", "ggml-hexagon/*.cpp", "ggml-hexagon/*.h",
            "ggml-hexagon/htp/*.c", "ggml-hexagon/htp/*/*.c",
            "ggml-opencl/*.cpp", "ggml-opencl/*.h",
            "jsi/*.h", "jsi/*.cpp",
            "tools/mtmd/*.h", "tools/mtmd/*.cpp",
            "tools/mtmd/models/*.cpp"):
    files += [f for f in glob.glob(pat) if os.path.isfile(f)]

missing = {}
for f in files:
    with open(f, errors="ignore") as fh:
        body = fh.read()
    for m in re.finditer(r'^\s*#\s*include\s+"([^"]+)"', body, re.M):
        inc = m.group(1)
        if inc in KNOWN_ABSENT:
            continue
        if inc.endswith(".cl.h"):
            # The OpenCL build generates <name>.cl.h from kernels/<name>.cl
            # (kernels/embed_kernel.py); a .cl.h include is only honest when
            # that .cl exists -- this is the check that would have caught the
            # hand-written CMake kernel list running 25 names behind the
            # overlay. A .cl.h shipped directly also passes (file-exists
            # branch below).
            if inc in GENERATED_KERNEL_HEADERS:
                continue
        here = os.path.join(os.path.dirname(f), inc)
        if os.path.exists(here) or any(os.path.exists(os.path.join(d, inc)) for d in SEARCH):
            continue
        missing.setdefault(inc, []).append(f)

if missing:
    print(f"[includes] FAIL: {len(missing)} unresolved include(s) in {cpp}")
    for inc, users in sorted(missing.items()):
        print(f"  {inc}  <- {', '.join(sorted(users)[:4])}")
        if inc.endswith(".cl.h"):
            print(f"    kernel include without a shipped .cl: add ggml-opencl/kernels/{inc[:-len('.cl.h')]}.cl")
    print("[includes] add the file to the copy lists in scripts/sync-kalsallama.sh,")
    print("[includes] then: scripts/sync-kalsallama.sh pin <sha>")
    raise SystemExit(1)

print(f"[includes] OK: every local include in {len(files)} files resolves")
PY

# rn-*.cpp include the bmoe stream port headers, which live in the app repo;
# point KALSA_BMOE_DIR at them to bring the rn-owned sources into the syntax
# pass. The jsi/*.cpp TUs need <jsi/jsi.h> and <ReactCommon/CallInvoker.h>;
# the react-native checkout of the same app repo provides them (the NDK build
# gets both from the ReactAndroid prefab). Without KALSA_BMOE_DIR (or without
# any usable clang, or without those react-native headers) the pass runs
# PARTIAL and the gate only succeeds with KALSA_ALLOW_PARTIAL_GATE=1.
#
# The TU list mirrors what the Android build compiles:
#   android/src/main/CMakeLists.txt      -> JNI_SOURCE_FILES (cpp/jsi/*.cpp)
#   android/src/main/rnllama/CMakeLists.txt -> RNLLAMA_SOURCE_FILES and its
#       file(GLOB ...) sets: common/*.cpp, common/jinja/*.cpp, models/*.cpp,
#       tools/mtmd/*.cpp + models/*.cpp, llama*.cpp, unicode*.cpp, gguf.cpp,
#       ggml-backend*.cpp, ggml-opt.cpp, ggml-threading.cpp,
#       ggml-cpu/*.cpp + amx/*.cpp + arch/{arm,x86}/repack.cpp, rn-*.cpp,
#       plus BMOE_SOURCE_FILES from the app repo (native/bmoe/{src/config.cpp,
#       src/io/*.cpp,src/moe/*.cpp,rn/bmoe_stream.cpp}) with the bmoe compile
#       shape: -I {include,src,rn}, -include bmoe_lmggml_compat.h and
#       -DBMOE_HAVE_EXPERT_READY_HOOK; the bmoe root is derived from
#       KALSA_BMOE_DIR (= <app>/native/bmoe/rn).
#
# Known non-parsed TUs (never silently): the ggml-hexagon host sources need
# the Hexagon SDK headers and -DLM_GGML_USE_HEXAGON (HEXAGON_SDK_ROOT).
# android/src/main/RNLlamaJSI.cpp needs <android/log.h> and the fbjni prefab
# headers -- the cpp/jsi TUs it links carry the drift risk. Everything else
# the build compiles is parsed: the .c sources (ggml.c, ggml-alloc.c,
# ggml-quants.c, anyascii.c, ggml-cpu/*.c, arch quants.c,
# ggml-hexagon/htp/**/*.c) by the include scan above, the OpenCL TUs
# (ggml-opencl.cpp, cl-program-cache.cpp) by the syntax pass against real
# kernel headers embedded from cpp/ggml-opencl/kernels/*.cl with the tree's
# own embed_kernel.py, plus third_party/OpenCL-Headers.
#
# llama.cpp's llama-version.h and ggml.c's ggml-version.h are generated into
# cpp/ by scripts/sync-kalsallama.sh from the pin's own .in templates (the
# Android build and the iOS podspec run no upstream CMake); the syntax pass
# resolves them from the tree like any other header.

# GNU mktemp rejects a -t template without X's (BSD accepts it): spell it so
# the gate also runs on a Linux runner.
SYNTAX_LOG="$(mktemp "${TMPDIR:-/tmp}/kalsa-syntax.XXXXXX")"
KERNEL_EMBED_DIR="$(mktemp -d)"
trap 'rm -f "$SYNTAX_LOG"; rm -rf "$KERNEL_EMBED_DIR"' EXIT

# Real kernel headers for the OpenCL syntax pass: embed every shipped .cl the
# way the build's custom command does (ggml-opencl/kernels/embed_kernel.py).
# The scan above guarantees every .cl.h include has its .cl; here we make the
# includes resolvable so ggml-opencl.cpp itself can be parsed.
for cl in "$CPP"/ggml-opencl/kernels/*.cl; do
  [ -f "$cl" ] || continue
  python3 "$CPP/ggml-opencl/kernels/embed_kernel.py" "$cl" \
    "$KERNEL_EMBED_DIR/$(basename "$cl").h"
done

# Compiler for the syntax pass: the NDK clang the APK build uses, so the pass
# sees what that build sees. The app declares no ndkVersion anywhere (its build
# takes AGP's default), so there is no number to read: set
# KALSA_GATE_NDK_VERSION when you know which one the build reported, and the
# gate will use that exact NDK or say it is missing. Env:
#   KALSA_GATE_NDK_VERSION   exact NDK version, looked up under
#                            $ANDROID_HOME/ndk and $ANDROID_SDK_ROOT/ndk
#   ANDROID_NDK_HOME / ANDROID_NDK_ROOT  NDK location (else the homebrew path)
#   KALSA_GATE_ANDROID_API   target API level (default 33 = minSdkVersion)
#   KALSA_GATE_SYNTAX_CXX    `host` forces the host clang++, or an explicit
#                            compiler path
#   KALSA_GATE_REQUIRE_NDK=1 a fallback is a failure, not a weaker pass
NDK_API="${KALSA_GATE_ANDROID_API:-33}"
NDK_VER="${KALSA_GATE_NDK_VERSION:-}"
# This repo declares the NDK its own Android build uses: android/gradle.properties
# RNLlama_ndkversion, read as `ndkVersion` in android/build.gradle. Follow it, so
# the gate and the build are the same compiler when that version is installed.
ENGINE_PIN=""
if [ -f "$ROOT/android/gradle.properties" ]; then
  ENGINE_PIN="$(sed -n 's/^RNLlama_ndkversion=//p' "$ROOT/android/gradle.properties" | tr -d '[:space:]')"
fi
[ -n "$NDK_VER" ] || NDK_VER="$ENGINE_PIN"
SYNTAX_CXX=""
SYNTAX_CXX_TAG=""
NDK_ROOT_USED=""
NDK_WHY="no NDK directory (ANDROID_NDK_HOME, ANDROID_NDK_ROOT, /opt/homebrew/share/android-ndk)"

ndk_try() {  # ndk_try <ndk root> -- sets SYNTAX_CXX when the wrapper is there
  [ -n "${1:-}" ] && [ -d "$1" ] || return 1
  local prebuilt
  for prebuilt in "$1"/toolchains/llvm/prebuilt/*; do
    # An unexpanded glob is a string, not a directory.
    [ -d "$prebuilt" ] || continue
    if [ -x "$prebuilt/bin/aarch64-linux-android${NDK_API}-clang++" ]; then
      SYNTAX_CXX="$prebuilt/bin/aarch64-linux-android${NDK_API}-clang++"
      NDK_ROOT_USED="$1"
      return 0
    fi
  done
  NDK_WHY="$1 has no aarch64-linux-android${NDK_API}-clang++"
  return 1
}

case "${KALSA_GATE_SYNTAX_CXX:-}" in
  host)
    NDK_WHY="KALSA_GATE_SYNTAX_CXX=host"
    ;;
  "")
    if [ -n "$NDK_VER" ]; then
      ndk_try "${ANDROID_HOME:-}/ndk/$NDK_VER" || ndk_try "${ANDROID_SDK_ROOT:-}/ndk/$NDK_VER" || true
      [ -n "$SYNTAX_CXX" ] || NDK_WHY="NDK $NDK_VER (this repo's pin) is not installed under \$ANDROID_HOME/ndk or \$ANDROID_SDK_ROOT/ndk"
    fi
    if [ -z "$SYNTAX_CXX" ] && [ "${KALSA_GATE_REQUIRE_NDK:-}" != "1" ]; then
      ndk_try "${ANDROID_NDK_HOME:-}" || ndk_try "${ANDROID_NDK_ROOT:-}" \
        || ndk_try /opt/homebrew/share/android-ndk || true
    fi
    ;;
  *)
    # Any other value is a compiler, not a typo to ignore silently.
    if [ -x "${KALSA_GATE_SYNTAX_CXX}" ] || command -v "${KALSA_GATE_SYNTAX_CXX}" >/dev/null 2>&1; then
      SYNTAX_CXX="$KALSA_GATE_SYNTAX_CXX"
      SYNTAX_CXX_TAG="explicit ${KALSA_GATE_SYNTAX_CXX##*/}"
    else
      echo "[includes] FAIL: KALSA_GATE_SYNTAX_CXX=$KALSA_GATE_SYNTAX_CXX is not an executable compiler" >&2
      exit 1
    fi
    ;;
esac

if [ -n "$SYNTAX_CXX" ] && [ -z "$SYNTAX_CXX_TAG" ]; then
  ndk_rev="$(sed -n 's/^Pkg.Revision *= *//p' "$NDK_ROOT_USED/source.properties" 2>/dev/null | tr -d '[:space:]')"
  SYNTAX_CXX_TAG="ndk ${ndk_rev:-unknown} api $NDK_API"
  if [ -n "$ENGINE_PIN" ] && [ "$ndk_rev" != "$ENGINE_PIN" ]; then
    SYNTAX_CXX_TAG="$SYNTAX_CXX_TAG, NOT this repo's pin $ENGINE_PIN"
  fi
fi
if [ -z "$SYNTAX_CXX" ]; then
  # A silent fallback that still exits 0 is how a weaker check passes for the
  # stronger one: under REQUIRE it is a failure, and no compiler at all always is.
  if [ "${KALSA_GATE_REQUIRE_NDK:-}" = "1" ]; then
    echo "[includes] FAIL: KALSA_GATE_REQUIRE_NDK=1 and the NDK clang is unusable: $NDK_WHY" >&2
    exit 1
  fi
  if command -v clang++ >/dev/null 2>&1; then
    SYNTAX_CXX="clang++"
    SYNTAX_CXX_TAG="host clang"
  else
    echo "[includes] FAIL: no compiler for the syntax pass ($NDK_WHY, and no clang++ on PATH)." >&2
    echo "[includes] The file scan alone cannot see an API that moved: it is not a pass." >&2
    exit 1
  fi
fi
echo "[includes] syntax pass compiler: $SYNTAX_CXX"

syntax_check() {  # syntax_check <file> [extra-compiler-arg...]
  local f="$1"; shift
  "$SYNTAX_CXX" -std=c++17 -fsyntax-only \
    -I "$CPP" -I "$CPP/common" -I "$CPP/common/jinja" \
    -I "$CPP/ggml-cpu" -I "$CPP/tools/mtmd" \
    ${@+"$@"} \
    "$f" > "$SYNTAX_LOG" 2>&1
}

partial=""
if [ -z "${KALSA_BMOE_DIR:-}" ]; then
  partial="KALSA_BMOE_DIR not set (rn-*.cpp, jsi and bmoe TUs skipped)"
fi

# The react-native checkout of the app repo KALSA_BMOE_DIR points into
# (kalsa/native/bmoe/rn -> kalsa/node_modules/react-native); the same
# derivation gives the bmoe root <app>/native/bmoe. A stale KALSA_BMOE_DIR
# must degrade to PARTIAL, not kill the script with a raw cd error.
RN_JSI_INCS=()
BMOE_ROOT=""
if [ -z "$partial" ]; then
  if [ ! -d "$KALSA_BMOE_DIR" ]; then
    partial="KALSA_BMOE_DIR is not a directory: ${KALSA_BMOE_DIR} (rn-*, jsi and bmoe TUs skipped)"
  else
    APP_ROOT="$(cd "$KALSA_BMOE_DIR/../../.." && pwd)"
    RN_DIR="$APP_ROOT/node_modules/react-native"
    if [ -f "$RN_DIR/ReactCommon/jsi/jsi/jsi.h" ] && \
       [ -f "$RN_DIR/ReactCommon/callinvoker/ReactCommon/CallInvoker.h" ]; then
      RN_JSI_INCS+=("$RN_DIR/ReactCommon/jsi" "$RN_DIR/ReactCommon/callinvoker")
    else
      partial="react-native headers not found under $APP_ROOT (jsi TUs skipped)"
    fi
    BMOE_ROOT="$(cd "$KALSA_BMOE_DIR/.." && pwd)"
    if [ ! -f "$BMOE_ROOT/bmoe_lmggml_compat.h" ]; then
      partial="bmoe sources not found next to KALSA_BMOE_DIR (bmoe TUs skipped)"
      BMOE_ROOT=""
    fi
  fi
fi
if [ -n "$partial" ]; then
  echo "[includes] NOTE: partial syntax pass ($partial)"
fi

# C++ TUs only; the ggml C sources are covered by the include scan.
parse_group() {  # parse_group [extra-compiler-arg...] -- <files...>
  local args=() f
  while [ "${1:-}" != "--" ]; do args+=("$1"); shift; done
  shift
  for f in "$@"; do
    [ -f "$f" ] || continue
    if ! syntax_check "$f" ${args[@]+"${args[@]}"}; then
      echo "[includes] SYNTAX FAIL: $(basename "$f")"
      grep -E "error:" "$SYNTAX_LOG" | head -3 | sed 's/^/    /' || true
      fails=$((fails + 1))
      continue
    fi
    case "$f" in
      # jinja before common: case patterns' * crosses "/", so common/* would
      # swallow common/jinja/*
      "$CPP"/common/jinja/*) c_jinja=$((c_jinja + 1)) ;;
      "$CPP"/common/*) c_common=$((c_common + 1)) ;;
      "$CPP"/models/*) c_models=$((c_models + 1)) ;;
      "$CPP"/tools/mtmd/*) c_mtmd=$((c_mtmd + 1)) ;;
      "$CPP"/jsi/*) c_jsi=$((c_jsi + 1)) ;;
      "$CPP"/ggml-cpu/*) c_cpu=$((c_cpu + 1)) ;;
      "$CPP"/ggml-opencl/*) c_opencl=$((c_opencl + 1)) ;;
      "$CPP"/rn-*) c_rn=$((c_rn + 1)) ;;
      "$CPP"/*.cpp) c_core=$((c_core + 1)) ;;
      # last: with BMOE_ROOT empty the pattern degrades to "/*", which must
      # never win over a $CPP arm
      "$BMOE_ROOT"/*) c_bmoe=$((c_bmoe + 1)) ;;
    esac
  done
}

fails=0
c_common=0; c_mtmd=0; c_models=0; c_rn=0; c_jsi=0; c_core=0; c_cpu=0; c_bmoe=0; c_jinja=0
c_opencl=0
parse_group -- "$CPP"/common/*.cpp "$CPP"/common/jinja/*.cpp "$CPP"/tools/mtmd/*.cpp \
  "$CPP"/models/*.cpp "$CPP"/tools/mtmd/models/*.cpp
parse_group -- "$CPP"/llama*.cpp "$CPP"/unicode.cpp "$CPP"/unicode-data.cpp \
  "$CPP"/ggml-backend.cpp "$CPP"/ggml-backend-dl.cpp "$CPP"/ggml-backend-meta.cpp \
  "$CPP"/ggml-backend-reg.cpp "$CPP"/ggml-opt.cpp "$CPP"/ggml-threading.cpp \
  "$CPP"/gguf.cpp
parse_group -- "$CPP"/ggml-cpu/*.cpp "$CPP"/ggml-cpu/amx/*.cpp \
  "$CPP"/ggml-cpu/arch/arm/repack.cpp "$CPP"/ggml-cpu/arch/x86/repack.cpp
# The OpenCL target's sources (android/src/main/rnllama/CMakeLists.txt
# ENABLE_OPENCL block), with its compile shape: the OpenCL-Headers tree plus
# the embedded kernel headers, and the same defines CMake passes.
parse_group -I "$ROOT/third_party/OpenCL-Headers" -I "$KERNEL_EMBED_DIR" \
  -DLM_GGML_USE_CPU -DLM_GGML_USE_CPU_REPACK -D_GNU_SOURCE \
  -DLM_GGML_USE_OPENCL -DLM_GGML_OPENCL_USE_ADRENO_KERNELS \
  -DLM_GGML_OPENCL_EMBED_KERNELS -DLM_GGML_OPENCL_SOA_Q \
  -DLM_GGML_OPENCL_TARGET_VERSION=300 -- \
  "$CPP"/ggml-opencl/ggml-opencl.cpp "$CPP"/ggml-opencl/cl-program-cache.cpp
if [ ${#RN_JSI_INCS[@]} -gt 0 ]; then
  parse_group -I "${RN_JSI_INCS[0]}" -I "${RN_JSI_INCS[1]}" -- "$CPP"/jsi/*.cpp
fi
if [ -n "$BMOE_ROOT" ]; then
  # The bmoe TUs the rnllama target compiles (android/src/main/rnllama/CMakeLists.txt
  # BMOE_SOURCE_FILES), with the same compile shape: include dirs {include,src,rn}
  # (:146-150), the forced compat header and -DBMOE_HAVE_EXPERT_READY_HOOK (:47-52).
  parse_group -I "$BMOE_ROOT/include" -I "$BMOE_ROOT/src" -I "$BMOE_ROOT/rn" \
    -DBMOE_HAVE_EXPERT_READY_HOOK -include "$BMOE_ROOT/bmoe_lmggml_compat.h" -- \
    "$BMOE_ROOT"/src/config.cpp "$BMOE_ROOT"/src/io/*.cpp "$BMOE_ROOT"/src/moe/*.cpp \
    "$BMOE_ROOT"/rn/bmoe_stream.cpp
fi
if [ -z "$partial" ]; then
  # The APK compiles these with the backends the app turns on (ENABLE_OPENCL in
  # android/src/main/CMakeLists.txt); rn-slot.cpp guards three blocks on
  # LM_GGML_USE_OPENCL, and without the define this gate never sees them.
  parse_group -I "$KALSA_BMOE_DIR" -DLM_GGML_USE_OPENCL -- "$CPP"/rn-*.cpp
fi

for t in "$CPP"/ggml-hexagon/ggml-hexagon.cpp "$CPP"/ggml-hexagon/htp-drv.cpp \
         "$ROOT"/android/src/main/RNLlamaJSI.cpp; do
  [ -f "$t" ] || continue
  case "$t" in
    */ggml-hexagon/*) r="Hexagon variant only: needs Hexagon SDK headers + -DLM_GGML_USE_HEXAGON (HEXAGON_SDK_ROOT)" ;;
    */android/src/main/RNLlamaJSI.cpp) r="JNI wrapper: needs <android/log.h> (NDK) and fbjni prefab headers; the cpp/jsi TUs it links are parsed" ;;
    *) r="not parsed by this gate" ;;
  esac
  echo "[includes] SKIP (not parsed): $(basename "$t") -- $r"
done

if [ "$fails" -gt 0 ]; then
  echo "[includes] $fails source(s) do not parse against $CPP."
  echo "[includes] An API the fork moved, or a header the sync did not bring."
  exit 1
fi
if [ -n "$partial" ] && [ "${KALSA_ALLOW_PARTIAL_GATE:-}" != "1" ]; then
  echo "[includes] FAIL: partial syntax pass ($partial)." >&2
  echo "[includes] set KALSA_BMOE_DIR=<app repo>/native/bmoe/rn, or KALSA_ALLOW_PARTIAL_GATE=1 to accept half coverage" >&2
  exit 1
fi
# The OK line must be impossible to misread: which compiler ran, the total,
# the binding counters first -- and any weakness spelled out, never a
# full-coverage sentence for a weaker pass.
total=$((c_common + c_jinja + c_mtmd + c_models + c_rn + c_jsi + c_core + c_cpu + c_opencl + c_bmoe))
counters="$c_rn rn, $c_jsi jsi, $c_bmoe bmoe, $c_core core, $c_cpu cpu, $c_opencl opencl, $c_common common, $c_jinja jinja, $c_mtmd mtmd, $c_models models"
# The counters are the only proof the pass was not vacuous: every TU list is a
# glob, and an unexpanded glob is a string the loop skips, so a tree that lost
# its sources parses nothing and would otherwise print OK. 307 today, 285 with
# the binding layer skipped.
if [ "$total" -lt 250 ] || [ "$c_common" -eq 0 ] || [ "$c_models" -lt 50 ]; then
  echo "[includes] FAIL: only $total TUs parsed ($counters)." >&2
  echo "[includes] The tree is missing sources this gate exists to cover." >&2
  exit 1
fi
if [ -z "$partial" ] && { [ "$c_rn" -eq 0 ] || [ "$c_jsi" -eq 0 ] || [ "$c_bmoe" -eq 0 ]; }; then
  echo "[includes] FAIL: full coverage claimed with rn=$c_rn jsi=$c_jsi bmoe=$c_bmoe." >&2
  exit 1
fi
weak_note=""
if [ "$SYNTAX_CXX_TAG" = "host clang" ]; then
  weak_note="host compiler, not the NDK clang the build compiles with"
fi
if [ -n "$partial" ]; then
  if [ -n "$weak_note" ]; then weak_note="$weak_note; "; fi
  weak_note="${weak_note}binding layer NOT covered: $partial"
fi
if [ -n "$weak_note" ]; then
  echo "[includes] OK [$SYNTAX_CXX_TAG] ($weak_note): $total TUs parse ($counters)"
else
  echo "[includes] OK [$SYNTAX_CXX_TAG]: $total TUs parse ($counters)"
fi
