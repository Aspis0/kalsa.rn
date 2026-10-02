# Single source of truth for the HTP skel pipeline, sourced (never executed)
# by scripts/build-htp-skels.sh and scripts/assert-htp-skels.sh: the supported
# DSP versions and the sha256 helpers both scripts need. The gate requires
# exactly this version set -- a missing version silently drops a DSP the host
# can ask for, an extra one is unreviewed protocol surface.

HTP_DSP_VERSIONS="v73 v75 v79 v81"

# The vendored source both sides of the host/DSP protocol are built from:
# the HTP DSP project compiles against the public ggml headers, the ggml root
# headers the htp sources include, and the whole ggml-hexagon tree (the
# recipe's archive list) -- QAIC IDL, toolchain file, and tracked host stub
# included. The Android build never compiles that tree through the engine's
# own CMakeLists, so the two files that define how the host hexagon backend
# is compiled (android/src/main/rnllama/CMakeLists.txt: the hand-written
# target_sources / target_include_directories; cmake/rnllama-sources.cmake:
# RNLLAMA_GGML_HEXAGON_DIR) are fingerprinted too -- a committed edit that
# makes the host backend resolve the protocol from different sources moves
# the fingerprint exactly like a vendored edit. One list, two consumers: the
# recipe archives exactly these paths, and htp_src_fingerprint hashes them.
HTP_SRC_PATHS=(
  vendor/llama.cpp/ggml/include
  vendor/llama.cpp/ggml/src/ggml-backend-impl.h
  vendor/llama.cpp/ggml/src/ggml-common.h
  vendor/llama.cpp/ggml/src/ggml-impl.h
  vendor/llama.cpp/ggml/src/ggml-quants.h
  vendor/llama.cpp/ggml/src/ggml-hexagon
  cmake/rnllama-sources.cmake
  android/src/main/rnllama/CMakeLists.txt
)

# sha256sum on Linux (the CI runner), shasum on macOS (local runs); both
# print the hex digest as the first field of line one.
htp_sha256_of() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

htp_sha256_stdin() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum | awk '{print $1}'
  else
    shasum -a 256 | awk '{print $1}'
  fi
}

# Content fingerprint of the committed sources: the git tree/blob hash of
# every HTP_SRC_PATHS entry at HEAD, hashed together with the path names, so
# an edit under any path -- or a change to the path list itself -- moves the
# fingerprint. Computed from HEAD, the committed tree the recipe archives;
# uncommitted worktree edits under the same paths are a separate explicit
# check in the gate, never a silent fingerprint change. The `|| return 1`
# is reachable only under set -o pipefail, which both callers must run:
# without it the loop's failure is lost inside the pipeline and a partial
# fingerprint would be hashed silently.
htp_src_fingerprint() {
  local p hash
  for p in "${HTP_SRC_PATHS[@]}"; do
    hash="$(git rev-parse "HEAD:$p")" || return 1
    printf '%s %s\n' "$p" "$hash"
  done | htp_sha256_stdin
}
