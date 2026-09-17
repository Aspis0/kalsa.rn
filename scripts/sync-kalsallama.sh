#!/usr/bin/env bash
# Regenerate cpp/ from the pinned kalsallama commit: flatten the pin (llama.rn
# v0.12.8 bootstrap layout + LM_ prefix), rsync it over cpp/, then re-apply
# scripts/kalsa-patches/*.patch.
#
# Invariant: cpp/ = kalsallama@<kalsallama.pin> flattened + scripts/kalsa-patches,
# except the llama.rn-owned files the sync excludes (rn-*, jsi/, ggml-ext.h,
# anyascii.*) and the KALSALLAMA_SHA marker. Nothing else under cpp/ is
# hand-edited; see KALSA_FORK.md.
set -euo pipefail

OS=$(uname)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN_FILE="$ROOT/kalsallama.pin"
CPP_DIR="$ROOT/cpp"
PATCH_DIR="$ROOT/scripts/kalsa-patches"
TMP_CLONE="$ROOT/tmp/kalsallama-src"

EXTRACT_TMP=""
FLATTEN_TMP=""
cleanup() {
  if [ -n "$EXTRACT_TMP" ] && [ -d "$EXTRACT_TMP" ]; then
    rm -rf "$EXTRACT_TMP"
  fi
  if [ -n "$FLATTEN_TMP" ] && [ -d "$FLATTEN_TMP" ]; then
    rm -rf "$FLATTEN_TMP"
  fi
}
trap cleanup EXIT

usage() {
  echo "usage: $0 pin <sha> | bump | verify" >&2
  exit 2
}

die() {
  echo "error: $*" >&2
  exit 1
}

sed_i() {
  if [ "$OS" = "Darwin" ]; then
    sed -i '' "$@"
  else
    sed -i "$@"
  fi
}

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -f "$src" ]; then
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
  else
    echo "skip missing: $src" >&2
  fi
}

copy_tree_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -d "$src" ]; then
    mkdir -p "$(dirname "$dest")"
    cp -r "$src" "$dest"
  fi
}

read_pin() {
  PIN_REPO=$(node -e '
    const p = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
    if (!p.repo || !p.branch || !p.commit) process.exit(1);
    process.stdout.write(p.repo);
  ' "$PIN_FILE") || die "invalid pin at $PIN_FILE"
  PIN_BRANCH=$(node -e '
    const p = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
    process.stdout.write(p.branch);
  ' "$PIN_FILE")
  PIN_COMMIT=$(node -e '
    const p = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
    process.stdout.write(p.commit);
  ' "$PIN_FILE")
}

write_pin() {
  local repo="$1" branch="$2" commit="$3"
  PIN_REPO="$repo" PIN_BRANCH="$branch" PIN_COMMIT="$commit" PIN_PATH="$PIN_FILE" node -e '
    const fs = require("fs");
    const pin = {
      repo: process.env.PIN_REPO,
      branch: process.env.PIN_BRANCH,
      commit: process.env.PIN_COMMIT,
    };
    if (!pin.repo || !pin.branch || !pin.commit) process.exit(1);
    fs.writeFileSync(process.env.PIN_PATH, JSON.stringify(pin, null, 2) + "\n");
  ' || die "failed to write $PIN_FILE"
}

ensure_commit_object() {
  local git_dir="$1"
  local sha="$2"
  if git -C "$git_dir" cat-file -e "${sha}^{commit}" 2>/dev/null; then
    return 0
  fi
  git -C "$git_dir" fetch origin "$sha"
  git -C "$git_dir" cat-file -e "${sha}^{commit}" 2>/dev/null \
    || die "commit $sha not in $git_dir after fetch"
}

fetch_prune() {
  local git_dir="$1"
  local required="${2:-}"
  if ! git -C "$git_dir" fetch --prune origin; then
    if [ "$required" = "required" ]; then
      die "git fetch --prune origin failed in $git_dir"
    fi
    echo "warning: git fetch --prune origin failed in $git_dir" >&2
  fi
}

resolve_git_dir() {
  # KALSALLAMA_SRC is the reproducible local-source escape hatch for pins that
  # are not pushed yet. CI omits it and resolves the immutable pin from repo.
  if [ -n "${KALSALLAMA_SRC:-}" ]; then
    [ -d "$KALSALLAMA_SRC/.git" ] || [ -f "$KALSALLAMA_SRC/.git" ] \
      || die "KALSALLAMA_SRC is not a git checkout: $KALSALLAMA_SRC"
    GIT_DIR="$KALSALLAMA_SRC"
    fetch_prune "$GIT_DIR"
    return
  fi
  mkdir -p "$ROOT/tmp"
  if [ ! -d "$TMP_CLONE/.git" ]; then
    git clone "$PIN_REPO" "$TMP_CLONE"
  fi
  GIT_DIR="$TMP_CLONE"
  fetch_prune "$GIT_DIR"
}

extract_tree() {
  local git_dir="$1"
  local sha="$2"
  local head
  head=$(git -C "$git_dir" rev-parse HEAD)
  if [ "$head" = "$sha" ] && [ -z "$(git -C "$git_dir" status --porcelain)" ]; then
    LLAMA="$git_dir"
    return
  fi
  EXTRACT_TMP=$(mktemp -d "${TMPDIR:-/tmp}/kalsallama-src.XXXXXX")
  git -C "$git_dir" archive "$sha" | tar -x -C "$EXTRACT_TMP"
  LLAMA="$EXTRACT_TMP"
}

normalize_lm_prefixes() {
  local file="$1"
  if [ "$OS" = "Darwin" ]; then
    sed -i '' -E 's/(LM_)+GGML_/LM_GGML_/g' "$file"
    sed -i '' -E 's/(lm_)+ggml_/lm_ggml_/g' "$file"
    sed -i '' -E 's/(LM_)+GGUF_/LM_GGUF_/g' "$file"
    sed -i '' -E 's/(lm_)+gguf_/lm_gguf_/g' "$file"
    sed -i '' -E 's/(LM)+GGMLMetalClass/LMGGMLMetalClass/g' "$file"
  else
    sed -i -E 's/(LM_)+GGML_/LM_GGML_/g' "$file"
    sed -i -E 's/(lm_)+ggml_/lm_ggml_/g' "$file"
    sed -i -E 's/(LM_)+GGUF_/LM_GGUF_/g' "$file"
    sed -i -E 's/(lm_)+gguf_/lm_gguf_/g' "$file"
    sed -i -E 's/(LM)+GGMLMetalClass/LMGGMLMetalClass/g' "$file"
  fi
}

prefix_file() {
  local file="$1"
  local base
  base=$(basename "$file")
  case "$file" in
    */rn-*|*/rn-*.*) return ;;
  esac
  if [ "$base" = "ggml-ext.h" ]; then
    return
  fi
  case "$base" in
    rn-*) return ;;
  esac
  sed_i 's/GGML_/LM_GGML_/g' "$file"
  sed_i 's/ggml_/lm_ggml_/g' "$file"
  sed_i 's/GGUF_/LM_GGUF_/g' "$file"
  sed_i 's/gguf_/lm_gguf_/g' "$file"
  sed_i 's/GGMLMetalClass/LMGGMLMetalClass/g' "$file"
  sed_i 's/<nlohmann\/json.hpp>/"nlohmann\/json.hpp"/g' "$file"
  sed_i 's/<nlohmann\/json_fwd.hpp>/"nlohmann\/json_fwd.hpp"/g' "$file"
  normalize_lm_prefixes "$file"
}

embed_metal_headers() {
  local metal_source="$DST/ggml-metal/ggml-metal.metal"
  local metal_tmp="$DST/ggml-metal/ggml-metal.metal.tmp1"
  local common_header="$LLAMA/ggml/src/ggml-common.h"
  local impl_header="$DST/ggml-metal/ggml-metal-impl.h"
  [ -f "$metal_source" ] || return 0
  [ -f "$common_header" ] || return 0
  [ -f "$impl_header" ] || return 0
  awk '
/^#if defined\(GGML_METAL_EMBED_LIBRARY\)/ { skip=1; next }
/__embed_ggml-common.h__/ {
    system("cat '"$common_header"'")
    next
}
/^#else/ && skip { skip_else=1; next }
/^#endif/ && skip_else { skip=0; skip_else=0; next }
!skip { print }
' < "$metal_source" > "$metal_tmp"
  sed -e '/#include "ggml-metal-impl.h"/r '"$impl_header" -e '/#include "ggml-metal-impl.h"/d' \
    < "$metal_tmp" > "$metal_source"
  rm -f "$metal_tmp"
}

generate_metal_embed_s() {
  local embed_asm="$DST/ggml-metal/ggml-metal-embed.s"
  local metal_file="$DST/ggml-metal/ggml-metal.metal"
  [ -f "$metal_file" ] || return 0
  {
    echo '.section __DATA,__ggml_metallib'
    echo '.globl _lm_ggml_metallib_start'
    echo '_lm_ggml_metallib_start:'
    od -An -vtx1 "$metal_file" | awk 'NF>0 {
      printf ".byte 0x%s", $1
      for (i=2; i<=NF; i++) printf ",0x%s", $i
      printf "\n"
    }'
    echo '.globl _lm_ggml_metallib_end'
    echo '_lm_ggml_metallib_end:'
  } > "$embed_asm"
}

fix_jinja_and_ext_includes() {
  if [ -f "$DST/common/jinja/string.h" ]; then
    mv "$DST/common/jinja/string.h" "$DST/common/jinja/jinja-string.h"
  fi
  if [ -f "$DST/common/jinja/value.h" ]; then
    sed_i 's|#include "string.h"|#include "jinja-string.h"|g' "$DST/common/jinja/value.h"
  fi
  if [ -f "$DST/common/jinja/string.cpp" ]; then
    sed_i 's|#include "jinja/string.h"|#include "jinja/jinja-string.h"|g' "$DST/common/jinja/string.cpp"
  fi
  # common/unicode.h collides with the root-level unicode.h llama.rn ships
  # from llama.cpp's src/ (-I cpp precedes -I cpp/common), so give the
  # fork's header a unique name -- same treatment as jinja-string.h above --
  # and repoint its includers under common/. Root-level sources that mean
  # the src/ header are left alone.
  if [ -f "$DST/common/unicode.h" ]; then
    mv "$DST/common/unicode.h" "$DST/common/unicode-stream.h"
  fi
  local uf
  shopt -s nullglob
  for uf in "$DST"/common/*.cpp "$DST"/common/*.h \
            "$DST"/common/jinja/*.cpp "$DST"/common/jinja/*.h
  do
    sed_i 's|#include "../unicode.h"|#include "unicode-stream.h"|g' "$uf"
    sed_i 's|#include "unicode.h"|#include "unicode-stream.h"|g' "$uf"
  done
  shopt -u nullglob
  local f
  for f in "$DST/common/fit.h" "$DST/common/fit.cpp" "$DST/common/speculative.cpp"; do
    if [ -f "$f" ]; then
      sed_i 's|#include "../src/llama-ext.h"|#include "../llama-ext.h"|g' "$f"
    fi
  done
}

copy_ggml_headers() {
  copy_if_exists "$LLAMA/ggml/include/ggml.h" "$DST/ggml.h"
  copy_if_exists "$LLAMA/ggml/include/ggml-alloc.h" "$DST/ggml-alloc.h"
  copy_if_exists "$LLAMA/ggml/include/ggml-backend.h" "$DST/ggml-backend.h"
  copy_if_exists "$LLAMA/ggml/include/ggml-cpu.h" "$DST/ggml-cpu.h"
  copy_if_exists "$LLAMA/ggml/include/ggml-cpp.h" "$DST/ggml-cpp.h"
  copy_if_exists "$LLAMA/ggml/include/ggml-opt.h" "$DST/ggml-opt.h"
  copy_if_exists "$LLAMA/ggml/include/ggml-metal.h" "$DST/ggml-metal.h"
  copy_if_exists "$LLAMA/ggml/include/ggml-opencl.h" "$DST/ggml-opencl.h"
  copy_if_exists "$LLAMA/ggml/include/ggml-blas.h" "$DST/ggml-blas.h"
  copy_if_exists "$LLAMA/ggml/include/ggml-hexagon.h" "$DST/ggml-hexagon.h"
  copy_if_exists "$LLAMA/ggml/include/gguf.h" "$DST/gguf.h"
}

copy_backend_trees() {
  copy_tree_if_exists "$LLAMA/ggml/src/ggml-metal" "$DST/ggml-metal"
  rm -f "$DST/ggml-metal/CMakeLists.txt"
  embed_metal_headers

  copy_tree_if_exists "$LLAMA/ggml/src/ggml-blas" "$DST/ggml-blas"
  rm -f "$DST/ggml-blas/CMakeLists.txt"

  rm -rf "$DST/ggml-opencl"
  copy_tree_if_exists "$LLAMA/ggml/src/ggml-opencl" "$DST/ggml-opencl"
  rm -f "$DST/ggml-opencl/CMakeLists.txt"

  copy_tree_if_exists "$LLAMA/ggml/src/ggml-hexagon" "$DST/ggml-hexagon"
}

copy_ggml_cpu() {
  mkdir -p "$DST/ggml-cpu" "$DST/ggml-cpu/arch"
  local name
  for name in \
    ggml-cpu.c ggml-cpu.cpp \
    quants.c \
    repack.cpp \
    traits.cpp \
    unary-ops.cpp \
    binary-ops.cpp \
    vec.cpp \
    ops.cpp
  do
    copy_if_exists "$LLAMA/ggml/src/ggml-cpu/$name" "$DST/ggml-cpu/$name"
  done
  # Any header at the ggml-cpu root can be included by the compiled sources
  # (fork pins may add their own, e.g. repack-q23k.h), so copy all of them.
  local header
  shopt -s nullglob
  for header in "$LLAMA"/ggml/src/ggml-cpu/*.h; do
    copy_if_exists "$header" "$DST/ggml-cpu/$(basename "$header")"
  done
  shopt -u nullglob
  copy_tree_if_exists "$LLAMA/ggml/src/ggml-cpu/amx" "$DST/ggml-cpu/amx"
  copy_tree_if_exists "$LLAMA/ggml/src/ggml-cpu/arch/arm" "$DST/ggml-cpu/arch/arm"
  copy_tree_if_exists "$LLAMA/ggml/src/ggml-cpu/arch/x86" "$DST/ggml-cpu/arch/x86"
}

copy_ggml_src() {
  copy_if_exists "$LLAMA/ggml/src/ggml.c" "$DST/ggml.c"
  copy_if_exists "$LLAMA/ggml/src/ggml-impl.h" "$DST/ggml-impl.h"
  copy_if_exists "$LLAMA/ggml/src/ggml-alloc.c" "$DST/ggml-alloc.c"
  copy_if_exists "$LLAMA/ggml/src/ggml-backend.cpp" "$DST/ggml-backend.cpp"
  copy_if_exists "$LLAMA/ggml/src/ggml-backend-impl.h" "$DST/ggml-backend-impl.h"
  copy_if_exists "$LLAMA/ggml/src/ggml-backend-reg.cpp" "$DST/ggml-backend-reg.cpp"
  copy_if_exists "$LLAMA/ggml/src/ggml-backend-meta.cpp" "$DST/ggml-backend-meta.cpp"
  copy_if_exists "$LLAMA/ggml/src/ggml-backend-dl.h" "$DST/ggml-backend-dl.h"
  copy_if_exists "$LLAMA/ggml/src/ggml-backend-dl.cpp" "$DST/ggml-backend-dl.cpp"
  copy_if_exists "$LLAMA/ggml/src/ggml-common.h" "$DST/ggml-common.h"
  copy_if_exists "$LLAMA/ggml/src/ggml-opt.cpp" "$DST/ggml-opt.cpp"
  copy_if_exists "$LLAMA/ggml/src/ggml-quants.h" "$DST/ggml-quants.h"
  copy_if_exists "$LLAMA/ggml/src/ggml-quants.c" "$DST/ggml-quants.c"
  copy_if_exists "$LLAMA/ggml/src/ggml-threading.cpp" "$DST/ggml-threading.cpp"
  copy_if_exists "$LLAMA/ggml/src/ggml-threading.h" "$DST/ggml-threading.h"
  copy_if_exists "$LLAMA/ggml/src/gguf.cpp" "$DST/gguf.cpp"
}

copy_llama_api() {
  copy_if_exists "$LLAMA/include/llama.h" "$DST/llama.h"
  copy_if_exists "$LLAMA/include/llama-cpp.h" "$DST/llama-cpp.h"
  rm -rf "$DST/models"
  copy_tree_if_exists "$LLAMA/src/models" "$DST/models"
  copy_if_exists "$LLAMA/src/llama.cpp" "$DST/llama.cpp"
  # Governor/KV runtime: llama-ext.h plus the 13 fork sources called out by
  # the S1 audit as the 14-file governor integration set.
  local name
  for name in \
    llama-chat.h llama-chat.cpp \
    llama-context.h llama-context.cpp \
    llama-mmap.h llama-mmap.cpp \
    llama-model-loader.h llama-model-loader.cpp \
    llama-model-saver.h llama-model-saver.cpp \
    llama-model.h llama-model.cpp \
    llama-kv-cells.h \
    llama-kv-cache.h llama-kv-cache.cpp \
    llama-kv-cache-dsa.h llama-kv-cache-dsa.cpp \
    llama-kv-cache-dsv4.h llama-kv-cache-dsv4.cpp \
    llama-kv-cache-iswa.h llama-kv-cache-iswa.cpp \
    llama-memory-hybrid.h llama-memory-hybrid.cpp \
    llama-memory-hybrid-iswa.h llama-memory-hybrid-iswa.cpp \
    llama-memory-recurrent.h llama-memory-recurrent.cpp \
    llama-adapter.h llama-adapter.cpp \
    llama-arch.h llama-arch.cpp \
    llama-batch.h llama-batch.cpp \
    llama-cparams.h llama-cparams.cpp \
    llama-hparams.h llama-hparams.cpp \
    llama-impl.h llama-impl.cpp \
    llama-ext.h \
    llama-governor.h llama-governor.cpp \
    llama-governor-runtime.cpp \
    llama-governor-policy.h llama-governor-policy.cpp \
    llama-governor-metrics.h llama-governor-metrics.cpp \
    llama-kv-commit.h llama-kv-commit.cpp \
    llama-kv-staged.h llama-kv-staged.cpp \
    llama-hybrid-commit.h llama-hybrid-commit.cpp \
    llama-vocab.h llama-vocab.cpp \
    llama-grammar.h llama-grammar.cpp \
    llama-sampler.h llama-sampler.cpp \
    unicode.h unicode.cpp unicode-data.h unicode-data.cpp \
    llama-graph.h llama-graph.cpp \
    llama-io.h llama-io.cpp \
    llama-memory.h llama-memory.cpp
  do
    copy_if_exists "$LLAMA/src/$name" "$DST/$name"
  done
}

copy_common() {
  mkdir -p "$DST/common"
  local name
  for name in \
    log.h log.cpp \
    common.h common.cpp \
    sampling.h sampling.cpp \
    speculative.h speculative.cpp \
    ngram-cache.h ngram-cache.cpp \
    ngram-map.h ngram-map.cpp \
    ngram-mod.h ngram-mod.cpp \
    json-schema-to-grammar.h json-schema-to-grammar.cpp \
    chat.h chat.cpp \
    chat-auto-parser.h \
    chat-auto-parser-helpers.h chat-auto-parser-helpers.cpp \
    chat-auto-parser-generator.cpp \
    chat-diff-analyzer.cpp \
    chat-peg-parser.h chat-peg-parser.cpp \
    peg-parser.h peg-parser.cpp \
    unicode.h unicode.cpp \
    reasoning-budget.h reasoning-budget.cpp \
    trie.h trie.cpp \
    fit.h fit.cpp \
    build-info.h
  do
    copy_if_exists "$LLAMA/common/$name" "$DST/common/$name"
  done
  rm -f "$DST/common/json-partial.h" "$DST/common/json-partial.cpp"
  rm -f "$DST/common/regex-partial.h" "$DST/common/regex-partial.cpp"
}

copy_mtmd() {
  rm -rf "$DST/tools/mtmd"
  mkdir -p "$DST/tools/mtmd"
  copy_tree_if_exists "$LLAMA/tools/mtmd/models" "$DST/tools/mtmd/models"
  copy_tree_if_exists "$LLAMA/tools/mtmd/debug" "$DST/tools/mtmd/debug"
  local name
  for name in \
    mtmd.h mtmd.cpp \
    clip.h clip.cpp clip-impl.h clip-model.h clip-graph.h \
    mtmd-helper.cpp mtmd-helper.h \
    mtmd-audio.h mtmd-audio.cpp \
    mtmd-image.h mtmd-image.cpp
  do
    copy_if_exists "$LLAMA/tools/mtmd/$name" "$DST/tools/mtmd/$name"
  done
}

copy_vendored_third_party() {
  rm -rf "$DST/common/jinja"
  copy_tree_if_exists "$LLAMA/common/jinja" "$DST/common/jinja"
  rm -rf "$DST/nlohmann"
  copy_tree_if_exists "$LLAMA/vendor/nlohmann" "$DST/nlohmann"
  rm -rf "$DST/tools/mtmd/miniaudio" "$DST/tools/mtmd/stb"
  copy_tree_if_exists "$LLAMA/vendor/miniaudio" "$DST/tools/mtmd/miniaudio"
  copy_tree_if_exists "$LLAMA/vendor/stb" "$DST/tools/mtmd/stb"
}

prefix_vendor_tree() {
  local f
  shopt -s nullglob
  for f in \
    "$DST"/ggml-metal/*.cpp \
    "$DST"/ggml-metal/*.h \
    "$DST"/ggml-metal/*.m \
    "$DST"/ggml-metal/*.metal \
    "$DST"/ggml-blas/*.cpp \
    "$DST"/ggml-opencl/*.cpp \
    "$DST"/ggml-opencl/*.h \
    "$DST"/ggml-hexagon/*.cpp \
    "$DST"/ggml-hexagon/*.h \
    "$DST"/ggml-hexagon/htp/*.c \
    "$DST"/ggml-hexagon/htp/*.h \
    "$DST"/ggml-cpu/*.h \
    "$DST"/ggml-cpu/*.c \
    "$DST"/ggml-cpu/*.cpp \
    "$DST"/ggml-cpu/amx/*.h \
    "$DST"/ggml-cpu/amx/*.cpp \
    "$DST"/ggml-cpu/arch/arm/*.c \
    "$DST"/ggml-cpu/arch/arm/*.cpp \
    "$DST"/ggml-cpu/arch/x86/*.c \
    "$DST"/ggml-cpu/arch/x86/*.cpp \
    "$DST"/models/*.h \
    "$DST"/models/*.cpp \
    "$DST"/tools/mtmd/*.h \
    "$DST"/tools/mtmd/*.cpp \
    "$DST"/tools/mtmd/models/*.h \
    "$DST"/tools/mtmd/models/*.cpp \
    "$DST"/tools/mtmd/debug/*.h \
    "$DST"/tools/mtmd/debug/*.cpp \
    "$DST"/*.h \
    "$DST"/*.cpp \
    "$DST"/*.c \
    "$DST"/common/*.h \
    "$DST"/common/*.cpp
  do
    [ -f "$f" ] || continue
    prefix_file "$f"
  done
  shopt -u nullglob

  for f in "$DST/ggml-quants.h" "$DST/ggml-quants.c" "$DST/ggml.c"; do
    [ -f "$f" ] || continue
    sed_i 's/iq2xs_init_impl/lm_iq2xs_init_impl/g' "$f"
    sed_i 's/iq2xs_free_impl/lm_iq2xs_free_impl/g' "$f"
    sed_i 's/iq3xs_init_impl/lm_iq3xs_init_impl/g' "$f"
    sed_i 's/iq3xs_free_impl/lm_iq3xs_free_impl/g' "$f"
  done
}

# The llama.rn bootstrap prepends build-info fallback defines at the top of
# cpp/ggml.c (lm_ggml_version()/lm_ggml_commit() reference them). Insert right
# after the two MSVC guards, idempotently.
prepend_ggml_build_info() {
  local ggml_c="$DST/ggml.c"
  [ -f "$ggml_c" ] || return 0
  if grep -q '^#define LM_GGML_VERSION' "$ggml_c"; then
    return 0
  fi
  local tmp="$ggml_c.buildinfo"
  awk '
    { print }
    !inserted && /^#define _USE_MATH_DEFINES/ {
      print ""
      print "// GGML build info"
      print "#ifndef LM_GGML_VERSION"
      print "#define LM_GGML_VERSION \"unknown\""
      print "#endif"
      print "#ifndef LM_GGML_COMMIT"
      print "#define LM_GGML_COMMIT \"unknown\""
      print "#endif"
      inserted = 1
    }
  ' "$ggml_c" > "$tmp"
  grep -q '^#define LM_GGML_VERSION' "$tmp" \
    || die "anchor '#define _USE_MATH_DEFINES' not found in $ggml_c; build-info defines not inserted"
  mv "$tmp" "$ggml_c"
}

write_build_info() {
  local git_dir="$1"
  local sha="$2"
  local src="$LLAMA/common/build-info.cpp.in"
  [ -f "$src" ] || return 0
  mkdir -p "$DST/common"
  local build_number build_commit
  build_number=$(git -C "$git_dir" rev-list --count "$sha")
  build_commit=$(git -C "$git_dir" rev-parse --short=7 "$sha")
  sed -e "s|@LLAMA_BUILD_NUMBER@|$build_number|g" \
      -e "s|@LLAMA_BUILD_COMMIT@|$build_commit|g" \
      -e "s|@BUILD_COMPILER@|unknown|g" \
      -e "s|@BUILD_TARGET@|unknown|g" \
      "$src" > "$DST/common/build-info.cpp"
}

flatten_and_prefix() {
  local sha="$1"
  resolve_git_dir
  ensure_commit_object "$GIT_DIR" "$sha"
  local resolved
  resolved=$(git -C "$GIT_DIR" rev-parse --verify "${sha}^{commit}")
  [ "$resolved" = "$sha" ] || die "resolved $resolved != pin $sha"
  extract_tree "$GIT_DIR" "$sha"

  # Regenerate into a throwaway pseudo-root (tree under cpp/) so install and
  # verify share one layout: the flattened tree always stands for "the future
  # cpp/" and the patches apply at pseudo-root without touching the worktree.
  FLATTEN_TMP=$(mktemp -d "${TMPDIR:-/tmp}/kalsallama-cpp.XXXXXX")
  DST="$FLATTEN_TMP/cpp"
  mkdir -p "$DST"

  copy_ggml_headers
  copy_backend_trees
  copy_ggml_cpu
  copy_ggml_src
  copy_llama_api
  copy_common
  copy_mtmd
  copy_vendored_third_party
  fix_jinja_and_ext_includes
  prefix_vendor_tree
  prepend_ggml_build_info
  generate_metal_embed_s
  write_build_info "$GIT_DIR" "$sha"

  printf '%s\n' "$sha" > "$DST/KALSALLAMA_SHA"
}

# rsync the regenerated tree over cpp/. rn-*, jsi/, ggml-ext.h and anyascii.*
# are llama.rn-owned files that live in cpp/ but do not come from llama.cpp;
# without these excludes --delete removes them. KALSALLAMA_SHA is excluded
# from the rsync and installed explicitly so a re-pin can never leave a
# stale marker behind.
install_cpp_tree() {
  rsync -a --delete \
    --exclude 'rn-*' \
    --exclude 'jsi/' \
    --exclude 'ggml-ext.h' \
    --exclude 'anyascii.*' \
    --exclude 'KALSALLAMA_SHA' \
    "$FLATTEN_TMP/cpp/" "$CPP_DIR/"
  cp "$FLATTEN_TMP/cpp/KALSALLAMA_SHA" "$CPP_DIR/KALSALLAMA_SHA"
}

# The patches are rooted at the fork root (a/cpp/common/common.cpp): apply
# from ROOT when installing into cpp/, from the regenerated pseudo-root during
# verify -- outside a repository git apply behaves like patch -p1 there.
apply_kalsa_patches() {
  local workdir="$1"
  local patch
  for patch in "$PATCH_DIR"/*.patch; do
    [ -f "$patch" ] || die "no patches found in $PATCH_DIR"
    git -C "$workdir" apply --check "$patch" \
      || die "patch does not apply (workdir $workdir): ${patch#$ROOT/}"
    git -C "$workdir" apply "$patch" \
      || die "patch application failed (workdir $workdir): ${patch#$ROOT/}"
  done
}

# write_pin + flatten + install + patches -- the whole regeneration path.
regen_cpp() {
  local full="$1"
  write_pin "$PIN_REPO" "$PIN_BRANCH" "$full"
  flatten_and_prefix "$full"
  install_cpp_tree
  apply_kalsa_patches "$ROOT"
}

cmd_pin() {
  local want="$1"
  [ -n "$want" ] || usage
  read_pin
  resolve_git_dir
  ensure_commit_object "$GIT_DIR" "$want"
  local full
  full=$(git -C "$GIT_DIR" rev-parse --verify "${want}^{commit}")
  regen_cpp "$full"
}

cmd_bump() {
  read_pin
  resolve_git_dir
  fetch_prune "$GIT_DIR" required
  local full
  full=$(git -C "$GIT_DIR" rev-parse --verify "origin/${PIN_BRANCH}^{commit}") \
    || die "origin/${PIN_BRANCH} not found after fetch"
  regen_cpp "$full"
}

cmd_verify() {
  read_pin
  flatten_and_prefix "$PIN_COMMIT"
  apply_kalsa_patches "$FLATTEN_TMP"
  # The last three excludes are gitignored build artifacts under cpp/
  # (*.metallib, the embedded Metal assembly, the hexagon v73 output dir):
  # they are not part of the tree this command certifies.
  local diff_out
  if diff_out=$(diff -r \
      -x 'rn-*' \
      -x 'jsi' \
      -x 'ggml-ext.h' \
      -x 'anyascii.*' \
      -x 'KALSALLAMA_SHA' \
      -x '*.metallib' \
      -x 'ggml-metal-embed.s' \
      -x 'v73' \
      "$FLATTEN_TMP/cpp" "$CPP_DIR"); then
    echo "verify: cpp/ == kalsallama@$PIN_COMMIT + scripts/kalsa-patches"
  else
    echo "verify: cpp/ does not match kalsallama@$PIN_COMMIT + scripts/kalsa-patches:" >&2
    printf '%s\n' "$diff_out" >&2
    exit 1
  fi
}

[ -f "$PIN_FILE" ] || die "missing pin file $PIN_FILE"

case "${1:-}" in
  pin) cmd_pin "${2:-}" ;;
  bump) cmd_bump ;;
  verify) cmd_verify ;;
  *) usage ;;
esac
