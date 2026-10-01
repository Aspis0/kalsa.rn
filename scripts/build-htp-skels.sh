#!/bin/bash -e
# Rebuild the four shipped HTP DSP skels from the vendored engine tree.
#
# This is the ONLY supported way to produce bin/arm64-v8a/libggml-htp-*.so
# and the QAIC host stub (vendor/llama.cpp/ggml/src/ggml-hexagon/htp/v73/).
# The skels are committed and shipped as-is; nothing in CI builds them. A
# build from any other tree or toolchain speaks a different DSP protocol
# than the host (the 2026-09 dspqueue hang), so provenance is not optional:
# the engine sha comes from vendor/VERSIONS, the source is a git archive of
# HEAD (the worktree is never bind-mounted), and the toolchain is the pinned
# container image (Hexagon SDK 6.6.0.0, tools 19.0.07).
#
# Usage: scripts/build-htp-skels.sh            (requires Docker; on Apple
#                                               silicon the amd64 image runs
#                                               under emulation)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.7"
PLATFORM="linux/amd64"
MANIFEST="$ROOT_DIR/bin/arm64-v8a/HTP_SKELS"

# The supported DSP versions and the sha256 helpers are shared with the skew
# gate, which requires exactly this set (missing or extra skels fail).
. "$ROOT_DIR/scripts/htp-skels-common.sh"

ENGINE_SHA="$(sed -n 's/^LLAMA_CPP_COMMIT=//p' "$ROOT_DIR/vendor/VERSIONS")"
[[ "$ENGINE_SHA" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "build-htp-skels: vendor/VERSIONS LLAMA_CPP_COMMIT='$ENGINE_SHA' is not a full sha; run npm run sync:vendor first" >&2; exit 1; }

# The source of truth is `git archive HEAD`: it exports tracked content
# only, so uncommitted modifications of tracked files would silently build
# skels no commit can vouch for. (Untracked files cannot enter the archive.)
if ! git -C "$ROOT_DIR" diff HEAD --quiet -- vendor/llama.cpp; then
  echo "build-htp-skels: vendor/llama.cpp has uncommitted changes; the archive is HEAD -- commit or stash first" >&2
  exit 1
fi

command -v docker > /dev/null 2>&1 || { echo "build-htp-skels: docker is required" >&2; exit 1; }
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "build-htp-skels: pulling $IMAGE"
  docker pull --platform "$PLATFORM" "$IMAGE"
fi

# Host scratch under $HOME: macOS temp dirs (/var/folders) are outside
# Docker Desktop's file-sharing list, where a missing mount source silently
# becomes an empty directory inside the container.
mkdir -p "${HOME:?}/.cache"
WORK="$(mktemp -d "${HOME:?}/.cache/htp-skels-build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Exactly the paths the htp DSP project compiles against: the public ggml
# headers, the six ggml root headers the htp sources include, and the whole
# ggml-hexagon tree (host backend + htp/ DSP project). Archive paths are
# repo-relative, so the extraction mirrors the repo layout.
ARCHIVE_PATHS=(
  vendor/llama.cpp/ggml/include
  vendor/llama.cpp/ggml/src/ggml-backend-impl.h
  vendor/llama.cpp/ggml/src/ggml-common.h
  vendor/llama.cpp/ggml/src/ggml-feats.h
  vendor/llama.cpp/ggml/src/ggml-impl.h
  vendor/llama.cpp/ggml/src/ggml-quants.h
  vendor/llama.cpp/ggml/src/ggml-threading.h
  vendor/llama.cpp/ggml/src/ggml-hexagon
)
# -o would restrict the target to inside the repo; plain stdout does not.
git -C "$ROOT_DIR" archive HEAD -- "${ARCHIVE_PATHS[@]}" > "$WORK/src.tar"

docker run --rm --platform "$PLATFORM" \
  -u "$(id -u):$(id -g)" \
  -v "$WORK/src.tar:/kalsa/src.tar:ro" \
  -v "$ROOT_DIR/scripts/htp-skels-container.sh:/kalsa/build-inner.sh:ro" \
  -v "$WORK/out:/artifacts" \
  -e "KALSA_HTP_ENGINE_SHA=$ENGINE_SHA" \
  -e "HTP_DSP_VERSIONS=$HTP_DSP_VERSIONS" \
  "$IMAGE" \
  bash /kalsa/build-inner.sh

for f in $HTP_DSP_VERSIONS; do
  [ -f "$WORK/out/libggml-htp-$f.so" ] || { echo "build-htp-skels: the container produced no libggml-htp-$f.so" >&2; exit 1; }
done

mkdir -p "$ROOT_DIR/bin/arm64-v8a"
for f in $HTP_DSP_VERSIONS; do
  cp "$WORK/out/libggml-htp-$f.so" "$ROOT_DIR/bin/arm64-v8a/"
done

# The Android host build compiles the QAIC stub; refresh the tracked copy.
HTP_STUB_DIR="$ROOT_DIR/vendor/llama.cpp/ggml/src/ggml-hexagon/htp/v73"
mkdir -p "$HTP_STUB_DIR"
cp "$WORK/out/htp_iface.h" "$WORK/out/htp_iface_stub.c" "$HTP_STUB_DIR/"

# Regenerate the vouched digests; the prose stays hand-maintained.
for f in $HTP_DSP_VERSIONS; do
  sha="$(htp_sha256_of "$ROOT_DIR/bin/arm64-v8a/libggml-htp-$f.so")"
  if grep -q "^libggml-htp-$f\.so=" "$MANIFEST"; then
    sed -i.bak "s/^libggml-htp-$f\.so=.*/libggml-htp-$f.so=$sha/" "$MANIFEST"
  else
    printf 'libggml-htp-%s.so=%s\n' "$f" "$sha" >> "$MANIFEST"
  fi
done
rm -f "$MANIFEST".bak
if grep -q "^ENGINE_COMMIT=" "$MANIFEST"; then
  sed -i.bak "s/^ENGINE_COMMIT=.*/ENGINE_COMMIT=$ENGINE_SHA/" "$MANIFEST"
  rm -f "$MANIFEST".bak
else
  echo "build-htp-skels: $MANIFEST has no ENGINE_COMMIT line; add it" >&2
  exit 1
fi

echo "build-htp-skels: manifest refreshed; running the skew gate"
"$ROOT_DIR/scripts/assert-htp-skels.sh"
