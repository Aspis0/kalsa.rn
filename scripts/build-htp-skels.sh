#!/bin/bash -e
# Rebuild the four shipped HTP DSP skels from the vendored engine tree.
#
# The supported way to produce bin/arm64-v8a/libggml-htp-*.so and the QAIC
# host stub (vendor/llama.cpp/ggml/src/ggml-hexagon/htp/v73/). The skels are
# committed and shipped as-is; nothing in CI builds them. The skew gate binds
# the shipped bytes to this recipe's inputs -- each skel is stamped with the
# engine sha and a fingerprint of the exact committed sources, both
# recomputed from the checkout -- so a build from any other tree or toolchain
# fails the gate instead of shipping (the 2026-09 dspqueue hang). Forging the
# stamp symbols themselves is deliberate ELF surgery and out of scope.
#
# Usage: scripts/build-htp-skels.sh            (requires Docker; on Apple
#                                               silicon the amd64 image runs
#                                               under emulation)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Tag v0.7, pinned by digest -- the OCI index carrying linux/amd64, and the
# image the committed skels were built in. A mutable tag would let a refreshed
# or locally stale image drift under an unchanged recipe.
IMAGE="ghcr.io/snapdragon-toolchain/arm64-android@sha256:91714433626f0d94a926538a1e46ec43756c5b8e3262b91b95df1e812940aed1"
PLATFORM="linux/amd64"
MANIFEST="$ROOT_DIR/bin/arm64-v8a/HTP_SKELS"

# The supported DSP versions and the sha256 helpers are shared with the skew
# gate, which requires exactly this set (missing or extra skels fail).
. "$ROOT_DIR/scripts/htp-skels-common.sh"

ENGINE_SHA="$(sed -n 's/^LLAMA_CPP_COMMIT=//p' "$ROOT_DIR/vendor/VERSIONS")"
[[ "$ENGINE_SHA" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "build-htp-skels: vendor/VERSIONS LLAMA_CPP_COMMIT='$ENGINE_SHA' is not a full sha; run npm run sync:vendor first" >&2; exit 1; }

# Content fingerprint of the committed sources (scripts/htp-skels-common.sh):
# stamped into every skel next to the engine sha, and recomputed by the gate,
# so a committed source edit under the same upstream sha cannot ship old
# skels against a new host protocol.
SRC_FP="$(htp_src_fingerprint)" \
  || { echo "build-htp-skels: cannot fingerprint the committed sources (git rev-parse failed)" >&2; exit 1; }
[[ "$SRC_FP" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "build-htp-skels: source fingerprint '$SRC_FP' is not a sha256" >&2; exit 1; }

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

# Exactly the paths the htp DSP project compiles against and the host hexagon
# backend compiles from (HTP_SRC_PATHS in scripts/htp-skels-common.sh, which
# also defines the fingerprint over them). Archive paths are repo-relative,
# so the extraction mirrors the repo layout.
git -C "$ROOT_DIR" archive HEAD -- "${HTP_SRC_PATHS[@]}" > "$WORK/src.tar"

docker run --rm --platform "$PLATFORM" \
  -u "$(id -u):$(id -g)" \
  -v "$WORK/src.tar:/kalsa/src.tar:ro" \
  -v "$ROOT_DIR/scripts/htp-skels-container.sh:/kalsa/build-inner.sh:ro" \
  -v "$WORK/out:/artifacts" \
  -e "KALSA_HTP_ENGINE_SHA=$ENGINE_SHA" \
  -e "KALSA_HTP_SRC_FP=$SRC_FP" \
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
if grep -q "^SRC_FINGERPRINT=" "$MANIFEST"; then
  sed -i.bak "s/^SRC_FINGERPRINT=.*/SRC_FINGERPRINT=$SRC_FP/" "$MANIFEST"
  rm -f "$MANIFEST".bak
else
  printf 'SRC_FINGERPRINT=%s\n' "$SRC_FP" >> "$MANIFEST"
fi

echo "build-htp-skels: manifest refreshed; running the skew gate"
"$ROOT_DIR/scripts/assert-htp-skels.sh"
