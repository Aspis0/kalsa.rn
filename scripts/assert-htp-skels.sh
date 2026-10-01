#!/bin/bash
#
# Asserts that the shipped HTP DSP skels and the vendored engine agree.
#
# The APK builds its HOST Hexagon backend from vendor/llama.cpp at
# LLAMA_CPP_COMMIT and ships the prebuilt DSP skels from bin/arm64-v8a/
# (android/build.gradle copies them into app assets). A skel built from any
# other engine commit speaks a different data-plane protocol than the host:
# control-plane RPCs still succeed, and the first in-app prefill batch hangs
# forever or aborts (the 2026-09 dspqueue hang, scratchpad
# inapp-htp-dspqueue-hang/DIAGNOSIS.md H2). This gate turns that drift into a
# red build, at commit time and in CI.
#
# Fails when:
#   1. bin/arm64-v8a/HTP_SKELS is missing or malformed
#   2. the manifest's ENGINE_COMMIT differs from vendor/VERSIONS
#      LLAMA_CPP_COMMIT -- the skels were not rebuilt after a vendor bump
#   3. the shipped set is not exactly the supported DSP versions (see
#      scripts/htp-skels-common.sh), a shipped libggml-htp-*.so has no
#      manifest entry, is listed but absent, or its sha256 differs from the
#      manifest -- the committed binary is not the one the manifest vouches for
#   4. a shipped skel is not a QDSP6 ELF of plausible size, or does not
#      carry exactly one KALSA_HTP_ENGINE=<LLAMA_CPP_COMMIT> string -- the
#      identity lives in the binary itself (scripts/htp-skels-container.sh
#      links a generated stamp TU into every skel), so no text edit can
#      vouch for a stale skel: only a rebuild through the committed recipe
#      can satisfy this leg
#
# The manifest also carries the build provenance (SDK/tools/image versions);
# those lines are informational and unchecked -- a gate on them could not be
# honest without rebuilding the skels, which is the skel build's job.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT_DIR/bin/arm64-v8a/HTP_SKELS"
VERSIONS="$ROOT_DIR/vendor/VERSIONS"
BIN_DIR="$ROOT_DIR/bin/arm64-v8a"

# Supported DSP versions + sha256 helpers, shared with the build recipe.
. "$ROOT_DIR/scripts/htp-skels-common.sh"

fail() { echo "assert-htp-skels: $*" >&2; exit 1; }

# sha256sum on Linux (the CI runner), shasum on macOS (local runs); both
# print the hex digest as the first field of line one.
sha256_of() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

[ -f "$MANIFEST" ] || fail "missing $MANIFEST -- the skel manifest is not optional"
[ -f "$VERSIONS" ] || fail "missing $VERSIONS -- run npm run sync:vendor first"

manifest_engine="$(sed -n 's/^ENGINE_COMMIT=//p' "$MANIFEST")"
[ -n "$manifest_engine" ] \
  || fail "HTP_SKELS has no ENGINE_COMMIT line -- it does not vouch for anything"
[[ "$manifest_engine" =~ ^[0-9a-f]{40}$ ]] \
  || fail "ENGINE_COMMIT='${manifest_engine}' is not a full 40-hex sha"

pin_engine="$(sed -n 's/^LLAMA_CPP_COMMIT=//p' "$VERSIONS")"
[ -n "$pin_engine" ] || fail "vendor/VERSIONS has no LLAMA_CPP_COMMIT line"

# (a) the skel provenance must be the vendored engine, to the full sha. A
# prefix match would let a rebuilt-but-unpushed engine pass while the pin
# moved on, and the phone reports the short sha while the pin is the truth.
if [ "$manifest_engine" != "$pin_engine" ]; then
  fail "skel engine sha ${manifest_engine:0:7} != vendored engine sha ${pin_engine:0:7} -- the DSP skels were not rebuilt from the vendored tree; rebuild them (see the provenance block in bin/arm64-v8a/HTP_SKELS)"
fi

# Every manifest entry must be "name=64-hex" and the name must be a shipped
# skel; every shipped skel must have an entry. Walk the real files, not the
# manifest, so a renamed or newly added skel cannot pass unvouched.
shipped="$(find "$BIN_DIR" -maxdepth 1 -name 'libggml-htp-*.so' -exec basename {} \; | sort)"
[ -n "$shipped" ] || fail "no libggml-htp-*.so under $BIN_DIR -- nothing ships, which is itself a change to prove"

# The shipped set must be EXACTLY the supported versions. A missing version
# silently drops a DSP the host can ask for; an extra one is an unreviewed
# protocol surface (the example app's Gradle task also lists all four, but
# this gate is what fails a drift).
expected="$(printf 'libggml-htp-%s.so\n' $HTP_DSP_VERSIONS | sort)"
shipped_set="$(printf '%s\n' "$shipped" | sort)"
if [ "$shipped_set" != "$expected" ]; then
  fail "shipped skels [$(printf '%s' "$shipped_set" | tr '\n' ' ')] != the supported set [$(printf '%s' "$expected" | tr '\n' ' ')] -- rebuild exactly $HTP_DSP_VERSIONS (scripts/build-htp-skels.sh)"
fi

manifest_list="$(sed -n 's/^\(libggml-htp-.*\.so\)=[0-9a-f]\{64\}$/\1/p' "$MANIFEST" | sort)"
[ -n "$manifest_list" ] \
  || fail "HTP_SKELS has no 'libggml-htp-*.so=<sha256>' entries"

unvouched="$(comm -23 <(printf '%s\n' "$shipped") <(printf '%s\n' "$manifest_list"))"
[ -z "$unvouched" ] \
  || fail "shipped skel(s) with no HTP_SKELS entry: $(echo "$unvouched" | tr '\n' ' ')-- add them to the manifest with their sha256"

phantom="$(comm -13 <(printf '%s\n' "$shipped") <(printf '%s\n' "$manifest_list"))"
[ -z "$phantom" ] \
  || fail "HTP_SKELS vouches for absent file(s): $(echo "$phantom" | tr '\n' ' ')-- rebuild the skels or fix the manifest"

# (b) the committed bytes must be the manifest's bytes. shasum reads the
# files; a mismatch means the .so was touched after the manifest was written.
mismatches=0
while IFS= read -r name; do
  want="$(sed -n "s/^${name}=//p" "$MANIFEST")"
  got="$(sha256_of "$BIN_DIR/$name")"
  if [ "$got" != "$want" ]; then
    echo "assert-htp-skels: $name sha256 $got != manifest $want" >&2
    mismatches=$((mismatches + 1))
  fi
done <<< "$shipped"
[ "$mismatches" -eq 0 ] \
  || fail "$mismatches skel file(s) do not match HTP_SKELS -- the shipped binary is not the one the manifest vouches for"

# (c) each shipped skel must BE a Hexagon DSP skel and must name its engine.
# Header: ELF magic, 32-bit little-endian, e_machine = 164 (EM_QDSP6), bytes
# 0..19 via od; size floor 256 KB (real skels are ~0.9 MB, and the failure
# this catches is a placeholder/garbage file far below it). Stamp: exactly
# one KALSA_HTP_ENGINE=<40 hex> string, equal to the pin -- a second or
# different engine sha means mixed-provenance bytes.
size_floor=$((256 * 1024))
bad_identity=0
while IFS= read -r name; do
  file="$BIN_DIR/$name"
  if [ "$(wc -c < "$file")" -lt "$size_floor" ] \
    || ! od -An -tu1 -v -N20 "$file" | awk '{
        for (i = 1; i <= NF; i++) { n++; b[n] = $i }
      }
      END {
        exit (b[1] == 127 && b[2] == 69 && b[3] == 76 && b[4] == 70 &&
              b[5] == 1 && b[6] == 1 && b[19] == 164 && b[20] == 0) ? 0 : 1
      }'; then
    echo "assert-htp-skels: $name is not a QDSP6 ELF of plausible size" >&2
    bad_identity=$((bad_identity + 1))
    continue
  fi
  stamps="$(grep -aoE 'KALSA_HTP_ENGINE=[0-9a-f]{40}' "$file" || true)"
  count="$(printf '%s' "$stamps" | grep -c 'KALSA_HTP_ENGINE=' || true)"
  if [ "$count" -ne 1 ]; then
    echo "assert-htp-skels: $name carries $count KALSA_HTP_ENGINE stamps (want exactly 1) -- rebuild through scripts/build-htp-skels.sh" >&2
    bad_identity=$((bad_identity + 1))
    continue
  fi
  skel_engine="${stamps#KALSA_HTP_ENGINE=}"
  if [ "$skel_engine" != "$pin_engine" ]; then
    echo "assert-htp-skels: $name engine sha ${skel_engine:0:7} != vendored engine sha ${pin_engine:0:7} -- the skel bytes predate the pin; rebuild (scripts/build-htp-skels.sh)" >&2
    bad_identity=$((bad_identity + 1))
  fi
done <<< "$shipped"
[ "$bad_identity" -eq 0 ] \
  || fail "$bad_identity skel file(s) fail the binary identity check"

echo "assert-htp-skels: ok (engine ${manifest_engine:0:7}, $(printf '%s\n' "$shipped" | wc -l | tr -d ' ') skels match the manifest, each names its engine)"
