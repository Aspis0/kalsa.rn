#!/bin/bash
#
# Asserts the kalsa invariants on the vendored llama.cpp tree.
#
# Run after `npm run sync:vendor`. sync-vendor.sh applies
# scripts/patches/llama.cpp/*.patch with `patch -p1`, which succeeds on a
# fuzzy or partial match, and a patch that applied twice is not an error to
# it either. These checks grade the RESULT, not the application:
#
#   1. the two kalsa hunks sit exactly once each, inside their own scope
#   2. the upstream hunks that share those two files are still there --
#      our patches are the UNION with upstream's, and regenerating ours
#      from a tree that lacks theirs would silently delete four features
#   3. build-info.cpp exists and its commit string is exactly 7 characters
#   4. the fork-only governor sources are under src/ -- everything in (1)
#      and (2) comes from OUR patches, so (4) is what separates the
#      kalsallama pin from a plain ggml-org tree with the same patches
#   5. the committed lib/ output agrees with src/version.ts -- lib/ is what
#      npm consumers read, and it has already drifted to a commit the pin
#      had left behind
#
# (3) is the post-condition of a guard the old flattened-tree sync enforced at
# the source: it wrote ${sha:0:7} rather than `git rev-parse --short=7`,
# because --short=7 is a MINIMUM and grows on an ambiguous prefix. The phone
# reports this string through llama_commit(), so a drift there is a build that
# lies about which engine it is.
#
# NOT covered here, because it is a pre-condition of the sync rather than a
# property of its output: sync-vendor.sh numbers the build with
# `git rev-list --count` and does not refuse a shallow clone, which would
# number a truncated graph. That guard belongs in sync-vendor.sh itself.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="$ROOT_DIR/vendor/llama.cpp"

fail() { echo "assert-kalsa-vendor: $*" >&2; exit 1; }

[ -d "$LLAMA" ] || fail "missing $LLAMA -- run npm run sync:vendor first"

common_cpp="$LLAMA/common/common.cpp"
common_h="$LLAMA/common/common.h"
for f in "$common_cpp" "$common_h"; do
  [ -f "$f" ] || fail "missing $f"
done

# 1. exactly once, in scope
marker_n=$(awk '/^[A-Za-z_].*common_params_get_system_info\(/ { in_fn = 1 }
    in_fn && /kalsa-native-patches/ { n++ }
    in_fn && /^\}/ { print n + 0; exit }' "$common_cpp")
moe_n=$(awk '/^struct common_params \{/ { in_fn = 1 }
    in_fn && /^[ \t]*\} kalsa_moe;/ { n++ }
    in_fn && /^\}/ { print n + 0; exit }' "$common_h")

[ "$marker_n" = "1" ] \
  || fail "expected exactly one 'kalsa-native-patches' line in common_params_get_system_info, found ${marker_n:-none}"
[ "$moe_n" = "1" ] \
  || fail "expected exactly one '} kalsa_moe;' line in struct common_params, found ${moe_n:-none}"

# 2. upstream's own hunks in the same two files
grep -q 'reasoning_budget_activate_immediately' "$common_h" \
  || fail "upstream's reasoning_budget_activate_immediately is gone from common.h"
grep -q 'bool vocab_only' "$common_h" \
  || fail "upstream's vocab_only is gone from common.h"
grep -q 'llama_progress_callback progress_callback' "$common_h" \
  || fail "upstream's progress_callback is gone from common.h"
# Anchor BOTH ends of the identifier. The prefix matters because
# load_progress_callback_user_data shares the suffix; the suffix matters
# because progress_callback_user_data_fake satisfies an unanchored pattern,
# which is exactly the drift this check exists to catch. What follows the
# name is deliberately left open ([=;]): pinning `= nullptr` would turn an
# upstream initializer change into a false alarm on a bump, and the point of
# this whole tree is that bumps stay cheap.
grep -qE '^[[:space:]]*void \* progress_callback_user_data[[:space:]]*[=;]' "$common_h" \
  || fail "upstream's progress_callback_user_data is gone from common.h"
grep -q 'mparams.vocab_only' "$common_cpp" \
  || fail "upstream's vocab_only wiring is gone from common.cpp"

# 3. the commit string the phone reports. A missing build-info.cpp is exactly
# what a broken sync produces, so absence is a failure, never a skip. The
# statement is matched whole and anchored at both ends: a line that merely
# contains LLAMA_COMMIT = "..." (a comment, a mention) is not a declaration.
build_info="$LLAMA/common/build-info.cpp"
[ -f "$build_info" ] || fail "missing $build_info -- a broken sync produces exactly this"
commit=$(sed -n 's/^char const \* LLAMA_COMMIT *= *"\([^"]*\)";$/\1/p' "$build_info")
[ -n "$commit" ] \
  || fail "could not read LLAMA_COMMIT from $build_info -- the LLAMA_COMMIT declaration is gone"
[ "${#commit}" = "7" ] \
  || fail "LLAMA_COMMIT is '${commit}' (${#commit} chars, want 7): --short=7 grew on an ambiguous prefix"

# 4. the engine itself is ours, not just the two patched files. The governor
# sources exist only in the kalsallama fork -- no patch under scripts/patches/
# creates them -- so a plain ggml-org tree with our patches applied fails here.
for f in llama-governor.cpp llama-governor.h \
         llama-governor-metrics.cpp llama-governor-metrics.h \
         llama-governor-policy.cpp llama-governor-policy.h \
         llama-governor-runtime.cpp; do
  p="$LLAMA/src/$f"
  # -f follows symlinks, and an empty file has no content to be wrong. The
  # claim being made is "this engine is our fork", and a filename does not
  # make it: require a real, non-symlink, non-empty file that names the
  # symbol the fork adds.
  [ -f "$p" ] && [ ! -L "$p" ] && [ -s "$p" ] \
    || fail "fork-only engine source missing, a symlink, or empty: src/$f -- this is not the kalsallama pin"
  grep -q 'llama_governor' "$p" \
    || fail "src/$f contains no llama_governor symbol -- a file with the right name alone is not the fork"
done

# 5. the committed lib/ identity. src/version.ts is what the sync writes; lib/
# is what package.json points consumers at, and it has already drifted once to
# a commit the pin had left behind. This gate only compares -- npm run build
# is the one generator, and a gate that regenerated would be a second one.
lib_val() {
  # First BUILD_<key> value in a file. The three generated version files quote
  # differently (' in the JS, " in the .d.ts) and commonjs re-exports, so match
  # any quoting after the key rather than a fixed layout.
  sed -n "s/.*$2 *= *['\"]\([^'\"]*\)['\"].*/\1/p" "$1" | head -n 1
}
version_ts="$ROOT_DIR/src/version.ts"
[ -f "$version_ts" ] || fail "missing $version_ts"
src_number=$(lib_val "$version_ts" BUILD_NUMBER)
src_commit=$(lib_val "$version_ts" BUILD_COMMIT)
[ -n "$src_number" ] && [ -n "$src_commit" ] \
  || fail "could not read BUILD_NUMBER/BUILD_COMMIT from $version_ts"
for rel in lib/commonjs/version.js lib/module/version.js lib/typescript/version.d.ts; do
  f="$ROOT_DIR/$rel"
  [ -f "$f" ] || fail "missing $rel -- run npm run build to regenerate lib/"
  [ "$(lib_val "$f" BUILD_NUMBER)" = "$src_number" ] \
    && [ "$(lib_val "$f" BUILD_COMMIT)" = "$src_commit" ] \
    || fail "$rel disagrees with src/version.ts (want $src_number/$src_commit) -- run npm run build"
done

echo "assert-kalsa-vendor: ok (marker x1, kalsa_moe x1, upstream hunks present, governor src real, lib/ matches src/version.ts)"
