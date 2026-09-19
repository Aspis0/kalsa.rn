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
#   3. build-info's commit string is exactly 7 characters
#
# (3) is the post-condition of a guard the old sync-kalsallama.sh enforced at
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
grep -q 'mparams.vocab_only' "$common_cpp" \
  || fail "upstream's vocab_only wiring is gone from common.cpp"

# 3. the commit string the phone reports
build_info="$LLAMA/common/build-info.cpp"
if [ -f "$build_info" ]; then
  commit=$(sed -n 's/.*LLAMA_BUILD_COMMIT *= *"\([^"]*\)".*/\1/p' "$build_info")
  [ -n "$commit" ] || fail "could not read LLAMA_BUILD_COMMIT from $build_info"
  [ "${#commit}" = "7" ] \
    || fail "LLAMA_BUILD_COMMIT is '${commit}' (${#commit} chars, want 7): --short=7 grew on an ambiguous prefix"
fi

echo "assert-kalsa-vendor: ok (marker x1, kalsa_moe x1, upstream hunks present)"
