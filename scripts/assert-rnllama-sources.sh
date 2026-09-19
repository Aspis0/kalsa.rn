#!/bin/bash
#
# Asserts two things about how the build is wired:
#
#   1. manifest coverage -- every explicitly-named path the build compiles
#      exists in the tree;
#   2. include style under cpp/jsi -- see the second block at the bottom.
#
# sync-vendor.sh exports whole directories AND explicitly-named files, so a
# file upstream added that no pathspec names is silently absent from vendor/
# and the failure surfaces much later as an undefined symbol far from its
# cause. cmake/rnllama-sources.cmake is the single source of truth for what
# is built, so this walks its lists and requires every explicitly-named path
# to resolve on disk:
#
#   - a globbed entry (dir/*.cpp) is skipped: the glob is evaluated at build
#     time, and a glob that matches nothing is a different bug
#   - a bare list reference (${RNLLAMA_GGML_SOURCES}) is skipped: it holds
#     glob output, not paths
#   - everything else must exist, file or directory (the lists also carry
#     include dirs)
#
# A comment can name a path, so comment lines are skipped. The parser refuses
# to pass when it extracts no paths at all, so a refactor of the cmake file
# cannot silence this check by accident.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCES="$ROOT_DIR/cmake/rnllama-sources.cmake"

fail() { echo "assert-rnllama-sources: $*" >&2; exit 1; }

[ -f "$SOURCES" ] || fail "missing $SOURCES"

# One "resolved<TAB>as-written" line per explicitly-named path. The file is
# read twice: first for the set(NAME "value") directory table, then for the
# path entries, which are expanded against that table. Paths stay relative to
# the repo root, so the checks below run against the tree as checked out.
entries=$(awk '
    BEGIN {
        # cmake built-ins and what get_filename_component() defines in the
        # list file itself; every other variable comes from pass 1.
        defs["CMAKE_CURRENT_LIST_DIR"] = "cmake"
        defs["RNLLAMA_ROOT_DIR"] = "${CMAKE_CURRENT_LIST_DIR}/.."
    }
    NR == FNR {
        if (substr($0, 1, 4) != "set(") next
        line = substr($0, 5)
        i = index(line, "\"")
        if (i == 0) next  # multi-line list; its entries are handled in pass 2
        name = substr(line, 1, i - 1)
        sub(/[[:space:]]+$/, "", name)
        if (name !~ /^[A-Za-z_][A-Za-z0-9_]*$/) next
        rest = substr(line, i + 1)
        defs[name] = substr(rest, 1, index(rest, "\"") - 1)
        next
    }
    {
        if ($0 ~ /^[[:space:]]*#/) next  # a comment can name anything
        line = $0
        while (match(line, /\$\{[A-Za-z_][A-Za-z0-9_]*\}\/[A-Za-z0-9_\/.+-]+/)) {
            token = substr(line, RSTART, RLENGTH)
            line = substr(line, RSTART + RLENGTH)
            if (token ~ /\/$/) next  # "dir/" then a glob: nothing explicit
            if (token in seen) next
            seen[token] = 1
            resolved = token
            for (n = 0; n < 20 && resolved ~ /\$\{/; n++) {
                if (!match(resolved, /\$\{[A-Za-z_][A-Za-z0-9_]*\}/)) break
                var = substr(resolved, RSTART + 2, RLENGTH - 3)
                if (!(var in defs)) {
                    printf "assert-rnllama-sources: unknown variable ${%s} in %s\n", var, token > "/dev/stderr"
                    exit 2
                }
                resolved = substr(resolved, 1, RSTART - 1) defs[var] substr(resolved, RSTART + RLENGTH)
            }
            if (resolved ~ /\$\{/) {
                printf "assert-rnllama-sources: too many nesting levels in %s\n", token > "/dev/stderr"
                exit 2
            }
            print resolved "\t" token
        }
    }
' "$SOURCES" "$SOURCES")

count=0
missing=0
while IFS=$'\t' read -r resolved token; do
  count=$((count + 1))
  if [ ! -e "$ROOT_DIR/$resolved" ]; then
    echo "assert-rnllama-sources: missing: $token -> $resolved" >&2
    missing=$((missing + 1))
  fi
done <<EOF
$entries
EOF

[ "$count" -ge 1 ] \
  || fail "extracted no explicitly-named paths from cmake/rnllama-sources.cmake -- the layout changed, fix the parser"
[ "$missing" = 0 ] \
  || fail "$missing of $count explicitly-named paths are absent from the tree"

echo "assert-rnllama-sources: ok ($count explicitly-named paths resolve)"

# With a prebuilt xcframework the pod compiles only cpp/jsi/**, and every engine
# header then comes from the framework, where the spelling is <rnllama/name.h>.
# A quoted include reaches the sibling directory but NOT the framework, so one
# written "name.h" builds fine from source and fails only in
# build-ios-frameworks, half an hour later and far from its cause.
#
# The invariant: a quoted include that does not resolve inside cpp/jsi is
# coming from the framework, so the same file must also carry the
# <rnllama/name.h> spelling behind RNLLAMA_USE_FRAMEWORK_HEADERS. That accepts
# both shapes already in the tree -- the shared block in JSINativeHeaders.h and
# the local #if in JSIParams.cpp -- and needs no exemption list.
#
# Known limit, deliberate: this does not evaluate preprocessor conditionals. A
# file that puts the two spellings in the WRONG branches, or hides one in `#if
# 0`, passes. Parsing #if from shell is the wrong tool; this is a cheap net
# that turns a thirty-minute iOS failure into a one-second one, and
# build-ios-frameworks remains the proof.

# A line-oriented match cannot tell code from a comment, and an #include inside
# /* ... */ is not an include. Strip comments first. (assert-kalsa-vendor.sh
# carries the same helper; never pipe either into `grep -q` -- the producer
# takes SIGPIPE and pipefail turns that into a random red.)
code_only() {
  perl -0777 -pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' "$1"
}

jsi_root="$(cd "$ROOT_DIR/cpp/jsi" && pwd -P)"
jsi_files="$(find "$ROOT_DIR/cpp/jsi" -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) | sort)"
[ -n "$jsi_files" ] \
  || fail "found no sources under cpp/jsi -- the layout changed, fix this check"

jsi_checked=0
jsi_bad=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  jsi_checked=$((jsi_checked + 1))
  body="$(code_only "$f")"
  includes="$(sed -n 's/^[[:space:]]*#[[:space:]]*include[[:space:]]*"\([^"]*\)".*/\1/p' <<< "$body")"
  while IFS= read -r h; do
    [ -n "$h" ] || continue
    # Resolve against the including file's own directory, not against cpp/jsi:
    # a sibling in a subdirectory is fine, and "../name.h" escapes and is not.
    cand="$(dirname "$f")/$h"
    resolved=""
    if [ -e "$cand" ]; then
      resolved="$(cd "$(dirname "$cand")" >/dev/null 2>&1 && pwd -P || true)"
    fi
    case "$resolved" in
      "$jsi_root" | "$jsi_root"/*) continue ;;
    esac
    case "$body" in
      *"<rnllama/$h>"*) ;;
      *)
        echo "assert-rnllama-sources: ${f#"$ROOT_DIR"/} includes \"$h\", which does not exist under cpp/jsi and has no <rnllama/$h> counterpart" >&2
        jsi_bad=$((jsi_bad + 1))
        ;;
    esac
  done <<< "$includes"
done <<< "$jsi_files"

[ "$jsi_bad" = 0 ] \
  || fail "$jsi_bad framework-unsafe include(s) under cpp/jsi -- guard each one with RNLLAMA_USE_FRAMEWORK_HEADERS and add the <rnllama/name.h> spelling"

echo "assert-rnllama-sources: ok ($jsi_checked cpp/jsi sources use framework-safe includes)"
