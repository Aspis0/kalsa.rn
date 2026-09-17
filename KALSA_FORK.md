# KALSA fork of llama.rn

This branch (`kalsa`) is a fork of llama.rn 0.12.8 whose `cpp/` engine tree is
owned by our llama.cpp fork **kalsallama** (flattened into the llama.rn
bootstrap layout with the `LM_` prefix), not maintained by hand.

## Invariant

    cpp/ = kalsallama@<commit in kalsallama.pin> flattened by
           scripts/sync-kalsallama.sh + scripts/kalsa-patches/*.patch

Nothing under `cpp/` is edited by hand except the 36 llama.rn-owned files the
sync excludes (`rn-*`, `jsi/`, `ggml-ext.h`, `anyascii.*`); `cpp/KALSALLAMA_SHA`
is the marker naming the pin the tree was regenerated from. Any kalsa-specific
edit to pin-derived engine code must live as a patch under
`scripts/kalsa-patches/`, or the next `pin` reverts it.

## Commands

    KALSA_BMOE_DIR=<app repo>/native/bmoe/rn \
      scripts/sync-kalsallama.sh pin <sha>  # re-point the pin, regenerate cpp/
    KALSA_BMOE_DIR=<app repo>/native/bmoe/rn \
      scripts/sync-kalsallama.sh bump       # pin to origin/<branch> of the pin
    KALSA_BMOE_DIR=<app repo>/native/bmoe/rn \
      scripts/sync-kalsallama.sh verify     # regenerate into a temp copy and
                                            # check it against cpp/

All three run the include+syntax gate, which needs
`KALSA_BMOE_DIR=<app repo>/native/bmoe/rn` to parse the rn-owned sources;
without it (or without clang++) the gate runs PARTIAL and fails unless
`KALSA_ALLOW_PARTIAL_GATE=1`, which accepts what it cannot see.

`pin`/`bump` flatten the pinned commit, apply the patches to the flattened
tree, assert the kalsa post-image, run the include+syntax gate on the
regenerated tree, rsync it over `cpp/` (`--delete`, minus the excludes
above) and rewrite `kalsallama.pin` LAST -- a failing check leaves `cpp/`
and the pin untouched. `bump` refuses to run with `KALSALLAMA_SRC` set:
it resolves `origin/<branch>` from the clone cache, which the escape hatch
never fetches. `KALSALLAMA_SRC=/path/to/kalsallama` flattens from a local
checkout read-only (never fetched or pruned) for pins that are not pushed
yet.

`verify` touches nothing under `cpp/` and exits 0 only when all of these
hold: `cpp/KALSALLAMA_SHA` equals the pin; the regenerated tree with the
patches applied diffs clean against `cpp/` (excluding the llama.rn-owned
files); the kalsa post-image markers are present; and every local
`#include "..."` in `cpp/` resolves plus every C++ TU parses
(`scripts/assert-cpp-includes.sh [dir]`, run against `cpp/`). What verify
cannot catch: the copy lists do not learn, so a fork file that should be
in `cpp/` but was never copied diffs clean in both trees -- that class is
caught only by the include check, and only if something in the tree
already includes it. Run verify after any manual touch of `cpp/`.

The gate parses `common/`, `tools/mtmd/` and `models/` unconditionally;
the rn-owned sources need the app repo's bmoe headers, so run it locally
as `KALSA_BMOE_DIR=<app repo>/native/bmoe/rn scripts/assert-cpp-includes.sh`.
Without `KALSA_BMOE_DIR` (or without clang++) the gate runs PARTIAL and
exits 1 unless `KALSA_ALLOW_PARTIAL_GATE=1`.

## Consuming from the app

The kalsa app depends on this fork as `github:Aspis0/llama.rn#<sha>`; after
committing a regenerated `cpp/`, bump that ref to the new commit.

## Upstream llama.rn merges

Merge upstream with `git merge v0.12.x` on `kalsa`. Do not resolve engine-file
conflicts by hand: take any resolution and re-run `pin`, so `cpp/` is
regenerated from kalsallama + patches.
