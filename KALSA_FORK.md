# KALSA fork of llama.rn

This branch (`kalsa`) is a fork of llama.rn 0.12.8. The engine is vendored,
not maintained by hand:

    vendor/llama.cpp = Aspis0/kalsallama@<LLAMA_CPP_COMMIT in vendor/VERSIONS>
                       exported by scripts/sync-vendor.sh, then patched by
                       scripts/patches/llama.cpp/*.patch

`cpp/` holds only the llama.rn binding layer (rn-*, jsi/, anyascii.* and the
rn governor glue) and is maintained by hand. The vendored layout and what
each tree carries are documented in `vendor/README.md`.

## Invariant

    vendor/llama.cpp = the kalsallama pin + scripts/patches/llama.cpp/*.patch

Nothing under `vendor/llama.cpp` is edited by hand: an edit there is lost on
the next `npm run sync:vendor`. Any kalsa-specific engine change lives as a
commit in the kalsallama fork (re-point `LLAMA_CPP_REF`) or as a patch under
`scripts/patches/llama.cpp/`.

## Commands

    npm run sync:vendor              # re-vendor every tree from vendor/VERSIONS
    scripts/assert-kalsa-vendor.sh   # grade the vendored llama.cpp tree

The assert gate grades the RESULT, never the `patch -p1` run (which succeeds
on fuzzy and partial matches): the two kalsa hunks sit exactly once each in
scope, the four upstream hunks sharing those files are still present,
`common/build-info.cpp` exists with a 7-char `LLAMA_COMMIT`, and the
fork-only governor sources are under `src/` -- the check that separates the
kalsallama pin from a plain ggml-org tree with our patches applied.

## Consuming from the app

The kalsa app depends on this fork as `github:Aspis0/llama.rn#<sha>`; after
committing a re-vendored `vendor/llama.cpp`, bump that ref to the new commit.

## Upstream llama.rn merges

Merge upstream with `git merge v0.12.x` on `kalsa`. Engine-file conflicts
resolve toward upstream's layout; kalsa engine changes live in the fork and
the patch set, never in a hand-resolved vendored file.

### Pin bump checklist

After moving `LLAMA_CPP_REF` and running `npm run sync:vendor`, grep the new
tree for `TAG_KV_CACHE_SHARE_CELLS` and re-read the draft-clear probe in
`init_mtp`/`initMTP`: the probe disables MTP for draft caches that refuse
shared-cell rollbacks, and a lifted fence means MTP could run again.
