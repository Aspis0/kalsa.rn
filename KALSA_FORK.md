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

    scripts/sync-kalsallama.sh pin <sha>  # re-point the pin, regenerate cpp/
    scripts/sync-kalsallama.sh bump       # pin to origin/<branch> of the pin
    scripts/sync-kalsallama.sh verify     # regenerate into a temp copy and
                                          # diff it against cpp/

`pin`/`bump` rewrite `kalsallama.pin`, rsync the flattened tree over `cpp/`
(`--delete`, minus the excludes above) and re-apply the patches with
`git apply`. `verify` touches nothing under `cpp/` and exits 0 only when
`cpp/` is exactly pin + patches; run it after any manual touch of `cpp/`.
`KALSALLAMA_SRC=/path/to/kalsallama` flattens from a local checkout instead
of cloning the pin repo (for pins that are not pushed yet).

## Consuming from the app

The kalsa app depends on this fork as `github:Aspis0/llama.rn#<sha>`; after
committing a regenerated `cpp/`, bump that ref to the new commit.

## Upstream llama.rn merges

Merge upstream with `git merge v0.12.x` on `kalsa`. Do not resolve engine-file
conflicts by hand: take any resolution and re-run `pin`, so `cpp/` is
regenerated from kalsallama + patches.
