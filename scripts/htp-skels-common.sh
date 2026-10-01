# Single source of truth for the HTP skel pipeline, sourced (never executed)
# by scripts/build-htp-skels.sh and scripts/assert-htp-skels.sh: the supported
# DSP versions and the sha256 helpers both scripts need. The gate requires
# exactly this version set -- a missing version silently drops a DSP the host
# can ask for, an extra one is unreviewed protocol surface.

HTP_DSP_VERSIONS="v73 v75 v79 v81"

# sha256sum on Linux (the CI runner), shasum on macOS (local runs); both
# print the hex digest as the first field of line one.
htp_sha256_of() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
