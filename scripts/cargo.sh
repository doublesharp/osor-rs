#!/bin/sh
set -eu
# rustup otherwise selects from the caller's cwd, even with --manifest-path.
export RUSTUP_TOOLCHAIN=1.95.0
osor_scratch="${OSOR_SCRATCH_ROOT:-/Volumes/Scratch/osor-rs}"
if [ -z "${OSOR_SCRATCH_ROOT:-}" ] && [ ! -d /Volumes/Scratch ]; then
    echo "Scratch is not mounted; refusing to place build output on Data." >&2
    exit 1
fi
mkdir -p "$osor_scratch/cargo/native-sdxl" "$osor_scratch/tmp/native-sdxl"
export CARGO_TARGET_DIR="$osor_scratch/cargo/native-sdxl"
export TMPDIR="$osor_scratch/tmp/native-sdxl"
if command -v sccache >/dev/null 2>&1; then
    export RUSTC_WRAPPER=sccache
    # Keep this project's server independent of other host build owners.
    # Explicit caller configuration takes precedence; Windows uses its normal
    # transport and other hosts retain their normal cache location.
    if [ "$(uname -s)" = Darwin ] && [ -z "${SCCACHE_SERVER_UDS:-}" ] && [ -z "${SCCACHE_SERVER_PORT:-}" ]; then
        export SCCACHE_SERVER_UDS="$osor_scratch/tmp/native-sdxl/sccache.sock"
    fi
    if [ -z "${OSOR_SCRATCH_ROOT:-}" ] && [ -d /Volumes/Scratch/sccache ] && [ -z "${SCCACHE_DIR:-}" ]; then
        export SCCACHE_DIR=/Volumes/Scratch/sccache
    fi
    # Avoid concurrent automatic server startup from Cargo's first compiler jobs.
    if ! sccache --show-stats >/dev/null 2>&1; then
        if ! sccache --start-server >/dev/null 2>&1 && ! sccache --show-stats >/dev/null 2>&1; then
            echo "Unable to initialize sccache; check its configured server endpoint and cache permissions." >&2
            exit 1
        fi
    fi
fi
exec cargo "$@"
