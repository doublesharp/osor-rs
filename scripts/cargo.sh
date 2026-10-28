#!/bin/sh
set -eu
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
fi
exec cargo "$@"
