#!/bin/sh
# Mandatory CPU source and test gates. scripts/cargo.sh owns build and temp paths.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/.."
"$script_dir/cargo.sh" fmt --all -- --check
"$script_dir/cargo.sh" clippy --locked --all-targets -- -D warnings
"$script_dir/cargo.sh" test --locked --all-targets
"$script_dir/cargo.sh" test --locked --doc
