# Publishing OSOR

The `Publish crate` workflow is started manually from `main`. It resolves a
release tag to one commit, checks its version, runs the complete CPU/GPU-feature
CI matrix and dependency policy, builds CPU API docs, and verifies the packaged
crate. Uploading requires the separate `publish` input.

## Prepare a release

1. Update `Cargo.toml`, `Cargo.lock`, and the changelog, then run
   `scripts/check.sh` and `scripts/cargo.sh publish --locked --dry-run`.
2. Commit and push the release source to `main`.
3. Create and push the matching tag, for example `v0.1.0` for package `0.1.0`.
4. Run `Publish crate` from the Actions tab with that tag and leave **Upload to
   crates.io** unchecked. All checks must pass.
5. Run it again with the same tag and **Upload to crates.io** checked to publish.

The equivalent GitHub CLI commands are:

```sh
gh workflow run publish.yml --ref main -f tag=v0.1.0
# When ready to upload:
gh workflow run publish.yml --ref main -f tag=v0.1.0 -F publish=true
```

Keep published tags fixed. Confirm the
[crate page](https://crates.io/crates/osor) and [API docs](https://docs.rs/osor).
Git consumers such as PhotoCraft keep their resolved source in `Cargo.lock`.

## Authentication

For the first publication, add a crates.io API token that can create `osor` as
the repository secret `CARGO_REGISTRY_TOKEN`. Verify the publishing account's
email on crates.io first. The token is used only by the upload job.

After the crate exists, configure its
[trusted publisher](https://crates.io/docs/trusted-publishing) for repository
`doublesharp/osor-rs` and workflow `publish.yml`, then remove the repository
token. With no token configured, the workflow obtains a short-lived credential
through `rust-lang/crates-io-auth-action` instead.

## Package and documentation

`cargo package --list` shows the published files. Repository-only agent guidance,
CI workflows and dependency-policy configuration are excluded; source, examples,
documentation and all license notices remain included. Weight files stay outside
the package. docs.rs builds the CPU library for `x86_64-unknown-linux-gnu`, with
no GPU toolkit required.
