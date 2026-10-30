# Local Phase II bundles

For downloads and a complete conversion plan, start with the
[model preparation guide](model-preparation.md).

OSOR uses four fixed files in one local directory: `manifest.json`,
`unet.safetensors`, `vae.safetensors`, and `prompt.safetensors`. No file is
downloaded by the library. The UNet and VAE names are unprefixed diffusers
names. The prompt file contains exactly `prompt_embeds` with shape
`[1, 77, 2048]` and `pooled_prompt_embeds` with shape `[1, 1280]`.

`artifact::Manifest` is the schema. Version 1 pins the upstream code revision,
`osor-sdxlinpainting-phase2`, the complete scheduler configuration, timestep
400, VAE scale 0.13025, and adapter rank and alpha 256. It records repository,
full immutable 40- or 64-hex revision, and license for the base model,
checkpoint and prompts, with separate VAE provenance and optional configuration provenance. It also records
SHA-256 for each of the four exact conversion inputs and the three output
tensor files. The caller must supply truthful provenance and independently
determine its rights under each source's terms. The code license does not grant
weight rights.

`ModelBundle::load(path, device, dtype)` validates the manifest, fixed file
locations, output hashes, tensor metadata, nine-channel input and five-channel
output heads, and both prompt shapes. It accepts F16, BF16 or F32 tensors and
returns canonical named maps. The native model constructor must still validate
all remaining required model weights. `load_with_limits` accepts explicit
`ArtifactLimits`; the defaults bound individual files, headers, tensor names,
shapes, tensor count and aggregate decoded bytes. Tensor bytes are read and
hashed as a stream, using one reusable bounded read buffer, without mmap. Tensors of 1 MiB or more are hashed on a helper thread while the tensor is constructed from the same immutable bytes; the digest order and coverage are unchanged. Narrower runtime dtypes are converted on CPU before upload to avoid a larger temporary GPU allocation. The digest verifies the bytes used to create
the tensors; it does not authenticate an untrusted manifest. Bundle directory
contents must not be concurrently modified by an untrusted writer. These limits
bound decoded input sizes, not peak memory: dtype conversion, LoRA merging,
device transfers and serialization can temporarily retain additional copies.

`conversion::convert(&ConversionRequest)` accepts local safetensors exports for
the base UNet, base VAE, Phase II checkpoint and fixed prompts, plus separate UNet/VAE source
provenance, scheduler configuration, output directory and limits. Supply the
actual immutable source revisions; the converter does not guess them. The
output directory must not exist and its parent must already exist. Input files
must be regular files, not symlinks, and must use F16, BF16 or F32 tensors.
Conversion computes in F32 and writes F32 safetensors. It constructs the complete fixed native graph to verify every required UNet and VAE tensor before creating the output directory.

The converter derives targets from the base weights for `to_k`, `to_q`, `to_v`,
`to_out.0`, `conv`, `conv1`, `conv2`, `conv_shortcut`, `proj_in`, `proj_out`,
`ff.net.2` and `ff.net.0.proj`. Every target requires the rank-256 A and B
weights. Linear and convolution updates are merged with alpha/rank equal to
one; convolution B must be 1x1. The trained `conv_out.weight` and
`conv_out.bias` must have shapes `[5, 320, 3, 3]` and `[5]`. Other biases are
preserved. Prefixes `module.`, `_orig_mod.`, and `unet.`, PEFT
`.base_layer.`, and both default-named and unnamed A/B weights are normalized.
Any full checkpoint copies of frozen base weights must match the supplied base
exactly after F32 conversion. Name collisions, unexpected adapter tensors,
missing targets, mismatched shapes and unknown checkpoint tensors fail closed.

The tensor files are created exclusively and a complete, synced manifest is
published last using a hard link, which must be supported by the output
filesystem. A failed conversion can leave an incomplete directory without a
manifest; remove that task-owned directory before retrying. The converter never
replaces an existing bundle.

The released upstream `.pth` checkpoints are pickle-based. If one is needed,
export it to safetensors in a separately provisioned, isolated development
environment outside this repository, then pass the exported files here. The
Rust runtime and converter do not load pickle or run a Python sidecar. Neither
the synthetic tests nor a successfully converted bundle establish numerical
parity with the released model; that requires the exact artifacts and captured
reference intermediates.

## Conversion through held capabilities

Installers that hold directory capabilities can use the additive
`conversion::convert_with_io(&options, readers, &mut sink, &mut checkpoint)`.
`ConversionReaders` borrows four already-open `Read + Seek` sources as
`base_unet`, `base_vae`, `checkpoint`, and `prompt`. Conversion seeks from byte
zero and never opens or reopens a path. `ConversionOptions` contains the same
source provenance and scheduler as `ConversionRequest`, plus trusted
`expected_input_sha256: InputDigests` and `limits: ArtifactLimits`.

The application implements this output boundary:

```rust
use std::io::Write;
use osor::{Result, conversion::ConversionOutput};

trait ConversionSink {
    type Writer: Write;
    fn create_new(&mut self, output: ConversionOutput) -> Result<Self::Writer>;
    fn finish(&mut self, output: ConversionOutput, writer: Self::Writer) -> Result<()>;
    fn publish_manifest(&mut self, bytes: &[u8]) -> Result<()>;
}
```

Implement the library's `osor::conversion::ConversionSink` trait; the
declaration above shows its signatures. `ConversionOutput::{Unet,Vae,Prompt}`
maps to the fixed bundle filenames through `file_name()`. The caller creates
each writer exclusively through its held directory, performs durability checks
in `finish`, and publishes the manifest atomically and exclusively through that
same directory in `publish_manifest`. The library flushes each writer before
`finish`. It never removes partial files, retries publication, or supplies an
ambient output path. The caller owns cleanup and must reconcile a publication
error if its implementation may have committed before returning the error.

All four input headers, aggregate F32 decoded sizes, and trusted whole-file
hashes are checked before any weight tensors are constructed. During decoding,
the converter checks the original size and header and verifies each tensor's
raw bytes against its preflight digest before constructing that tensor. This
binds decoding to verified bytes even when an open source can be modified.
Limits include temporary checkpoint tensors: the preserved recipe needs
13,716,870,592 decoded input bytes, whereas its final model needs
10,605,175,216. A runtime model cap of 12 GiB therefore cannot also serve as
this recipe's conversion cap. Input and output aggregates are checked
separately; neither is a peak RSS bound. Budget merging and other transient
storage separately. The full native graph, all three output plans, and manifest
size are validated before creating any output writer.

The callback receives `ConversionEvent` during verification, decoding, adapter
merging, model validation, output writing, and immediately before publication.
Return `Error::Cancelled` to stop. Transfers have checkpoints at most 64 KiB
apart; merges have checkpoints between adapter updates. Individual tensor
construction, neural operations, caller I/O methods, and synchronization remain
synchronous. Cancellation may leave partial unpublished outputs for caller
cleanup. There is no fallible callback after successful manifest publication.
The original path-based `convert` API retains its behavior.

Expected hashes must come from the installer's trusted recipe. The mirror's
F32 prompt and the preserved qualified F16 export are distinct inputs; changing
the recipe requires independent numerical qualification. This API adds no
network access or automatic download.

## Pinning an imported bundle

Applications that approve a multi-file bundle without copying its weights can retain a trusted SHA-256 of the exact imported manifest bytes and pass it to `Osor::load_with_options_and_manifest_sha256(path, options, expected_manifest_sha256)`. The loader reads a bounded manifest buffer, verifies its digest, and parses that same buffer. A mismatch fails before any weight-file read or weight allocation. The existing tensor loader then hashes the exact bytes used to construct tensors against the verified manifest's digests.

The additive lower-level `ModelBundle::load_with_manifest_sha256(path, device, dtype, expected_hash)` provides the same pin check before returning its tensor maps. Pins must contain 64 hexadecimal characters; hexadecimal case is ignored. Existing unpinned load methods and custom artifact limits retain their behavior.

Use `Osor::load_with_options_and_manifest_sha256_and_limits(path, options, expected_manifest_sha256, &limits)` to enforce application-owned artifact limits during pinned loading. The corresponding lower-level method is `ModelBundle::load_with_limits_and_manifest_sha256(path, device, dtype, &limits, expected_hash)`. The same aggregate budget spans UNet, VAE, and prompt tensors. Header preflight counts the target decoded dtype before allocating file weights, so F16-to-F32 expansion consumes twice the source tensor bytes. A valid manifest and tensor hashes do not exempt extra named tensors from this budget.

For example, `ArtifactLimits { max_total_tensor_bytes: 24_u64 * 1024 * 1024 * 1024, ..Default::default() }` imposes a 24 GiB decoded-weight cap. Choose this cap within the application's admission policy and budget transient/model execution memory separately. It is not a peak RSS or total GPU-memory cap. Existing pinned and unpinned methods retain their default limits.

Acquire the expected digest from a trusted import or approval step and retain it separately from the replaceable bundle path. Computing a fresh digest from that path immediately before loading merely approves its current contents. A digest establishes byte identity with the trusted import; provenance and permission to use the model remain application responsibilities. `Osor::manifest()`, `device()`, and `dtype()` expose the loaded configuration without reopening bundle files.
