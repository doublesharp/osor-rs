# Prepare a local OSOR model

Download four safetensors files, verify their hashes, and merge them into a
local Phase II bundle with the Rust converter. The base UNet and VAE come from
the pinned SDXL-Inpainting repository; the Phase II adapters and fixed prompt
come from a pinned safetensors mirror. No Python environment is needed for this
recipe. The [model license](https://huggingface.co/eerie-road/osor-sdxl/blob/a14effb4aabab39cbcbefd7517d2af147ff4d283/LICENSE.md)
is CreativeML Open RAIL++-M, separate from this crate's source licenses.

## Download the inputs

On this Mac, keep the files on Scratch. On another host, set `OSOR_MODEL_ROOT`
to an absolute path outside the checkout and set `OSOR_SCRATCH_ROOT` for build
output as described in [platform setup](platforms.md).

```sh
test -d /Volumes/Scratch || exit 1
export OSOR_MODEL_ROOT=/Volumes/Scratch/osor-rs/data/phase2
mkdir -p "$OSOR_MODEL_ROOT/inputs"

curl --fail --location --output "$OSOR_MODEL_ROOT/inputs/base-unet.safetensors" \
  https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/115134f363124c53c7d878647567d04daf26e41e/unet/diffusion_pytorch_model.fp16.safetensors
curl --fail --location --output "$OSOR_MODEL_ROOT/inputs/base-vae.safetensors" \
  https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/115134f363124c53c7d878647567d04daf26e41e/vae/diffusion_pytorch_model.fp16.safetensors
curl --fail --location --output "$OSOR_MODEL_ROOT/inputs/phase2.safetensors" \
  https://huggingface.co/eerie-road/osor-sdxl/resolve/a14effb4aabab39cbcbefd7517d2af147ff4d283/sdxlinpainting_phase2.safetensors
curl --fail --location --output "$OSOR_MODEL_ROOT/inputs/fixed-prompt.safetensors" \
  https://huggingface.co/eerie-road/osor-sdxl/resolve/a14effb4aabab39cbcbefd7517d2af147ff4d283/fixed_prompt_embeds.safetensors
```

The four inputs total about 8.42 GB. Conversion creates about 10.61 GB of F32
weights; allow at least 20 GB of free disk space and a 40 GiB memory budget for
preparation. Verify the files before conversion (`sha256sum -c` can replace
`shasum -a 256 -c` on Linux):

```sh
(
  cd "$OSOR_MODEL_ROOT/inputs" || exit 1
  shasum -a 256 -c <<'SHA256'
6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f  base-unet.safetensors
4ad62825e5c8b31eefb77355ba6693785619bf7d669d9b1a6fd9f19dec6d65b3  base-vae.safetensors
fff6bea76360666f1233dbf74840f45a438140f2c0f093365ef017ebd7a457ff  phase2.safetensors
93bc3ef2fe9e1616c164450aa58f9f34ba23569215722158d83a3fb5ad157e50  fixed-prompt.safetensors
SHA256
)
```

## Create the conversion plan

From the `osor-rs` checkout, write this plan alongside the inputs. The output
`bundle` directory must not exist. The paths in the plan are absolute.

```sh
cat > "$OSOR_MODEL_ROOT/conversion.json" <<JSON
{
  "base_unet": "$OSOR_MODEL_ROOT/inputs/base-unet.safetensors",
  "base_vae": "$OSOR_MODEL_ROOT/inputs/base-vae.safetensors",
  "checkpoint": "$OSOR_MODEL_ROOT/inputs/phase2.safetensors",
  "prompt": "$OSOR_MODEL_ROOT/inputs/fixed-prompt.safetensors",
  "output_dir": "$OSOR_MODEL_ROOT/bundle",
  "base_model_source": {
    "repository": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "revision": "115134f363124c53c7d878647567d04daf26e41e",
    "license": "CreativeML Open RAIL++-M"
  },
  "vae_source": {
    "repository": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "revision": "115134f363124c53c7d878647567d04daf26e41e",
    "license": "CreativeML Open RAIL++-M"
  },
  "checkpoint_source": {
    "repository": "https://huggingface.co/eerie-road/osor-sdxl",
    "revision": "a14effb4aabab39cbcbefd7517d2af147ff4d283",
    "license": "CreativeML Open RAIL++-M"
  },
  "prompt_source": {
    "repository": "https://huggingface.co/eerie-road/osor-sdxl",
    "revision": "a14effb4aabab39cbcbefd7517d2af147ff4d283",
    "license": "CreativeML Open RAIL++-M"
  },
  "config_source": {
    "repository": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "revision": "115134f363124c53c7d878647567d04daf26e41e",
    "license": "CreativeML Open RAIL++-M"
  },
  "scheduler": {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "trained_betas": null,
    "prediction_type": "epsilon",
    "clip_sample": false,
    "clip_sample_range": 1.0,
    "thresholding": false,
    "dynamic_thresholding_ratio": 0.995,
    "sample_max_value": 1.0
  }
}
JSON

scripts/cargo.sh run --locked --release --bin osor -- \
  convert "$OSOR_MODEL_ROOT/conversion.json"
```

The converter validates the graph and writes `unet.safetensors`,
`vae.safetensors`, `prompt.safetensors`, and `manifest.json` into `bundle/`.
Pass that directory to `Osor::load` or `Osor::builder`. Keep the manifest digest
from `shasum -a 256 "$OSOR_MODEL_ROOT/bundle/manifest.json"` if your application
uses trusted-manifest loading.

This recipe uses the mirror's F32 prompt. The earlier F16-export recipe has a
different prompt digest; their numerical records are identified separately in
[qualification](qualification.md). See the [artifact contract](artifacts.md) for
the bundle format and capability-based conversion API.
