//! Conversion through caller-owned handles, without ambient filesystem paths.

use std::collections::HashMap;
use std::io::{Read, Seek, Write};

use candle_core::{DType, Device, Tensor};

use super::{canonicalize_weights, merge_checkpoint_with_progress};
use crate::{
    Result, UPSTREAM_REVISION,
    artifact::{
        ArtifactLimits, Budget, BundleDigests, InputDigests, Manifest, PIPELINE, SourceProvenance,
        artifact_error, require_shape, stream, validate_plain_names, validate_prompt,
        validate_unet_contract,
    },
    scheduler::SchedulerConfig,
};

/// A caller-opened seekable input. Conversion starts from byte zero.
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek + ?Sized> ReadSeek for T {}

/// The four already-open safetensors inputs. No paths are reopened.
pub struct ConversionReaders<'a> {
    pub base_unet: &'a mut dyn ReadSeek,
    pub base_vae: &'a mut dyn ReadSeek,
    pub checkpoint: &'a mut dyn ReadSeek,
    pub prompt: &'a mut dyn ReadSeek,
}

/// Trusted conversion metadata, expected input identities, and resource limits.
#[derive(Clone, Debug)]
pub struct ConversionOptions {
    pub base_model_source: SourceProvenance,
    pub vae_source: SourceProvenance,
    pub checkpoint_source: SourceProvenance,
    pub prompt_source: SourceProvenance,
    pub config_source: Option<SourceProvenance>,
    pub expected_input_sha256: InputDigests,
    pub scheduler: SchedulerConfig,
    pub limits: ArtifactLimits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConversionInput {
    BaseUnet,
    BaseVae,
    Checkpoint,
    Prompt,
}

/// Fixed output identities; manifest publication has its own final operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConversionOutput {
    Unet,
    Vae,
    Prompt,
}

impl ConversionOutput {
    pub const fn file_name(self) -> &'static str {
        match self {
            Self::Unet => crate::artifact::UNET_FILE,
            Self::Vae => crate::artifact::VAE_FILE,
            Self::Prompt => crate::artifact::PROMPT_FILE,
        }
    }
}

/// Cancellation checkpoints run between bounded transfers and adapter merges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConversionEvent {
    VerifyingInput {
        input: ConversionInput,
        bytes_read: u64,
        total_bytes: u64,
    },
    ReadingInput {
        input: ConversionInput,
        bytes_read: u64,
        total_bytes: u64,
    },
    MergingAdapters {
        completed: usize,
        total: usize,
    },
    ValidatingModel,
    ModelValidated,
    WritingOutput {
        output: ConversionOutput,
        bytes_written: u64,
        total_bytes: u64,
    },
    PublishingManifest,
}

/// Application-owned output authority, normally backed by a held directory.
///
/// `create_new` must create exclusively. `finish` must complete the writer and
/// perform the application's durability checks. `publish_manifest` must publish
/// the supplied bytes atomically and exclusively through the same held directory.
/// Partial outputs and reconciliation after a publication error belong to the
/// caller; conversion does not reopen paths, retry, or remove caller files.
pub trait ConversionSink {
    type Writer: Write;
    fn create_new(&mut self, output: ConversionOutput) -> Result<Self::Writer>;
    fn finish(&mut self, output: ConversionOutput, writer: Self::Writer) -> Result<()>;
    fn publish_manifest(&mut self, bytes: &[u8]) -> Result<()>;
}

/// Convert through handles and application-owned output capabilities.
/// Return `Error::Cancelled` from `checkpoint` to stop before publication.
/// Limits include all four decoded F32 inputs, including checkpoint weights;
/// they are checked separately for the three outputs and do not bound peak RSS.
/// No fallible checkpoint runs after successful manifest publication.
pub fn convert_with_io<S: ConversionSink>(
    options: &ConversionOptions,
    readers: ConversionReaders<'_>,
    sink: &mut S,
    checkpoint: &mut dyn FnMut(ConversionEvent) -> Result<()>,
) -> Result<Manifest> {
    // Validate caller-owned strings and scheduler lengths before cloning them.
    for source in [
        (&options.base_model_source, "base model"),
        (&options.vae_source, "VAE model"),
        (&options.checkpoint_source, "checkpoint"),
        (&options.prompt_source, "prompt"),
    ] {
        source.0.validate(source.1)?;
    }
    if let Some(config) = &options.config_source {
        config.validate("config")?;
    }
    for hash in [
        &options.expected_input_sha256.base_unet,
        &options.expected_input_sha256.base_vae,
        &options.expected_input_sha256.checkpoint,
        &options.expected_input_sha256.prompt,
    ] {
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(artifact_error("SHA-256 must be 64 hexadecimal characters"));
        }
    }
    if options.scheduler.num_train_timesteps > 100_000
        || options
            .scheduler
            .trained_betas
            .as_ref()
            .is_some_and(|b| b.len() > 100_000)
        || !matches!(
            options.scheduler.beta_schedule.as_str(),
            "linear" | "scaled_linear"
        )
        || !matches!(
            options.scheduler.prediction_type.as_str(),
            "epsilon" | "sample" | "v_prediction"
        )
    {
        return Err(artifact_error("unsupported scheduler configuration"));
    }
    let mut manifest = initial_manifest(options);
    manifest.validate()?;
    encode_manifest(&manifest, &options.limits)?;
    checkpoint(ConversionEvent::VerifyingInput {
        input: ConversionInput::BaseUnet,
        bytes_read: 0,
        total_bytes: 0,
    })?;

    let mut budget = Budget::default();
    let mut prepare = |reader: &mut dyn ReadSeek, input, hash: &str| {
        stream::prepare(
            reader,
            hash,
            DType::F32,
            &options.limits,
            &mut budget,
            &mut |bytes_read, total_bytes| {
                checkpoint(ConversionEvent::VerifyingInput {
                    input,
                    bytes_read,
                    total_bytes,
                })
            },
        )
    };
    let unet_plan = prepare(
        readers.base_unet,
        ConversionInput::BaseUnet,
        &options.expected_input_sha256.base_unet,
    )?;
    let vae_plan = prepare(
        readers.base_vae,
        ConversionInput::BaseVae,
        &options.expected_input_sha256.base_vae,
    )?;
    let checkpoint_plan = prepare(
        readers.checkpoint,
        ConversionInput::Checkpoint,
        &options.expected_input_sha256.checkpoint,
    )?;
    let prompt_plan = prepare(
        readers.prompt,
        ConversionInput::Prompt,
        &options.expected_input_sha256.prompt,
    )?;
    manifest.input_sha256 = InputDigests {
        base_unet: unet_plan.sha256().to_owned(),
        base_vae: vae_plan.sha256().to_owned(),
        checkpoint: checkpoint_plan.sha256().to_owned(),
        prompt: prompt_plan.sha256().to_owned(),
    };
    let mut decode = |reader: &mut dyn ReadSeek, input, plan: &stream::PreparedTensors| {
        stream::decode(
            reader,
            plan,
            &Device::Cpu,
            &mut |bytes_read, total_bytes| {
                checkpoint(ConversionEvent::ReadingInput {
                    input,
                    bytes_read,
                    total_bytes,
                })
            },
        )
    };
    let base = decode(readers.base_unet, ConversionInput::BaseUnet, &unet_plan)?;
    let vae = decode(readers.base_vae, ConversionInput::BaseVae, &vae_plan)?;
    let adapter = decode(
        readers.checkpoint,
        ConversionInput::Checkpoint,
        &checkpoint_plan,
    )?;
    let prompt = decode(readers.prompt, ConversionInput::Prompt, &prompt_plan)?;

    let base = canonicalize_weights(base, "base UNet")?;
    require_shape(&base, "conv_in.weight", &[320, 9, 3, 3])?;
    require_shape(&base, "conv_out.weight", &[4, 320, 3, 3])?;
    require_shape(&base, "conv_out.bias", &[4])?;
    let unet =
        merge_checkpoint_with_progress(base, adapter, 256, 256.0, &mut |completed, total| {
            checkpoint(ConversionEvent::MergingAdapters { completed, total })
        })?;
    validate_unet_contract(&unet)?;
    let vae = canonicalize_weights(vae, "base VAE")?;
    validate_plain_names(&vae, "VAE")?;
    if vae.is_empty() {
        return Err(artifact_error("VAE has no tensors"));
    }
    validate_prompt(&prompt)?;
    checkpoint(ConversionEvent::ValidatingModel)?;
    let validated =
        crate::native::NativeModel::new(unet.clone(), vae.clone(), DType::F32, &Device::Cpu)?;
    drop(validated);
    checkpoint(ConversionEvent::ModelValidated)?;
    publish_validated_maps(
        manifest,
        [&unet, &vae, &prompt],
        &options.limits,
        sink,
        checkpoint,
    )
}

fn initial_manifest(options: &ConversionOptions) -> Manifest {
    let zero = "0".repeat(64);
    Manifest {
        schema_version: 1,
        upstream_revision: UPSTREAM_REVISION.to_owned(),
        pipeline: PIPELINE.to_owned(),
        base_model: options.base_model_source.clone(),
        vae_model: options.vae_source.clone(),
        checkpoint: options.checkpoint_source.clone(),
        prompt: options.prompt_source.clone(),
        config: options.config_source.clone(),
        input_sha256: options.expected_input_sha256.clone(),
        sha256: BundleDigests {
            unet: zero.clone(),
            vae: zero.clone(),
            prompt: zero,
        },
        scheduler: options.scheduler.clone(),
        timestep: 400,
        vae_scaling_factor: 0.13025,
        adapter_rank: 256,
        adapter_alpha: 256.0,
    }
}

// The caller must validate the complete model before using this publishing path.
fn publish_validated_maps<S: ConversionSink>(
    mut manifest: Manifest,
    maps: [&HashMap<String, Tensor>; 3],
    limits: &ArtifactLimits,
    sink: &mut S,
    checkpoint: &mut dyn FnMut(ConversionEvent) -> Result<()>,
) -> Result<Manifest> {
    // All digests have a fixed encoded length. Check the complete manifest
    // before consuming any caller-owned exclusive output names.
    manifest.validate()?;
    encode_manifest(&manifest, limits)?;
    let mut budget = Budget::default();
    let unet_plan = stream::prepare_output(maps[0], limits, &mut budget)?;
    let vae_plan = stream::prepare_output(maps[1], limits, &mut budget)?;
    let prompt_plan = stream::prepare_output(maps[2], limits, &mut budget)?;
    let mut write =
        |output, map: &HashMap<String, Tensor>, plan: &stream::PreparedOutput| -> Result<String> {
            let mut writer = sink.create_new(output)?;
            let hash =
                stream::write_output(&mut writer, map, plan, &mut |bytes_written, total_bytes| {
                    checkpoint(ConversionEvent::WritingOutput {
                        output,
                        bytes_written,
                        total_bytes,
                    })
                })?;
            sink.finish(output, writer)?;
            Ok(hash)
        };
    manifest.sha256 = BundleDigests {
        unet: write(ConversionOutput::Unet, maps[0], &unet_plan)?,
        vae: write(ConversionOutput::Vae, maps[1], &vae_plan)?,
        prompt: write(ConversionOutput::Prompt, maps[2], &prompt_plan)?,
    };
    manifest.validate()?;
    let encoded = encode_manifest(&manifest, limits)?;
    checkpoint(ConversionEvent::PublishingManifest)?;
    sink.publish_manifest(&encoded)?;
    Ok(manifest)
}

fn encode_manifest(manifest: &Manifest, limits: &ArtifactLimits) -> Result<Vec<u8>> {
    // Provenance and scheduler sizes have already been validated and bounded.
    let encoded = serde_json::to_vec_pretty(manifest)?;
    if u64::try_from(encoded.len()).map_err(|_| artifact_error("manifest is too large"))?
        > limits.max_manifest_bytes
    {
        return Err(artifact_error(
            "output manifest exceeds the configured size limit",
        ));
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use sha2::{Digest, Sha256};
    use std::{
        cell::Cell,
        io::{self, Cursor, SeekFrom},
    };

    struct Reader<'a> {
        cursor: Cursor<Vec<u8>>,
        calls: &'a Cell<usize>,
        fail_read: bool,
    }
    impl Read for Reader<'_> {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.calls.set(self.calls.get() + 1);
            if self.fail_read {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "prompt source sentinel",
                ));
            }
            self.cursor.read(bytes)
        }
    }
    impl Seek for Reader<'_> {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.calls.set(self.calls.get() + 1);
            self.cursor.seek(pos)
        }
    }
    #[derive(Default)]
    struct UnusedSink {
        calls: usize,
    }
    impl ConversionSink for UnusedSink {
        type Writer = Vec<u8>;
        fn create_new(&mut self, _: ConversionOutput) -> Result<Self::Writer> {
            self.calls += 1;
            Err(Error::Unavailable("unexpected output access".into()))
        }
        fn finish(&mut self, _: ConversionOutput, _: Self::Writer) -> Result<()> {
            self.calls += 1;
            Err(Error::Unavailable("unexpected output access".into()))
        }
        fn publish_manifest(&mut self, _: &[u8]) -> Result<()> {
            self.calls += 1;
            Err(Error::Unavailable("unexpected publication".into()))
        }
    }
    fn tiny_source() -> Vec<u8> {
        let mut h = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
        h.extend(std::iter::repeat_n(b' ', (8 - h.len() % 8) % 8));
        let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
        bytes.extend(h);
        bytes.extend(0_f32.to_le_bytes());
        bytes
    }
    fn options(hash: &str) -> ConversionOptions {
        let source = SourceProvenance {
            repository: "https://example.invalid/fixture".into(),
            revision: "a".repeat(40),
            license: "fixture".into(),
        };
        ConversionOptions {
            base_model_source: source.clone(),
            vae_source: source.clone(),
            checkpoint_source: source.clone(),
            prompt_source: source,
            config_source: None,
            expected_input_sha256: InputDigests {
                base_unet: hash.into(),
                base_vae: hash.into(),
                checkpoint: hash.into(),
                prompt: hash.into(),
            },
            scheduler: SchedulerConfig::default(),
            limits: ArtifactLimits::default(),
        }
    }

    #[test]
    fn cancellation_precedes_all_input_and_output_access() -> Result<()> {
        let calls = Cell::new(0);
        let mut a = Reader {
            cursor: Cursor::new(tiny_source()),
            calls: &calls,
            fail_read: false,
        };
        let mut b = Reader {
            cursor: Cursor::new(tiny_source()),
            calls: &calls,
            fail_read: false,
        };
        let mut c = Reader {
            cursor: Cursor::new(tiny_source()),
            calls: &calls,
            fail_read: false,
        };
        let mut d = Reader {
            cursor: Cursor::new(tiny_source()),
            calls: &calls,
            fail_read: false,
        };
        let mut sink = UnusedSink::default();
        let result = convert_with_io(
            &options(&"0".repeat(64)),
            ConversionReaders {
                base_unet: &mut a,
                base_vae: &mut b,
                checkpoint: &mut c,
                prompt: &mut d,
            },
            &mut sink,
            &mut |_| Err(Error::Cancelled),
        );
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(calls.get(), 0);
        assert_eq!(sink.calls, 0);
        Ok(())
    }

    #[test]
    fn invalid_expected_hash_precedes_reader_access() -> Result<()> {
        let calls = Cell::new(0);
        let mut a = Reader {
            cursor: Cursor::new(tiny_source()),
            calls: &calls,
            fail_read: false,
        };
        let mut b = Reader {
            cursor: Cursor::new(tiny_source()),
            calls: &calls,
            fail_read: false,
        };
        let mut c = Reader {
            cursor: Cursor::new(tiny_source()),
            calls: &calls,
            fail_read: false,
        };
        let mut d = Reader {
            cursor: Cursor::new(tiny_source()),
            calls: &calls,
            fail_read: false,
        };
        let mut config = options(&"0".repeat(64));
        config.expected_input_sha256.prompt = "bad".into();
        let mut sink = UnusedSink::default();
        let result = convert_with_io(
            &config,
            ConversionReaders {
                base_unet: &mut a,
                base_vae: &mut b,
                checkpoint: &mut c,
                prompt: &mut d,
            },
            &mut sink,
            &mut |_| Ok(()),
        );
        assert!(
            matches!(result, Err(Error::InvalidArtifact(message)) if message.contains("64 hexadecimal"))
        );
        assert_eq!(calls.get(), 0);
        assert_eq!(sink.calls, 0);
        Ok(())
    }

    #[test]
    fn verifies_all_sources_before_decoding_or_graph_validation() -> Result<()> {
        let bytes = tiny_source();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let calls = Cell::new(0);
        let mut a = Reader {
            cursor: Cursor::new(bytes.clone()),
            calls: &calls,
            fail_read: false,
        };
        let mut b = Reader {
            cursor: Cursor::new(bytes.clone()),
            calls: &calls,
            fail_read: false,
        };
        let mut c = Reader {
            cursor: Cursor::new(bytes.clone()),
            calls: &calls,
            fail_read: false,
        };
        let mut d = Reader {
            cursor: Cursor::new(bytes),
            calls: &calls,
            fail_read: true,
        };
        let mut sink = UnusedSink::default();
        let result = convert_with_io(
            &options(&hash),
            ConversionReaders {
                base_unet: &mut a,
                base_vae: &mut b,
                checkpoint: &mut c,
                prompt: &mut d,
            },
            &mut sink,
            &mut |_| Ok(()),
        );
        assert!(
            matches!(result, Err(error) if error.to_string().contains("prompt source sentinel"))
        );
        assert_eq!(sink.calls, 0);
        Ok(())
    }

    #[test]
    fn late_hash_mismatch_and_aggregate_input_cap_precede_decode() {
        let bytes = tiny_source();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        for over_budget in [false, true] {
            let mut config = options(&hash);
            if over_budget {
                config.limits.max_total_tensor_bytes = 12;
            } else {
                config.expected_input_sha256.prompt = "0".repeat(64);
            }
            let mut a = Cursor::new(bytes.clone());
            let mut b = Cursor::new(bytes.clone());
            let mut c = Cursor::new(bytes.clone());
            let mut d = Cursor::new(bytes.clone());
            let mut sink = UnusedSink::default();
            let mut read_events = 0;
            let result = convert_with_io(
                &config,
                ConversionReaders {
                    base_unet: &mut a,
                    base_vae: &mut b,
                    checkpoint: &mut c,
                    prompt: &mut d,
                },
                &mut sink,
                &mut |event| {
                    if matches!(event, ConversionEvent::ReadingInput { .. }) {
                        read_events += 1;
                    }
                    Ok(())
                },
            );
            let message = result.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(
                message.contains(if over_budget {
                    "budget"
                } else {
                    "SHA-256 mismatch"
                }),
                "{message}"
            );
            assert_eq!(read_events, 0);
            assert_eq!(sink.calls, 0);
        }
    }

    #[test]
    fn invalid_provenance_and_scheduler_precede_all_input_access() {
        for invalid_provenance in [true, false] {
            let calls = Cell::new(0);
            let mut a = Reader {
                cursor: Cursor::new(tiny_source()),
                calls: &calls,
                fail_read: false,
            };
            let mut b = Reader {
                cursor: Cursor::new(tiny_source()),
                calls: &calls,
                fail_read: false,
            };
            let mut c = Reader {
                cursor: Cursor::new(tiny_source()),
                calls: &calls,
                fail_read: false,
            };
            let mut d = Reader {
                cursor: Cursor::new(tiny_source()),
                calls: &calls,
                fail_read: false,
            };
            let mut config = options(&"0".repeat(64));
            if invalid_provenance {
                config.prompt_source.revision = "mutable".into();
            } else {
                config.scheduler.num_train_timesteps = 400;
            }
            let mut sink = UnusedSink::default();
            assert!(
                convert_with_io(
                    &config,
                    ConversionReaders {
                        base_unet: &mut a,
                        base_vae: &mut b,
                        checkpoint: &mut c,
                        prompt: &mut d,
                    },
                    &mut sink,
                    &mut |_| Ok(())
                )
                .is_err()
            );
            assert_eq!(calls.get(), 0);
            assert_eq!(sink.calls, 0);
        }
    }

    #[derive(Default)]
    struct SyntheticSink {
        creates: usize,
        finishes: Vec<(ConversionOutput, Vec<u8>)>,
        published: Vec<Vec<u8>>,
        publish_attempts: usize,
        fail_create: bool,
        fail_write: bool,
        fail_finish: bool,
        fail_publish: bool,
    }
    struct SyntheticWriter {
        bytes: Vec<u8>,
        fail: bool,
    }
    impl Write for SyntheticWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail {
                return Err(io::Error::other("synthetic write failure"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl ConversionSink for SyntheticSink {
        type Writer = SyntheticWriter;
        fn create_new(&mut self, _: ConversionOutput) -> Result<Self::Writer> {
            self.creates += 1;
            if self.fail_create {
                return Err(Error::Unavailable("synthetic create failure".into()));
            }
            Ok(SyntheticWriter {
                bytes: Vec::new(),
                fail: self.fail_write,
            })
        }
        fn finish(&mut self, output: ConversionOutput, writer: Self::Writer) -> Result<()> {
            if self.fail_finish {
                return Err(Error::Unavailable("synthetic finish failure".into()));
            }
            self.finishes.push((output, writer.bytes));
            Ok(())
        }
        fn publish_manifest(&mut self, bytes: &[u8]) -> Result<()> {
            self.publish_attempts += 1;
            if self.fail_publish {
                return Err(Error::Unavailable("synthetic publish failure".into()));
            }
            self.published.push(bytes.to_vec());
            Ok(())
        }
    }

    // These maps exercise the real publishing path, not full graph validation.
    fn synthetic_maps() -> Result<HashMap<String, Tensor>> {
        Ok(HashMap::from([(
            "weight".into(),
            Tensor::from_slice(&[1_f32], (1,), &Device::Cpu)?,
        )]))
    }

    #[test]
    fn synthetic_output_plans_and_sink_failures_never_publish() -> Result<()> {
        let map = synthetic_maps()?;
        let config = options(&"0".repeat(64));
        for mode in 0..8 {
            let mut limits = config.limits.clone();
            let mut sink = SyntheticSink::default();
            match mode {
                0 => limits.max_total_tensor_bytes = 8,
                1 => sink.fail_create = true,
                2 => sink.fail_write = true,
                3 => sink.fail_finish = true,
                4 => limits.max_manifest_bytes = 1,
                5 => {} // cancellation during output
                6 => {} // cancellation immediately before publication
                7 => sink.fail_publish = true,
                _ => {}
            }
            let result = publish_validated_maps(
                initial_manifest(&config),
                [&map, &map, &map],
                &limits,
                &mut sink,
                &mut |event| {
                    if (mode == 5 && matches!(event, ConversionEvent::WritingOutput { .. }))
                        || (mode == 6 && matches!(event, ConversionEvent::PublishingManifest))
                    {
                        return Err(Error::Cancelled);
                    }
                    Ok(())
                },
            );
            assert!(result.is_err(), "mode {mode}");
            assert!(sink.published.is_empty(), "mode {mode}");
            assert_eq!(sink.publish_attempts, usize::from(mode == 7));
            if mode == 0 || mode == 4 {
                assert_eq!(sink.creates, 0);
            }
        }
        Ok(())
    }

    #[test]
    fn synthetic_output_hashes_and_final_publication() -> Result<()> {
        let map = synthetic_maps()?;
        let config = options(&"0".repeat(64));
        let mut sink = SyntheticSink::default();
        let mut events = Vec::new();
        let manifest = publish_validated_maps(
            initial_manifest(&config),
            [&map, &map, &map],
            &config.limits,
            &mut sink,
            &mut |event| {
                events.push(event);
                Ok(())
            },
        )?;
        assert_eq!(sink.creates, 3);
        assert_eq!(sink.finishes.len(), 3);
        assert_eq!(sink.published.len(), 1);
        assert_eq!(sink.publish_attempts, 1);
        assert_eq!(events.last(), Some(&ConversionEvent::PublishingManifest));
        for ((identity, bytes), expected) in sink.finishes.iter().zip([
            &manifest.sha256.unet,
            &manifest.sha256.vae,
            &manifest.sha256.prompt,
        ]) {
            assert!(matches!(
                identity,
                ConversionOutput::Unet | ConversionOutput::Vae | ConversionOutput::Prompt
            ));
            assert_eq!(&format!("{:x}", Sha256::digest(bytes)), expected);
        }
        let decoded: Manifest = serde_json::from_slice(&sink.published[0])?;
        assert_eq!(decoded.sha256.unet, manifest.sha256.unet);
        Ok(())
    }
}
