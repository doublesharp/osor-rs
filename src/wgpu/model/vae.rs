use super::super::Weights;
use super::layers::{Attention, Conv, GroupNorm, Resnet, upsample};
use crate::Result;
use burn::tensor::{Tensor, activation};

struct Mid {
    first: Resnet,
    norm: GroupNorm,
    attention: Attention,
    second: Resnet,
}
impl Mid {
    fn new(w: &mut Weights, p: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            first: Resnet::new(w, &format!("{p}.resnets.0"), channels, channels, None, 1e-6)?,
            norm: GroupNorm::new(w, &format!("{p}.attentions.0.group_norm"), channels, 1e-6)?,
            attention: Attention::new(
                w,
                &format!("{p}.attentions.0"),
                channels,
                channels,
                1,
                true,
            )?,
            second: Resnet::new(w, &format!("{p}.resnets.1"), channels, channels, None, 1e-6)?,
        })
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.first.forward(x, None);
        let [b, c, h, w] = x.dims();
        let y = self
            .norm
            .forward(x.clone())
            .reshape([b, c, h * w])
            .swap_dims(1, 2);
        let y = self
            .attention
            .forward(y.clone(), y)
            .swap_dims(1, 2)
            .reshape([b, c, h, w]);
        self.second.forward(x + y, None)
    }
}
struct Block {
    resnets: Vec<Resnet>,
    sample: Option<Conv>,
}
impl Block {
    fn encode(&self, mut x: Tensor<4>) -> Tensor<4> {
        for resnet in &self.resnets {
            x = resnet.forward(x, None);
        }
        if let Some(conv) = &self.sample {
            x = conv.forward(x);
        }
        x
    }
    fn decode(&self, mut x: Tensor<4>) -> Tensor<4> {
        for resnet in &self.resnets {
            x = resnet.forward(x, None);
        }
        if let Some(conv) = &self.sample {
            let [_, _, h, w] = x.dims();
            x = conv.forward(upsample(x, [h * 2, w * 2]));
        }
        x
    }
}
pub(super) struct Vae {
    encoder_in: Conv,
    down: Vec<Block>,
    encoder_mid: Mid,
    encoder_norm: GroupNorm,
    encoder_out: Conv,
    quant: Conv,
    post_quant: Conv,
    decoder_in: Conv,
    decoder_mid: Mid,
    up: Vec<Block>,
    decoder_norm: GroupNorm,
    decoder_out: Conv,
}
impl Vae {
    pub fn new(w: &mut Weights) -> Result<Self> {
        let channels = [128, 256, 512, 512];
        let encoder_in = Conv::new(w, "encoder.conv_in", 3, 128, 3, 1, 1, false)?;
        let mut down = Vec::new();
        for (i, &out) in channels.iter().enumerate() {
            let input = if i == 0 { 128 } else { channels[i - 1] };
            let p = format!("encoder.down_blocks.{i}");
            let resnets = vec![
                Resnet::new(w, &format!("{p}.resnets.0"), input, out, None, 1e-6)?,
                Resnet::new(w, &format!("{p}.resnets.1"), out, out, None, 1e-6)?,
            ];
            let sample = if i < 3 {
                Some(Conv::new(
                    w,
                    &format!("{p}.downsamplers.0.conv"),
                    out,
                    out,
                    3,
                    2,
                    0,
                    true,
                )?)
            } else {
                None
            };
            down.push(Block { resnets, sample });
        }
        let encoder_mid = Mid::new(w, "encoder.mid_block", 512)?;
        let encoder_norm = GroupNorm::new(w, "encoder.conv_norm_out", 512, 1e-6)?;
        let encoder_out = Conv::new(w, "encoder.conv_out", 512, 8, 3, 1, 1, false)?;
        let quant = Conv::new(w, "quant_conv", 8, 8, 1, 1, 0, false)?;
        let post_quant = Conv::new(w, "post_quant_conv", 4, 4, 1, 1, 0, false)?;
        let decoder_in = Conv::new(w, "decoder.conv_in", 4, 512, 3, 1, 1, false)?;
        let decoder_mid = Mid::new(w, "decoder.mid_block", 512)?;
        let mut up = Vec::new();
        let rev = [512, 512, 256, 128];
        for (i, &out) in rev.iter().enumerate() {
            let input = if i == 0 { 512 } else { rev[i - 1] };
            let p = format!("decoder.up_blocks.{i}");
            let resnets = vec![
                Resnet::new(w, &format!("{p}.resnets.0"), input, out, None, 1e-6)?,
                Resnet::new(w, &format!("{p}.resnets.1"), out, out, None, 1e-6)?,
                Resnet::new(w, &format!("{p}.resnets.2"), out, out, None, 1e-6)?,
            ];
            let sample = if i < 3 {
                Some(Conv::new(
                    w,
                    &format!("{p}.upsamplers.0.conv"),
                    out,
                    out,
                    3,
                    1,
                    1,
                    false,
                )?)
            } else {
                None
            };
            up.push(Block { resnets, sample });
        }
        let decoder_norm = GroupNorm::new(w, "decoder.conv_norm_out", 128, 1e-6)?;
        let decoder_out = Conv::new(w, "decoder.conv_out", 128, 3, 3, 1, 1, false)?;
        Ok(Self {
            encoder_in,
            down,
            encoder_mid,
            encoder_norm,
            encoder_out,
            quant,
            post_quant,
            decoder_in,
            decoder_mid,
            up,
            decoder_norm,
            decoder_out,
        })
    }
    pub fn encode(&self, image: Tensor<4>) -> Tensor<4> {
        let mut x = self.encoder_in.forward(image);
        for block in &self.down {
            x = block.encode(x);
        }
        x = self.encoder_mid.forward(x);
        self.quant.forward(
            self.encoder_out
                .forward(activation::silu(self.encoder_norm.forward(x))),
        )
    }
    #[cfg(test)]
    pub(super) fn encode_with_trace(
        &self,
        image: Tensor<4>,
        mut trace: impl FnMut(&str, Tensor<4>) -> Result<()>,
    ) -> Result<Tensor<4>> {
        let mut x = self.encoder_in.forward(image);
        trace("encoder.conv_in", x.clone())?;
        for (i, block) in self.down.iter().enumerate() {
            x = block.encode(x);
            trace(&format!("encoder.down_blocks.{i}"), x.clone())?;
        }
        x = self.encoder_mid.forward(x);
        trace("encoder.mid_block", x.clone())?;
        x = self.encoder_norm.forward(x);
        trace("encoder.conv_norm_out", x.clone())?;
        x = self.encoder_out.forward(activation::silu(x));
        trace("encoder.conv_out", x.clone())?;
        x = self.quant.forward(x);
        trace("quant_conv", x.clone())?;
        Ok(x)
    }
    pub fn decode(&self, latent: Tensor<4>) -> Tensor<4> {
        let mut x = self
            .decoder_mid
            .forward(self.decoder_in.forward(self.post_quant.forward(latent)));
        for block in &self.up {
            x = block.decode(x);
        }
        self.decoder_out
            .forward(activation::silu(self.decoder_norm.forward(x)))
    }
}
