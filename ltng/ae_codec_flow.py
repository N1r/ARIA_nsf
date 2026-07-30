"""
Codec-latent flow enhancement wrapper.

Stage-2 over a frozen aria_golf backbone:
  source  Z0 = encodec_encoder( aria_golf deterministic synthesis )
  target  Z1 = encodec_encoder( real recording )
  flow    Z0 → ODE → Z_enhanced
  decode  encodec_decoder( Z_enhanced ) → waveform

Controllability lives in aria_golf (analytic poles); codec flow only improves perceptual quality.
Key difference from mel_flow: uses EnCodec's continuous pre-quantization latent instead of
Vocos mel. The codec encoder+decoder are co-trained adversarially, so the decoder is more
robust to its own latent space than Vocos is to DDSP-origin mels.
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.audiotensor import AudioTensor
from ltng.ae import VoiceAutoEncoder


class VoiceAutoEncoderCodecFlow(VoiceAutoEncoder):
    def __init__(
        self,
        *args,
        codec_flow: nn.Module,               # models.codec_flow.CodecFlowEnhancer
        codec_flow_lr: float = 2e-4,
        flow_steps: int = 8,
        encodec_bandwidth: float = 6.0,      # kbps; controls RVQ depth (unused at inference)
        generator_ckpt: str = None,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.codec_flow = codec_flow
        self.codec_flow_lr = codec_flow_lr
        self.flow_steps = flow_steps
        self.encodec_bandwidth = encodec_bandwidth

        from encodec import EncodecModel
        self._encodec = EncodecModel.encodec_model_24khz()
        self._encodec.set_target_bandwidth(encodec_bandwidth)
        for p in self._encodec.parameters():
            p.requires_grad_(False)
        self._encodec.eval()

        if generator_ckpt:
            sd = torch.load(generator_ckpt, map_location="cpu",
                            weights_only=False)["state_dict"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"[CODECFLOW] warm-started backbone from {generator_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    def _encode(self, wav):
        """ARIA waveform (B, T) → continuous EnCodec latent (B, 128, T_lat)."""
        # EnCodec expects (B, 1, T)
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        return self._encodec.encoder(wav)    # pre-quantization continuous latent

    def _decode(self, latent):
        """Enhanced latent (B, 128, T_lat) → waveform (B, T)."""
        wav = self._encodec.decoder(latent)  # (B, 1, T)
        return wav.squeeze(1)

    @torch.no_grad()
    def _coarse_synth(self, x, f0):
        """Frozen aria_golf synthesis → coarse waveform."""
        params = self.encoder(x, f0=f0 if self.train_with_true_f0 else None)
        f0_hat = params.pop("f0", None)
        if self.train_with_true_f0:
            rnd = f0.as_tensor().new_empty(f0.shape[0], 1).uniform_(50, 500)
            phase = torch.where(f0 == 0, rnd, f0) / self.sample_rate
        elif self.detach_f0:
            phase = f0_hat.new_tensor(f0_hat.as_tensor().detach()) / self.sample_rate
        else:
            phase = f0_hat / self.sample_rate
        params["phase"] = phase
        vl = params.pop("voicing_logits", None)
        if vl is not None:
            params["voicing"] = torch.sigmoid(vl)
        elif "voicing" not in params:
            params["voicing"] = (f0 > 0).float()
        return self.decoder(**params).as_tensor()

    def _prepare(self, batch):
        x, f0 = batch[0], batch[1]
        x = AudioTensor(x); f0 = AudioTensor(f0)
        wav_coarse = self._coarse_synth(x, f0)
        real = x.as_tensor()
        T = min(wav_coarse.shape[1], real.shape[1])
        # EnCodec needs 24kHz input — check sample_rate
        Z0 = self._encode(wav_coarse[:, :T])
        Z1 = self._encode(real[:, :T])
        return Z0, Z1

    def training_step(self, batch, batch_idx):
        Z0, Z1 = self._prepare(batch)
        loss = self.codec_flow.flow_loss(Z0, Z1, cond=Z0)
        self.log("flow_loss",  loss, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        Z0, Z1 = self._prepare(batch)
        loss = self.codec_flow.flow_loss(Z0, Z1, cond=Z0)
        Z_enh = self.codec_flow.sample(Z0, cond=Z0, steps=self.flow_steps)
        self.log("prior_lat_l1", F.l1_loss(Z0, Z1), prog_bar=True, sync_dist=True)
        self.log("flow_lat_l1",  F.l1_loss(Z_enh, Z1), prog_bar=True, sync_dist=True)
        self.tmp_val_outputs.append([loss])
        return loss

    @torch.no_grad()
    def synthesize(self, batch, steps=None):
        """aria_golf coarse → codec latent → flow enhance → decoder → waveform."""
        Z0, _ = self._prepare(batch)
        Z_enh = self.codec_flow.sample(Z0, cond=Z0, steps=steps or self.flow_steps)
        return self._decode(Z_enh)

    def configure_optimizers(self):
        return torch.optim.Adam(self.codec_flow.parameters(), lr=self.codec_flow_lr,
                                betas=(0.9, 0.99))
