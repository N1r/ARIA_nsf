"""
Excitation flow-matching wrapper around VoiceAutoEncoder.

Two-stage / warm-start design (mirrors the GAN finetune, but the head is a flow instead
of a discriminator):
  Stage 1 (already done): aria_golf deterministic pretraining -> good analytic filter +
    glottal table.
  Stage 2 (here): FREEZE encoder+decoder, train an ExcitationFlow head that maps the
    deterministic glottal excitation x0 (informed prior) to the true excitation x1 =
    inverse-filter(real speech). At inference the flow ODE refines x0 -> excitation, which
    is re-filtered by the SAME analytic vocal tract, so formant controllability is intact.

x0 = src * gain   (glottal flow + shaped noise, on the gain-included filter-input scale)
x1 = end_filter.reverse(real_speech, ctrl)   (FIR inverse of the all-pole cascade)
cond = [f0_norm, voicing] upsampled to sample rate (periodicity / gate hints for the flow)
"""

import torch
import torch.nn.functional as F
from torch import nn

from models.audiotensor import AudioTensor
from ltng.ae import VoiceAutoEncoder


class VoiceAutoEncoderFlow(VoiceAutoEncoder):
    def __init__(
        self,
        *args,
        flow: nn.Module,             # models.flow.ExcitationFlow
        flow_lr: float = 2e-4,
        flow_steps: int = 4,         # ODE steps (training + inference)
        spec_weight: float = 1.0,    # end-to-end spectral loss (0 = pure flow-matching)
        generator_ckpt: str = None,  # warm-start aria_golf (encoder+decoder)
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flow = flow
        self.flow_lr = flow_lr
        self.flow_steps = flow_steps
        self.spec_weight = spec_weight
        self.freeze_backbone = freeze_backbone

        if generator_ckpt:
            sd = torch.load(generator_ckpt, map_location="cpu",
                            weights_only=False)["state_dict"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"[FLOW] warm-started backbone from {generator_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    # ---- assemble x0 (glottal prior), x1 (true excitation), cond ----
    def _prepare(self, batch):
        if len(batch) == 5:
            x, f0_in_hz, *_ = batch
        else:
            x, f0_in_hz = batch
        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)

        params = self.encoder(x, f0=f0_in_hz if self.train_with_true_f0 else None)
        params.pop("f0", None)

        # phase from TRUE f0 (filter is frozen; deterministic glottal as in training)
        random_f0 = f0_in_hz.as_tensor().new_empty(f0_in_hz.shape[0], 1).uniform_(50, 500)
        phase = torch.where(f0_in_hz == 0, random_f0, f0_in_hz) / self.sample_rate
        params["phase"] = phase

        voicing_logits = params.pop("voicing_logits", None)
        if voicing_logits is not None:
            params["voicing"] = torch.sigmoid(voicing_logits).detach()
        elif "voicing" not in params:
            params["voicing"] = (f0_in_hz > 0).float()

        dec = self.decoder
        src = dec.excitation(**params)                          # x0 pre-gain (B, T)
        ctrl = params["end_filter_params"][0]
        gain, _ = dec.end_filter._logits_to_gain_lpc(ctrl)
        x0 = (src * gain).as_tensor()                           # gain-included prior
        x1 = dec.end_filter.reverse(x, ctrl).as_tensor()        # true excitation

        T = min(x0.shape[1], x1.shape[1])
        x0, x1 = x0[:, :T], x1[:, :T]
        cond = self._build_cond(f0_in_hz, params["voicing"], T)
        return x0, x1, cond, ctrl

    def _build_cond(self, f0_in_hz, voicing, T):
        f0 = f0_in_hz.as_tensor().float()
        if f0.ndim == 1:
            f0 = f0[:, None]
        v = voicing.as_tensor().float() if isinstance(voicing, AudioTensor) else voicing.float()
        if v.ndim == 1:
            v = v[:, None]
        f0u = F.interpolate(f0.unsqueeze(1), size=T, mode="linear",
                            align_corners=False).squeeze(1)
        vu = F.interpolate(v.unsqueeze(1), size=T, mode="nearest").squeeze(1)
        return torch.stack([f0u / 500.0, vu], dim=1)            # (B, 2, T)

    def training_step(self, batch, batch_idx):
        x0, x1, cond, ctrl = self._prepare(batch)
        fm_loss = self.flow.flow_loss(x0, x1, cond)        # flow-matching velocity loss
        loss = fm_loss
        self.log("flow_loss", fm_loss, prog_bar=True, sync_dist=True)
        # end-to-end spectral loss: integrate the flow, re-filter to a waveform, and match
        # the real spectrum. Spectral magnitude is phase-insensitive, so this avoids the
        # harmonic-phase trap that made the pure time-domain MSE over-smooth.
        if self.spec_weight > 0:
            refined = self.flow.sample(x0, cond, self.flow_steps)
            wav = self.decoder.end_filter.synth_from_excitation(AudioTensor(refined), ctrl)
            wav = self.decoder.room_filter(wav)
            real = AudioTensor(batch[0])
            Tw = min(wav.shape[1], real.shape[1])
            spec_loss = self.criterion(wav[:, :Tw], real[:, :Tw]).as_tensor()
            loss = fm_loss + self.spec_weight * spec_loss
            self.log("spec_loss", spec_loss, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, cond, ctrl = self._prepare(batch)
        loss = self.flow.flow_loss(x0, x1, cond)
        # report how much the flow improves over the raw glottal prior (sanity signal)
        refined = self.flow.sample(x0, cond, steps=self.flow_steps)
        self.log("prior_l1", F.l1_loss(x0, x1), prog_bar=True, sync_dist=True)
        self.log("flow_l1", F.l1_loss(refined, x1), prog_bar=True, sync_dist=True)
        # feed the parent's accumulator so on_validation_epoch_end averages val_loss
        self.tmp_val_outputs.append([loss])
        return loss

    @torch.no_grad()
    def synthesize(self, batch, steps=None):
        """Inference path that actually exercises the flow head: glottal prior x0 →
        flow ODE refine → re-filter through the analytic vocal tract (+ room). A standard
        decoder.forward would bypass the flow. batch = (audio, f0[, ...]); returns (B, T)."""
        x0, _, cond, ctrl = self._prepare(batch)
        refined = self.flow.sample(x0, cond, steps or self.flow_steps)
        wav = self.decoder.end_filter.synth_from_excitation(AudioTensor(refined), ctrl)
        wav = self.decoder.room_filter(wav)
        return wav.as_tensor()

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow.parameters(), lr=self.flow_lr, betas=(0.9, 0.99))
