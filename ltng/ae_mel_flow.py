"""
Mel-domain flow enhancement wrapper (2026-SOTA-style, cf. FlowSE / DiT-Flow).

Stage-2 over a frozen aria_golf:
  source M0 = mel( aria_golf deterministic synthesis )   — controllable but "coarse"
  target M1 = mel( real recording )
  cond      = M0                                          — locks formants/content
  flow learns M0 -> M1 in the mel domain (phase-insensitive, dodges the time-domain trap);
  inference: M0 -> ODE -> M_enhanced -> Vocos vocoder -> waveform.

Controllability stays in aria_golf (analytic filter); the mel flow only enhances quality.
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.audiotensor import AudioTensor
from ltng.ae import VoiceAutoEncoder


class VoiceAutoEncoderMelFlow(VoiceAutoEncoder):
    def __init__(
        self,
        *args,
        mel_flow: nn.Module,                 # models.mel_flow.MelFlowEnhancer
        mel_flow_lr: float = 2e-4,
        flow_steps: int = 8,
        vocos_repo: str = "charactr/vocos-mel-24khz",
        generator_ckpt: str = None,          # warm-start aria_golf (encoder+decoder)
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mel_flow = mel_flow
        self.mel_flow_lr = mel_flow_lr
        self.flow_steps = flow_steps

        from vocos import Vocos
        self.vocos = Vocos.from_pretrained(vocos_repo)
        for p in self.vocos.parameters():
            p.requires_grad_(False)
        self.vocos.eval()

        if generator_ckpt:
            sd = torch.load(generator_ckpt, map_location="cpu",
                            weights_only=False)["state_dict"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"[MELFLOW] warm-started backbone from {generator_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    def _mel(self, wav):
        return self.vocos.feature_extractor(wav)          # (B, n_mels, T)

    @torch.no_grad()
    def _coarse_synth(self, x, f0):
        """aria_golf deterministic synthesis (frozen) → coarse waveform."""
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
        M0 = self._mel(wav_coarse[:, :T])                 # source (coarse mel)
        M1 = self._mel(real[:, :T])                       # target (clean mel)
        return M0, M1

    def training_step(self, batch, batch_idx):
        M0, M1 = self._prepare(batch)
        loss = self.mel_flow.flow_loss(M0, M1, cond=M0)
        self.log("flow_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        M0, M1 = self._prepare(batch)
        loss = self.mel_flow.flow_loss(M0, M1, cond=M0)
        M_enh = self.mel_flow.sample(M0, cond=M0, steps=self.flow_steps)
        self.log("prior_mel_l1", F.l1_loss(M0, M1), prog_bar=True, sync_dist=True)
        self.log("flow_mel_l1", F.l1_loss(M_enh, M1), prog_bar=True, sync_dist=True)
        self.tmp_val_outputs.append([loss])
        return loss

    @torch.no_grad()
    def synthesize(self, batch, steps=None):
        """aria_golf coarse → mel → flow enhance → Vocos → waveform."""
        M0, _ = self._prepare(batch)
        M_enh = self.mel_flow.sample(M0, cond=M0, steps=steps or self.flow_steps)
        return self.vocos.decode(M_enh)                   # (B, T)

    def configure_optimizers(self):
        return torch.optim.Adam(self.mel_flow.parameters(), lr=self.mel_flow_lr,
                                betas=(0.9, 0.99))
