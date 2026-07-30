"""
Adversarial wrapper around VoiceAutoEncoder.

VoiceAutoEncoderGAN keeps the full generator pipeline (encoder + ARIA decoder,
formant supervision, voiced gate, MSS loss) and adds the user's multi-scale
spectrogram discriminator with manual G/D alternation (LSGAN + label smoothing +
feature matching). Can warm-start the generator from a pretrained ARIA checkpoint
so adversarial training is a fine-tune (recommended).
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Optional
import torchaudio

from models.audiotensor import AudioTensor
from models.discriminator import (
    spectrogram_pyramid, lsgan_discriminator_loss, lsgan_generator_loss,
    feature_matching_loss,
)
from ltng.ae import VoiceAutoEncoder


class VoiceAutoEncoderGAN(VoiceAutoEncoder):
    def __init__(
        self,
        *args,
        discriminator: nn.Module,
        period_discriminator: Optional[nn.Module] = None,   # legacy single-disc kwarg (kept for old configs)
        wave_discriminators: Optional[List[nn.Module]] = None,  # waveform-domain discriminators (MPD, MS-STFTD, MS-SB-CQTD, …)
        adv_weight: float = 1.0,        # beta (weight on generator adv loss)
        fm_weight: float = 2.0,         # feature-matching weight
        mel_weight: float = 0.0,        # mel-spectrogram L1 (HiFi-GAN style; 0=off)
        mel_n_fft: int = 1024,
        mel_hop: int = 256,
        mel_n_mels: int = 80,
        gen_lr: float = 1e-4,        # BigVGAN finding: 2e-4 collapses D early; 1e-4 stable
        disc_lr: float = 1e-4,
        lr_decay: float = 0.999,     # HiFi-GAN ExponentialLR per epoch (1.0 = off)
        gan_start_step: int = 0,        # warmup: pure MSS before this step
        generator_ckpt: str = None,
        scales: tuple = (2048, 1024, 512, 256, 128, 64),
        fft_factor: int = 1,
        overlap: float = 0.75,
        real_label_min: float = 0.9,    # label smoothing for D stability
        real_label_max: float = 1.0,
        grad_clip: float = 0.5,         # manual G grad clip (trainer clip N/A here)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        # Normalise wave discriminators: accept both old kwarg and new list.
        # wave_discriminators takes precedence; period_discriminator kept for backward compat.
        if wave_discriminators is not None:
            self._wave_discs = nn.ModuleList(wave_discriminators)
        elif period_discriminator is not None:
            self._wave_discs = nn.ModuleList([period_discriminator])
        else:
            self._wave_discs = nn.ModuleList([])
        # Legacy attribute so old eval code that accesses .period_discriminator still works
        self.period_discriminator = self._wave_discs[0] if len(self._wave_discs) == 1 else None
        self.grad_clip = grad_clip
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight
        self.mel_weight = mel_weight
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.lr_decay = lr_decay
        self.gan_start_step = gan_start_step
        self.scales = tuple(scales)
        self.fft_factor = fft_factor
        self.overlap = overlap
        self.real_label_min = real_label_min
        self.real_label_max = real_label_max
        self.automatic_optimization = False
        if mel_weight > 0:
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_fft=mel_n_fft, hop_length=mel_hop,
                n_mels=mel_n_mels, power=1.0)

        if generator_ckpt:
            sd = torch.load(generator_ckpt, map_location="cpu",
                            weights_only=False)["state_dict"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"[GAN] warm-started generator from {generator_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.gen_lr, betas=(0.8, 0.99))
        d_params = list(self.discriminator.parameters())
        for wd in self._wave_discs:
            d_params += list(wd.parameters())
        opt_d = torch.optim.Adam(d_params, lr=self.disc_lr, betas=(0.8, 0.99))
        if self.lr_decay and self.lr_decay < 1.0:
            sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.lr_decay)
            sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.lr_decay)
            return [opt_g, opt_d], [sch_g, sch_d]
        return [opt_g, opt_d]

    def on_train_epoch_end(self):
        # manual optimization: schedulers must be stepped by hand (HiFi-GAN: per epoch)
        schs = self.lr_schedulers()
        if schs:
            for sch in (schs if isinstance(schs, list) else [schs]):
                sch.step()
            self.log("lr_g", schs[0].get_last_lr()[0], sync_dist=True)

    # ---- combined discriminator forward ----------------------------------------
    # MSSD consumes the log1p STFT pyramid (specs).
    # Wave discriminators (MPD, MS-STFTD, MS-SB-CQTD, …) consume the raw waveform.
    # All scores/feature lists are concatenated so LSGAN + FM losses work unchanged.
    def _disc_all(self, specs, wave, detach=False, return_features=False):
        if return_features:
            sc, fm = self.discriminator(specs, detach=detach, return_features=True)
            for wd in self._wave_discs:
                wsc, wfm = wd(wave, detach=detach, return_features=True)
                sc = sc + wsc; fm = fm + wfm
            return sc, fm
        sc = self.discriminator(specs, detach=detach)
        for wd in self._wave_discs:
            sc = sc + wd(wave, detach=detach)
        return sc

    def _log_mel(self, wav):
        return torch.log(self.mel_spec(wav).clamp_min(1e-5))

    # ---- generator forward + non-adversarial losses (mirrors AE.training_step) ----
    def _synth(self, batch):
        formant_tgt = None
        if len(batch) == 5:
            x, f0_in_hz, f1_tgt, f2_tgt, vmask = batch
            formant_tgt = (f1_tgt, f2_tgt, vmask)
        else:
            x, f0_in_hz = batch

        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)
        params = self.encoder(x, f0=f0_in_hz if self.train_with_true_f0 else None)
        f0_hat = params.pop("f0", None)

        if self.train_with_true_f0:
            random_f0 = f0_in_hz.as_tensor().new_empty(
                f0_in_hz.shape[0], 1).uniform_(50, 500)
            phase = torch.where(f0_in_hz == 0, random_f0, f0_in_hz) / self.sample_rate
        elif self.detach_f0:
            phase = f0_hat.new_tensor(f0_hat.as_tensor().detach()) / self.sample_rate
        else:
            phase = f0_hat / self.sample_rate
        params["phase"] = phase

        voicing_logits = params.pop("voicing_logits", None)
        if voicing_logits is not None:
            voicing = torch.sigmoid(voicing_logits)
            if self.detach_voicing:
                voicing = voicing.detach()
            params["voicing"] = voicing
        elif "voicing" not in params:
            params["voicing"] = (f0_in_hz > 0).float()

        x_hat = self.decoder(**params)

        aux = self.criterion(x_hat[:, : x.shape[1]], x[:, : x_hat.shape[1]]).as_tensor()
        self.log("mss_loss", aux, prog_bar=True, sync_dist=True)
        if (formant_tgt is not None and self.formant_loss_weight > 0
                and self._formant_mod is not None):
            floss, rreg = self._formant_loss(params, *formant_tgt)
            aux = aux + self.formant_loss_weight * floss \
                      + self.residual_reg_weight * rreg
            self.log("formant_loss", floss, prog_bar=True, sync_dist=True)
        if self.mel_weight > 0:
            Tm = min(x_hat.shape[1], x.shape[1])
            mloss = F.l1_loss(self._log_mel(x_hat.as_tensor()[:, :Tm]),
                              self._log_mel(x.as_tensor()[:, :Tm]))
            aux = aux + self.mel_weight * mloss
            self.log("mel_loss", mloss, prog_bar=True, sync_dist=True)
        return x_hat, x, aux

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        x_hat, x, aux = self._synth(batch)

        T = min(x_hat.shape[1], x.shape[1])
        xt = x.as_tensor()[:, :T]
        xht = x_hat.as_tensor()[:, :T]

        use_gan = self.global_step >= self.gan_start_step

        if use_gan:
            # log1p-compress STFT magnitude so discriminator inputs/scores stay in a
            # sane range (raw |STFT| is huge → unstable GAN losses).
            real_specs = [torch.log1p(s) for s in
                          spectrogram_pyramid(xt, self.scales, self.fft_factor, self.overlap)]
            fake_specs = [torch.log1p(s) for s in
                          spectrogram_pyramid(xht, self.scales, self.fft_factor, self.overlap)]

            # --- Discriminator step ---
            opt_d.zero_grad()
            real_scores = self._disc_all(real_specs, xt)
            fake_scores = self._disc_all(fake_specs, xht, detach=True)
            d_loss = lsgan_discriminator_loss(
                real_scores, fake_scores,
                real_label_min=self.real_label_min, real_label_max=self.real_label_max)
            self.manual_backward(d_loss)
            opt_d.step()
            self.log("d_loss", d_loss, prog_bar=True, sync_dist=True)
            # collapse monitors: healthy LSGAN keeps real→1, fake→0 but NOT saturated;
            # d_loss→0 with real≈1/fake≈0 means D won and G gradients vanish
            with torch.no_grad():
                self.log("d_real_score",
                         torch.stack([s.mean() for s in real_scores]).mean(), sync_dist=True)
                self.log("d_fake_score",
                         torch.stack([s.mean() for s in fake_scores]).mean(), sync_dist=True)

        # --- Generator step ---
        opt_g.zero_grad()
        g_loss = aux
        if use_gan:
            if self.fm_weight > 0:
                real_scores, real_fm = self._disc_all(
                    real_specs, xt, detach=True, return_features=True)
                fake_scores, fake_fm = self._disc_all(
                    fake_specs, xht, return_features=True)
                adv = lsgan_generator_loss(fake_scores)
                fm = feature_matching_loss(real_fm, fake_fm)
            else:
                fake_scores = self._disc_all(fake_specs, xht)
                adv = lsgan_generator_loss(fake_scores)
                fm = xt.new_tensor(0.0)
            g_loss = g_loss + self.adv_weight * adv + self.fm_weight * fm
            self.log("adv", adv, prog_bar=True, sync_dist=True)
            self.log("fm", fm, sync_dist=True)
        self.manual_backward(g_loss)
        if self.grad_clip:
            self.clip_gradients(opt_g, self.grad_clip, "norm")
        opt_g.step()
        self.log("train_loss", g_loss, prog_bar=True, sync_dist=True)
