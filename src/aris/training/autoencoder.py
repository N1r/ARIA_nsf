import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Mapping, Tuple, Callable, Union, Any
import numpy as np
import yaml
import math
from collections import defaultdict
from importlib import import_module
from diffsptk import MelCepstralAnalysis, STFT

from aris.models.audiotensor import AudioTensor
from aris.models.sf import SourceFilterSynth
from aris.models.hpn import HarmonicPlusNoiseSynth


class VoiceAutoEncoderCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        return


class VoiceAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        decoder: Union[SourceFilterSynth, HarmonicPlusNoiseSynth],
        criterion: nn.Module,
        encoder_class_path: str,
        encoder_init_args: Dict = {},
        sample_rate: int = 24000,
        detach_f0: bool = False,
        detach_voicing: bool = False,
        train_with_true_f0: bool = True,
        f0_loss_weight: float = 1.0,
        voicing_loss_weight: float = 1.0,
        formant_loss_weight: float = 0.0,
        residual_reg_weight: float = 0.0,
        formant_smooth_weight: float = 0.0,
        aperiodicity_loss_weight: float = 0.0,
    ):
        super().__init__()

        self.decoder = decoder
        self.criterion = criterion

        module_path, class_name = encoder_class_path.rsplit(".", 1)
        module = import_module(module_path)
        split_sizes, trsfms, args_keys = self.decoder.split_sizes_and_trsfms
        self.encoder = getattr(module, class_name)(
            split_sizes=split_sizes,
            trsfms=trsfms,
            args_keys=args_keys,
            **encoder_init_args,
        )

        self.sample_rate = sample_rate
        self.f0_loss_weight = f0_loss_weight
        self.voicing_loss_weight = voicing_loss_weight
        self.formant_loss_weight = formant_loss_weight
        self.residual_reg_weight = residual_reg_weight
        self.formant_smooth_weight = formant_smooth_weight
        self.aperiodicity_loss_weight = aperiodicity_loss_weight
        self.detach_f0 = detach_f0
        self.detach_voicing = detach_voicing
        self.train_with_true_f0 = train_with_true_f0
        # locate the formant-controllable decoder component (ARIA), if any
        self._formant_key, self._formant_mod = self._find_formant_module()

    def _find_formant_module(self):
        for name in ("harm_oscillator", "end_filter", "harm_filter"):
            mod = getattr(self.decoder, name, None)
            if mod is not None and hasattr(mod, "get_formant_params") \
                    and hasattr(mod, "n_formant_ctrl"):
                return name + "_params", mod
        return None, None

    def _formant_loss(self, params, f1_tgt, f2_tgt, vmask):
        """L1(F1,F2) on voiced frames (kHz) + L2 residual reg. Returns (floss, rreg)."""
        ctrl = params[self._formant_key][0]            # AudioTensor (B,T,n_ctrl)
        ctrl_t = ctrl.as_tensor()
        p = self._formant_mod.get_formant_params(ctrl_t)
        f1p, f2p = p["f1"], p["f2"]                     # (B, T_frame)
        T = min(f1p.shape[1], f1_tgt.shape[1])
        f1p, f2p = f1p[:, :T], f2p[:, :T]
        f1t, f2t = f1_tgt[:, :T], f2_tgt[:, :T]
        m = vmask[:, :T] > 0
        if m.any():
            floss = (F.l1_loss(f1p[m] / 1000.0, f1t[m] / 1000.0)
                     + F.l1_loss(f2p[m] / 1000.0, f2t[m] / 1000.0))
        else:
            floss = ctrl_t.new_zeros(())
        residual = ctrl_t[..., self._formant_mod.n_formant_ctrl:]
        rreg = residual.pow(2).mean() if residual.numel() > 0 else ctrl_t.new_zeros(())
        # temporal smoothness: penalise abrupt frame-to-frame formant jumps
        if self.formant_smooth_weight > 0 and f1p.shape[1] > 1:
            sfloss = (F.l1_loss(f1p[:, 1:] / 1000.0, f1p[:, :-1].detach() / 1000.0)
                      + F.l1_loss(f2p[:, 1:] / 1000.0, f2p[:, :-1].detach() / 1000.0))
            floss = floss + self.formant_smooth_weight * sfloss
        return floss, rreg

    def _aperiodicity_loss(self, params, ap_tgt, vmask):
        """L1(A_pred, WORLD-D4C aperiodicity) on voiced frames. A_pred = the v4
        AperiodicityFIRFilter's bounded magnitude sigmoid(logits+freq_bias), [B,T,n_mag].
        Supervises the band-aperiodicity head so the high band dissolves harmonics into
        noise (MSS is texture-blind, so without this the head is unconstrained)."""
        ap_filt = getattr(self.decoder, "aperiodicity_filter", None)
        if ap_filt is None or "aperiodicity_filter_params" not in params:
            return None
        logits = params["aperiodicity_filter_params"][0].as_tensor()   # [B,T,n_mag]
        A_pred = ap_filt._mag(logits)                                   # sigmoid(logits+bias)
        T = min(A_pred.shape[1], ap_tgt.shape[1])
        A_pred, ap_t = A_pred[:, :T], ap_tgt[:, :T]
        m = vmask[:, :T] > 0
        if not m.any():
            return A_pred.new_zeros(())
        return F.l1_loss(A_pred[m], ap_t[m])

    def forward(
        self,
        x: AudioTensor = None,
        f0: AudioTensor = None,
        params: Dict[str, Union[AudioTensor, Tuple[AudioTensor]]] = None,
    ) -> Tuple[AudioTensor, Dict[str, Union[AudioTensor, Tuple[AudioTensor]]]]:
        params = {} if params is None else params
        if x is not None:
            enc_params: Dict = self.encoder(x, f0=f0)
            params.update(enc_params)

            if "phase" not in params:
                params["phase"] = params["f0"] / self.sample_rate

            params.pop("f0", None)

            voicing_logits = params.pop("voicing_logits", None)
            if voicing_logits is not None:
                params["voicing"] = torch.sigmoid(voicing_logits)
            elif f0 is not None and "voicing" not in params:
                params["voicing"] = (f0 > 0).float()   # voiced gate from true F0

        y = self.decoder(**params)
        return y, None if x is None else enc_params

    def f0_loss(self, f0_hat, f0):
        return F.l1_loss(torch.log(f0_hat + 1e-3), torch.log(f0 + 1e-3))

    def training_step(self, batch, batch_idx):
        formant_tgt = None
        ap_tgt = None
        if len(batch) == 6:
            x, f0_in_hz, f1_tgt, f2_tgt, vmask, ap_tgt = batch
            formant_tgt = (f1_tgt, f2_tgt, vmask)
        elif len(batch) == 5:
            x, f0_in_hz, f1_tgt, f2_tgt, vmask = batch
            formant_tgt = (f1_tgt, f2_tgt, vmask)
        else:
            x, f0_in_hz = batch

        # convert to AudioTensor
        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)

        params: Dict = self.encoder(x, f0=f0_in_hz if self.train_with_true_f0 else None)
        f0_hat = params.pop("f0", None)

        if self.train_with_true_f0:
            # phase = torch.where(f0_in_hz == 0, 150, f0_in_hz) / self.sample_rate
            random_f0 = (
                f0_in_hz.as_tensor().new_empty(f0_in_hz.shape[0], 1).uniform_(50, 500)
            )
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
            params["voicing"] = (f0_in_hz > 0).float()   # voiced gate from true F0

        x_hat = self.decoder(**params)
        loss = self.criterion(
            x_hat[:, : x.shape[1]], x[:, : x_hat.shape[1]]
        ).as_tensor()

        if f0_hat is not None:
            f0_target = f0_in_hz[:, :: f0_hat.hop_length].as_tensor()[
                :, : f0_hat.shape[1]
            ]
            f0_pred = f0_hat.as_tensor()[:, : f0_target.shape[1]]
            mask = f0_target > 50
            f0_loss = self.f0_loss(f0_pred[mask], f0_target[mask])
            loss = loss + f0_loss * self.f0_loss_weight
            self.log("train_f0_loss", f0_loss, prog_bar=True, sync_dist=True)

        if voicing_logits is not None:
            voicing_target = (f0_in_hz > 50).float()
            voicing_target = voicing_target[
                :, :: voicing_logits.hop_length
            ].as_tensor()[:, : voicing_logits.shape[1]]
            voicing_logits = voicing_logits.as_tensor()[:, : voicing_target.shape[1]]
            voicing_loss = F.binary_cross_entropy_with_logits(
                voicing_logits, voicing_target, reduction="mean"
            )
            loss = loss + voicing_loss * self.voicing_loss_weight
            self.log("train_voicing_loss", voicing_loss, prog_bar=True, sync_dist=True)

        # formant supervision (ARIA): anchor F1/F2 to Praat targets + residual reg
        if (formant_tgt is not None and self.formant_loss_weight > 0
                and self._formant_mod is not None):
            floss, rreg = self._formant_loss(params, *formant_tgt)
            loss = loss + self.formant_loss_weight * floss \
                        + self.residual_reg_weight * rreg
            self.log("train_formant_loss", floss, prog_bar=True, sync_dist=True)
            self.log("train_residual_reg", rreg, sync_dist=True)

        # band-aperiodicity supervision (v6): anchor A(f) to WORLD D4C targets
        if ap_tgt is not None and self.aperiodicity_loss_weight > 0:
            ap_loss = self._aperiodicity_loss(params, ap_tgt, vmask)
            if ap_loss is not None:
                loss = loss + self.aperiodicity_loss_weight * ap_loss
                self.log("train_ap_loss", ap_loss, prog_bar=True, sync_dist=True)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.tmp_val_outputs = []

    def validation_step(self, batch, batch_idx):
        if len(batch) >= 5:
            x, f0_in_hz = batch[0], batch[1]
        else:
            x, f0_in_hz = batch

        # convert to AudioTensor
        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)

        if self.train_with_true_f0:
            phase = torch.where(f0_in_hz == 0, 150, f0_in_hz) / self.sample_rate
            x_hat, enc_params = self(x, f0_in_hz, {"phase": phase})
        else:
            x_hat, enc_params = self(x)
        loss = self.criterion(x_hat[:, : x.shape[1]], x[:, : x_hat.shape[1]]).item()

        val_outputs = []
        if "f0" in enc_params:
            f0_hat = enc_params["f0"]
            f0_target = f0_in_hz[:, :: f0_hat.hop_length].as_tensor()[
                :, : f0_hat.shape[1]
            ]
            f0_pred = f0_hat.as_tensor()[:, : f0_target.shape[1]]
            mask = f0_target > 50
            f0_loss = self.f0_loss(f0_pred[mask], f0_target[mask]).item()
            loss = loss + f0_loss * self.f0_loss_weight
            val_outputs.append(f0_loss)

        if "voicing_logits" in enc_params:
            voicing_logits = enc_params["voicing_logits"]
            voicing_target = (f0_in_hz > 50).float()
            voicing_target = voicing_target[
                :, :: voicing_logits.hop_length
            ].as_tensor()[:, : voicing_logits.shape[1]]
            voicing_logits = voicing_logits.as_tensor()[:, : voicing_target.shape[1]]
            voicing_loss = F.binary_cross_entropy_with_logits(
                voicing_logits, voicing_target, reduction="mean"
            ).item()
            loss = loss + voicing_loss * self.voicing_loss_weight
            val_outputs.append(voicing_loss)

        val_outputs.insert(0, loss)

        self.tmp_val_outputs.append(val_outputs)

        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self.tmp_val_outputs
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        if len(outputs[0]) > 1:
            avg_f0_loss = sum(x[1] for x in outputs) / len(outputs)
            self.log("val_f0_loss", avg_f0_loss, prog_bar=True, sync_dist=True)

        if len(outputs[0]) > 2:
            avg_voicing_loss = sum(x[2] for x in outputs) / len(outputs)
            self.log(
                "val_voicing_loss", avg_voicing_loss, prog_bar=True, sync_dist=True
            )

        delattr(self, "tmp_val_outputs")

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        return super().load_state_dict(state_dict, False)

    def on_test_start(self) -> None:
        self.tmp_test_outputs = defaultdict(list)
        self.mcep = nn.Sequential(
            STFT(512, self.sample_rate // 200, 512, window="hanning"),
            MelCepstralAnalysis(34, 512, alpha=0.46),
        ).to(self.device)

        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        x, f0_in_hz = batch

        # convert to AudioTensor
        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)  # .set_hop_length(self.sample_rate // 200)

        if self.train_with_true_f0:
            phase = torch.where(f0_in_hz == 0, 150, f0_in_hz) / self.sample_rate
            x_hat, enc_params = self(x, f0_in_hz, {"phase": phase})
        else:
            x_hat, enc_params = self(x)
        loss = self.criterion(x_hat[:, : x.shape[1]], x[:, : x_hat.shape[1]]).item()
        N = x_hat.shape[0]
        tmp_dict = {"loss": loss, "N": N}

        # get mceps
        x_mceps = self.mcep(x.as_tensor())
        x_hat_mceps = self.mcep(x_hat.as_tensor())
        x_mceps, x_hat_mceps = (
            x_mceps[:, : x_hat_mceps.shape[1]],
            x_hat_mceps[:, : x_mceps.shape[1]],
        )
        mcd = (
            10
            * math.sqrt(2)
            / math.log(10)
            * torch.linalg.norm(x_mceps - x_hat_mceps, dim=-1).mean().item()
        )
        tmp_dict["mcd"] = mcd

        list(
            map(lambda x: self.tmp_test_outputs[x].append(tmp_dict[x]), tmp_dict.keys())
        )

        return tmp_dict

    def on_test_epoch_end(self) -> None:
        outputs = self.tmp_test_outputs
        log_dict = {}
        weights = outputs["N"]
        avg_mss_loss = np.average(outputs["loss"], weights=weights)
        log_dict["avg_mss_loss"] = avg_mss_loss

        avg_mcd = np.average(outputs["mcd"], weights=weights)
        log_dict["avg_mcd"] = avg_mcd

        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
        )
        delattr(self, "tmp_test_outputs")
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, f0_in_hz, rel_path = batch

        assert x.shape[0] == 1, "batch size must be 1 for inference"

        # convert to AudioTensor
        x = AudioTensor(x)
        f0_in_hz = AudioTensor(f0_in_hz)

        if self.train_with_true_f0:
            phase = torch.where(f0_in_hz == 0, 150, f0_in_hz) / self.sample_rate
            x_hat, enc_params = self(x, f0_in_hz, {"phase": phase})
        else:
            x_hat, enc_params = self(x)

        return x_hat, enc_params
