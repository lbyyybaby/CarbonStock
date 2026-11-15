from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Metric, SumMetric


class ValidPixelRMSE(Metric):
    """Pixel-weighted RMSE whose state is reduced correctly across all batches."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "squared_error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("pixel_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        error = prediction.float() - target.float()
        self.squared_error_sum += torch.square(error).sum()
        self.pixel_count += target.numel()

    def compute(self) -> torch.Tensor:
        return torch.sqrt(self.squared_error_sum / self.pixel_count.clamp_min(1))


class DepthAnythingV2Module(L.LightningModule):
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    size_map = {
        "vits": "DarthReca/depth-any-canopy-small",
        "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
        "vitg": None,
    }

    def __init__(
        self,
        architecture: Literal["depth_anything"] = "depth_anything",
        encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vits",
        in_channels: int = 14,
        image_size: int = 252,
        target_scale: float = 1.0,
        train_target_mean: float = 0.0,
        lr: float = 3e-5,
        pretrained: bool = False,
        use_huggingface: bool = False,
        hf_model_path: str | None = None,
        hf_local_files_only: bool = True,
        checkpoint_path: str | None = None,
        loss_weighting: Literal["batch", "pixel"] = "pixel",
        train_mean_valid_pixels: float | None = None,
        scheduler: Literal["plateau"] = "plateau",
        plateau_patience: int = 2,
        plateau_factor: float = 0.5,
        plateau_min_lr: float = 1e-7,
        **legacy_hparams: object,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if architecture != "depth_anything":
            raise ValueError("Only the depth_anything ViT architecture is supported.")
        # Older Plateau checkpoints contain options from exploratory runs.
        # Accept their inactive defaults, and reject changed inference semantics.
        for name, value in {
            "input_adaptation": "random", "reset_output_head": False,
            "output_activation": "relu", "backbone_lr_factor": 1.0,
            "augmentation": "none", "loss_function": "mse",
        }.items():
            if legacy_hparams.get(name, value) != value:
                raise ValueError(f"Only {name}={value!r} is supported.")
        if not math.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("model.target_scale must be finite and positive.")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("model.lr must be finite and positive.")
        if loss_weighting not in {"batch", "pixel"}:
            raise ValueError("model.loss_weighting must be 'batch' or 'pixel'.")
        if loss_weighting == "pixel" and (
            train_mean_valid_pixels is None
            or not math.isfinite(train_mean_valid_pixels)
            or train_mean_valid_pixels <= 0
        ):
            raise ValueError("Pixel weighting requires a finite positive model.train_mean_valid_pixels.")
        if scheduler != "plateau":
            raise ValueError("Only model.scheduler=plateau is supported.")
        if isinstance(plateau_patience, bool) or not isinstance(plateau_patience, int) or plateau_patience < 0:
            raise ValueError("model.plateau_patience must be a nonnegative integer.")
        if not math.isfinite(plateau_factor) or not 0 < plateau_factor < 1:
            raise ValueError("model.plateau_factor must be finite and between zero and one.")
        if not math.isfinite(plateau_min_lr) or plateau_min_lr < 0:
            raise ValueError("model.plateau_min_lr must be finite and nonnegative.")
        self._uses_huggingface = use_huggingface

        if use_huggingface:
            if self.size_map[encoder] is None:
                raise ValueError(f"No Hugging Face model is configured for {encoder}.")
            import transformers

            model_source = hf_model_path or self.size_map[encoder]
            if hf_local_files_only:
                local_model_path = Path(model_source).expanduser()
                if not local_model_path.is_dir():
                    raise FileNotFoundError(
                        f"Offline Hugging Face model directory not found: {local_model_path}"
                    )
                model_source = str(local_model_path.resolve())
            hf_config = transformers.AutoConfig.from_pretrained(
                model_source,
                local_files_only=hf_local_files_only,
            )
            pretrained_channels = hf_config.backbone_config.num_channels
            hf_config.backbone_config.num_channels = in_channels
            if pretrained:
                self.model, loading_info = (
                    transformers.AutoModelForDepthEstimation.from_pretrained(
                        model_source,
                        config=hf_config,
                        ignore_mismatched_sizes=in_channels != pretrained_channels,
                        local_files_only=hf_local_files_only,
                        output_loading_info=True,
                    )
                )
                expected_mismatches = (
                    {"backbone.embeddings.patch_embeddings.projection.weight"}
                    if in_channels != pretrained_channels
                    else set()
                )
                # Transformers versions expose mismatched keys either as plain
                # strings or as tuples/dicts containing the parameter name.
                def mismatch_name(item: object) -> str | None:
                    if isinstance(item, str):
                        return item
                    if isinstance(item, dict):
                        value = item.get("key") or item.get("name")
                        return value if isinstance(value, str) else None
                    if isinstance(item, (tuple, list)) and item:
                        return item[0] if isinstance(item[0], str) else None
                    return None

                actual_mismatches = {
                    mismatch_name(item) for item in loading_info["mismatched_keys"]
                }
                if (
                    loading_info["missing_keys"]
                    or loading_info["unexpected_keys"]
                    or loading_info["error_msgs"]
                    or actual_mismatches != expected_mismatches
                ):
                    raise RuntimeError(
                        "Unexpected Hugging Face checkpoint loading result: "
                        f"{loading_info}"
                    )
            else:
                self.model = transformers.AutoModelForDepthEstimation.from_config(hf_config)
        else:
            from model import DepthAnythingV2

            self.model = DepthAnythingV2(
                **self.model_configs[encoder], in_channels=in_channels
            )
            if pretrained:
                if not checkpoint_path:
                    raise ValueError(
                        "model.checkpoint_path is required when pretrained=true and "
                        "use_huggingface=false."
                    )
                self._load_compatible_checkpoint(checkpoint_path)

        self.train_rmse = ValidPixelRMSE()
        self.val_rmse = ValidPixelRMSE()
        self.test_rmse = ValidPixelRMSE()
        self.val_mean_baseline_rmse = ValidPixelRMSE()
        self.test_mean_baseline_rmse = ValidPixelRMSE()
        self.train_pixels = SumMetric()
        self.val_pixels = SumMetric()
        self.test_pixels = SumMetric()

    def _load_compatible_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        current = self.model.state_dict()
        compatible = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.removeprefix("model.") in current
            and current[key.removeprefix("model.")].shape == value.shape
        }
        if not compatible:
            raise RuntimeError(
                "Checkpoint has no tensors compatible with the native model. "
                "Use model.use_huggingface=true for a Hugging Face checkpoint."
            )
        self.model.load_state_dict(compatible, strict=False)
        print(f"Loaded {len(compatible)}/{len(current)} compatible checkpoint tensors.")

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.hparams.plateau_factor,
            patience=self.hparams.plateau_patience, threshold=0.1,
            threshold_mode="abs", min_lr=self.hparams.plateau_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, "interval": "epoch",
                "monitor": "val_RMSE", "reduce_on_plateau": True,
                "strict": True,
            },
        }

    def _predict(
        self,
        image: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        model_input_size = (self.hparams.image_size, self.hparams.image_size)
        if image.shape[-2:] != model_input_size:
            image = F.interpolate(
                image,
                size=model_input_size,
                mode="bilinear",
                align_corners=False,
            )
        output = self.model(image)
        prediction = output.predicted_depth if self._uses_huggingface else output
        if prediction.shape[-2:] != output_size:
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return prediction

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: Literal["train", "val", "test"]
    ) -> torch.Tensor | None:
        target_agbd = batch["mask"]
        target_normalized = target_agbd / self.hparams.target_scale
        raw_prediction_normalized = self._predict(
            batch["image"], tuple(target_agbd.shape[-2:])
        )
        valid = (
            batch["valid_mask"]
            & torch.isfinite(target_agbd)
            & (target_agbd >= 0)
        )
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            # Unlabeled map patches can pass through prediction or a mixed loader.
            return None
        if not torch.isfinite(raw_prediction_normalized[valid]).all():
            raise FloatingPointError(f"Non-finite AGBD prediction during {stage}.")

        valid_raw_prediction = raw_prediction_normalized[valid]
        valid_target_normalized = target_normalized[valid]
        mse = (valid_raw_prediction - valid_target_normalized).square().mean()
        optimization_loss = mse
        if stage == "train" and self.hparams.loss_weighting == "pixel":
            # A fixed training-only denominator avoids giving sparse batches
            # the same total influence as dense batches. Under uniform patch
            # sampling this estimates the pooled valid-pixel MSE objective.
            denominator = self.hparams.train_mean_valid_pixels * target_agbd.shape[0]
            optimization_loss = optimization_loss * (valid_count / denominator)
        prediction_agbd = (
            raw_prediction_normalized.detach() * self.hparams.target_scale
        )
        rmse_metric: ValidPixelRMSE = getattr(self, f"{stage}_rmse")
        pixel_metric: SumMetric = getattr(self, f"{stage}_pixels")
        rmse_metric.update(prediction_agbd[valid], target_agbd[valid])
        pixel_metric.update(torch.tensor(valid_count, device=self.device))
        if stage != "train":
            baseline_metric = getattr(self, f"{stage}_mean_baseline_rmse")
            baseline_metric.update(
                torch.full_like(target_agbd[valid], self.hparams.train_target_mean),
                target_agbd[valid],
            )
            self.log(
                f"{stage}_mean_baseline_RMSE", baseline_metric,
                on_step=False, on_epoch=True,
            )

        self.log(
            f"{stage}_Loss(MSE)",
            mse,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage == "train",
            batch_size=valid_count,
        )
        self.log(f"{stage}_Loss(objective)", optimization_loss,
                 on_step=False, on_epoch=True, batch_size=valid_count)
        if stage == "train":
            valid_raw_prediction = valid_raw_prediction.detach().float()
            diagnostics = {
                "train_raw_pred_mean_normalized": valid_raw_prediction.mean(),
                "train_below_0_pct": (valid_raw_prediction < 0).float().mean() * 100,
                "train_zero_pct": (valid_raw_prediction == 0).float().mean() * 100,
            }
            for name, value in diagnostics.items():
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=valid_count,
                )
            self.log(
                "lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=valid_count,
            )
        self.log(
            f"{stage}_RMSE",
            rmse_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}_valid_pixels",
            pixel_metric,
            on_step=False,
            on_epoch=True,
        )
        return optimization_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor | None:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        self.print(
            f"Validation: RMSE={self.val_rmse.compute().item():.4f} Mg/ha on "
            f"{int(self.val_pixels.compute().item())} valid GEDI pixels"
        )

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor | list[str]]:
        target = batch["mask"]
        return {
            "patch_id": batch["patch_id"],
            "prediction_agbd": self._predict(
                batch["image"], tuple(target.shape[-2:])
            )
            * self.hparams.target_scale,
            "valid_mask": batch["valid_mask"],
        }
