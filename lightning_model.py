from __future__ import annotations

from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
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


class TinyCanopyRegressor(nn.Module):
    """Fast offline model used only to exercise the complete pipeline."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image).squeeze(1)


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
        architecture: Literal["tiny", "depth_anything"] = "tiny",
        encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vits",
        in_channels: int = 14,
        # image_size: int = 64,
        # max_height: float = 30.0,
        # lr: float = 1e-3,
        pretrained: bool = False,
        use_huggingface: bool = False,
        checkpoint_path: str | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if architecture == "tiny":
            self.model = TinyCanopyRegressor(in_channels)
        elif use_huggingface:
            if self.size_map[encoder] is None:
                raise ValueError(f"No Hugging Face model is configured for {encoder}.")
            import transformers

            self.model = transformers.AutoModelForDepthEstimation.from_pretrained(
                self.size_map[encoder],
                num_channels=in_channels,
                ignore_mismatched_sizes=in_channels != 3,
                cache_dir="cache",
            )
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
        self.model.load_state_dict(compatible, strict=False)
        print(f"Loaded {len(compatible)}/{len(current)} compatible checkpoint tensors.")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def _predict(self, image: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        image = F.interpolate(
            image,
            size=(self.hparams.image_size, self.hparams.image_size),
            mode="bilinear",
            align_corners=False,
        )
        output = self.model(image)
        prediction = output.predicted_depth if self.hparams.use_huggingface else output
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return prediction.clamp(0.0, self.hparams.max_height)

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: Literal["train", "val", "test"]
    ) -> torch.Tensor:
        target = batch["mask"]
        prediction = self._predict(batch["image"], tuple(target.shape[-2:]))
        valid = batch["valid_mask"] & torch.isfinite(target) & torch.isfinite(prediction)
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            raise RuntimeError(f"Batch has no valid GEDI pixels during {stage}.")

        valid_prediction = prediction[valid]
        valid_target = target[valid]
        mse = F.mse_loss(valid_prediction, valid_target)
        rmse_metric: ValidPixelRMSE = getattr(self, f"{stage}_rmse")
        pixel_metric: SumMetric = getattr(self, f"{stage}_pixels")
        rmse_metric.update(valid_prediction, valid_target)
        pixel_metric.update(torch.tensor(valid_count, device=self.device))

        self.log(
            f"{stage}_Loss(MSE)",
            mse,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
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
        return mse

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        self.print(
            f"Validation: RMSE={self.val_rmse.compute().item():.4f} m on "
            f"{int(self.val_pixels.compute().item())} valid GEDI pixels"
        )

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor | list[str]]:
        target = batch["mask"]
        return {
            "patch_id": batch["patch_id"],
            "prediction_m": self._predict(batch["image"], tuple(target.shape[-2:])),
            "valid_mask": batch["valid_mask"],
        }
