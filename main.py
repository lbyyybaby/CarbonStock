from __future__ import annotations

import hydra
import lightning as L
import torch
from dataset import GediSentinelDataModule
from hydra.utils import to_absolute_path
from lightning_model import DepthAnythingV2Module
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    dataset_config["data_root"] = to_absolute_path(dataset_config["data_root"])
    data_module = GediSentinelDataModule(**dataset_config)
    model = DepthAnythingV2Module(**config.model)
    trainer = L.Trainer(**config.trainer, logger=False)

    trainer.fit(model, datamodule=data_module)
    rmse = trainer.callback_metrics.get("val_RMSE")
    valid_pixels = trainer.callback_metrics.get("val_valid_pixels")
    if rmse is None or valid_pixels is None:
        raise RuntimeError("Validation did not produce val_RMSE and val_valid_pixels.")
    print(
        f"SMOKE TEST PASSED: val_RMSE={float(rmse):.4f} m, "
        f"valid_GEDI_pixels={int(valid_pixels)}"
    )


if __name__ == "__main__":
    main()
