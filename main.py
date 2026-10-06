from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import torch
from dataset import GediSentinelDataModule
from lightning_model import DepthAnythingV2Module
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    repository_root = Path(__file__).resolve().parent

    def resolve_repository_path(value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = repository_root / path
        return str(path.resolve())

    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    dataset_config["data_root"] = resolve_repository_path(dataset_config["data_root"])
    model_config = OmegaConf.to_container(config.model, resolve=True)
    for path_key in ("hf_model_path", "checkpoint_path"):
        if model_config.get(path_key):
            model_config[path_key] = resolve_repository_path(model_config[path_key])
    data_module = GediSentinelDataModule(**dataset_config)
    model = DepthAnythingV2Module(**model_config)
    trainer = L.Trainer(**config.trainer, logger=False)

    trainer.fit(model, datamodule=data_module)
    rmse = trainer.callback_metrics.get("val_RMSE")
    valid_pixels = trainer.callback_metrics.get("val_valid_pixels")
    if rmse is None or valid_pixels is None:
        raise RuntimeError("Validation did not produce val_RMSE and val_valid_pixels.")
    print(
        f"TRAINING FINISHED: val_RMSE={float(rmse):.4f} m, "
        f"valid_GEDI_pixels={int(valid_pixels)}"
    )


if __name__ == "__main__":
    main()
