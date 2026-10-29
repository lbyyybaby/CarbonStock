from __future__ import annotations

import json
from pathlib import Path

import hydra
import lightning as L
import torch
from dataset import GediSentinelDataModule
from lightning_model import DepthAnythingV2Module
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint


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
    # Split first: neither normalization nor biomass scale may see val/test.
    data_module.setup("fit")
    model_config["target_scale"] = data_module.target_scale
    model_config["train_target_mean"] = data_module.train_target_mean
    model = DepthAnythingV2Module(**model_config)
    output_dir = Path(resolve_repository_path(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "target": "agbd", "target_units": "Mg/ha",
        "target_scale": data_module.target_scale,
        "train_target_mean": data_module.train_target_mean,
        "normalization_mean": data_module.normalization_mean.tolist(),
        "normalization_std": data_module.normalization_std.tolist(),
        "split": data_module.split_summary,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    (output_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints", monitor="val_RMSE", mode="min",
        save_top_k=1, filename="best-{epoch:03d}",
    ) if config.trainer.enable_checkpointing else None
    trainer = L.Trainer(
        **config.trainer, logger=False,
        callbacks=[checkpoint] if checkpoint else [],
    )

    trainer.fit(model, datamodule=data_module)
    rmse = trainer.callback_metrics.get("val_RMSE")
    valid_pixels = trainer.callback_metrics.get("val_valid_pixels")
    if rmse is None or valid_pixels is None:
        raise RuntimeError("Validation did not produce val_RMSE and val_valid_pixels.")
    print(
        f"TRAINING FINISHED: val_RMSE={float(rmse):.4f} Mg/ha, "
        f"valid_GEDI_pixels={int(valid_pixels)}"
    )
    metrics = {key: float(value) for key, value in trainer.callback_metrics.items()}
    if config.run_test:
        # Independent test is evaluated only on request, after model selection.
        metrics["independent_test"] = trainer.test(
            model, datamodule=data_module,
            ckpt_path=checkpoint.best_model_path if checkpoint else None,
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
