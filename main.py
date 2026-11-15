from __future__ import annotations

import json
import hashlib
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

import hydra
import lightning as L
import torch
from dataset import GediSentinelDataModule
from lightning_model import DepthAnythingV2Module
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
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
    data_started = time.perf_counter()
    data_module.setup("fit")
    data_setup_seconds = time.perf_counter() - data_started
    # Disk patches retain 14 Sentinel bands; context adds two channels at load time.
    config.model.in_channels = data_module.input_channels
    model_config["in_channels"] = data_module.input_channels
    model_config["target_scale"] = data_module.target_scale
    model_config["train_target_mean"] = data_module.train_target_mean
    train_counts = data_module.split_summary["counts"]["train"]
    model_config["train_mean_valid_pixels"] = train_counts["valid_label_pixels"] / train_counts["labeled_patches"]
    model_started = time.perf_counter()
    model = DepthAnythingV2Module(**model_config)
    model_setup_seconds = time.perf_counter() - model_started
    output_dir = Path(resolve_repository_path(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "target": "agbd", "target_units": "Mg/ha",
        "target_scale": data_module.target_scale,
        "train_target_mean": data_module.train_target_mean,
        "loss_setup": {"loss_function": "mse"},
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
    early_stopping = EarlyStopping(
        monitor="val_RMSE", mode="min", patience=config.early_stopping.patience,
        min_delta=config.early_stopping.min_delta, check_finite=True,
        check_on_train_epoch_end=False, verbose=True,
    ) if config.early_stopping.enabled else None
    callbacks = [callback for callback in (checkpoint, early_stopping) if callback is not None]
    trainer = L.Trainer(
        **config.trainer,
        logger=CSVLogger(save_dir=str(output_dir), name="history", version=""),
        callbacks=callbacks,
    )

    fit_started = time.perf_counter()
    trainer.fit(model, datamodule=data_module)
    fit_seconds = time.perf_counter() - fit_started
    rmse = trainer.callback_metrics.get("val_RMSE")
    valid_pixels = trainer.callback_metrics.get("val_valid_pixels")
    if rmse is None or valid_pixels is None:
        raise RuntimeError("Validation did not produce val_RMSE and val_valid_pixels.")
    print(
        f"TRAINING FINISHED: val_RMSE={float(rmse):.4f} Mg/ha, "
        f"valid_GEDI_pixels={int(valid_pixels)}"
    )
    metrics = {key: float(value) for key, value in trainer.callback_metrics.items()}
    metrics["training_summary"] = {
        "epochs_completed": trainer.current_epoch,
        "optimizer_steps": trainer.global_step,
        "best_val_RMSE": float(checkpoint.best_model_score) if checkpoint else None,
        "best_checkpoint": checkpoint.best_model_path if checkpoint else None,
        "early_stopping_enabled": early_stopping is not None,
        "stopped_early": bool(early_stopping and early_stopping.stopped_epoch > 0),
        "early_stopping_patience": config.early_stopping.patience,
        "early_stopping_min_delta_mg_ha": config.early_stopping.min_delta,
        "fit_seconds": fit_seconds,
        "loss_weighting": model.hparams.loss_weighting,
        "loss_function": "mse",
        "train_mean_valid_pixels": model.hparams.train_mean_valid_pixels,
    }
    test_started = time.perf_counter()
    if config.run_test:
        # Independent test is evaluated only on request, after model selection.
        metrics["independent_test"] = trainer.test(
            model, datamodule=data_module,
            ckpt_path=checkpoint.best_model_path if checkpoint else None,
        )
    metrics["runtime"] = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "test_seconds": time.perf_counter() - test_started if config.run_test else 0.0,
        "torch_version": torch.__version__,
        "accelerator": type(trainer.accelerator).__name__,
        "cpu_threads": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("torch", "torchvision", "lightning", "transformers", "numpy", "rasterio")},
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "data_setup_seconds": data_setup_seconds,
        "model_setup_seconds": model_setup_seconds,
        "source_sha256": {name: hashlib.sha256((repository_root / name).read_bytes()).hexdigest()
                          for name in ("main.py", "lightning_model.py", "dataset/sentinel_gedi_canopy.py")},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
