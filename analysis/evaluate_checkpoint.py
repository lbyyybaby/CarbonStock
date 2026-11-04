"""Evaluate a saved AGBD checkpoint using its original split and normalization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

from .analyze_run import (REPOSITORY, audit_dataset, load_run, plt, read_history,
                          regression_metrics, resolve_data_root, write_csv, write_json)


def checkpoint_path(run, metrics, override):
    if override:
        path = override.expanduser().resolve()
    else:
        recorded = metrics.get("training_summary", {}).get("best_checkpoint")
        if not recorded:
            raise ValueError("No best checkpoint recorded. Specify --checkpoint.")
        name = PurePosixPath(recorded.replace("\\", "/")).name
        path = run / "checkpoints" / name
        if not path.is_file() and Path(recorded).is_file():
            path = Path(recorded)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Set --checkpoint.")
    return path.resolve()


def load_model(path, summary, hf_override):
    import torch
    from lightning_model import DepthAnythingV2Module

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    parameters = dict(checkpoint["hyper_parameters"])
    epoch = checkpoint.get("epoch")
    state = checkpoint["state_dict"]
    del checkpoint  # Discard optimizer states; evaluation does not resume training.
    for key in ("target_scale", "train_target_mean"):
        if not np.isclose(parameters[key], summary[key], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Checkpoint and data_summary.json disagree about {key}.")
    for key in ("architecture", "encoder", "in_channels", "image_size", "use_huggingface"):
        if parameters.get(key) != summary["config"]["model"].get(key):
            raise ValueError(f"Checkpoint and training configuration disagree about {key}.")
    # Build the architecture from local config, then load ALL trained weights strictly.
    # Reloading the original pretrained RGB weights is unnecessary here.
    parameters.update(pretrained=False, checkpoint_path=None, hf_local_files_only=True)
    if parameters.get("use_huggingface") and parameters.get("architecture") != "tiny":
        candidates = [hf_override] if hf_override else [REPOSITORY / summary["config"]["model"]["hf_model_path"],
                                                       Path(parameters["hf_model_path"])]
        existing = [Path(p).expanduser().resolve() for p in candidates if p is not None and (Path(p).expanduser() / "config.json").is_file()]
        if not existing:
            raise FileNotFoundError("Local Hugging Face config not found. Set --hf-model-path to the offline model folder.")
        parameters["hf_model_path"] = str(existing[0])
    model = DepthAnythingV2Module(**parameters)
    model.load_state_dict(state, strict=True)
    return model, epoch


def plot_predictions(output, split, target, prediction, metrics):
    residual = prediction - target
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    lower, upper = float(min(target.min(), prediction.min(), 0)), float(max(target.max(), prediction.max(), 1))
    axes[0].scatter(target, prediction, s=5, alpha=.2, rasterized=True)
    axes[0].plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axes[0].set(xlabel="GEDI AGBD (Mg/ha)", ylabel="Predicted AGBD (Mg/ha)", xlim=(lower, upper), ylim=(lower, upper))
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].scatter(target, residual, s=5, alpha=.2, rasterized=True)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xlabel="GEDI AGBD (Mg/ha)", ylabel="Prediction - GEDI (Mg/ha)")
    axes[2].hist(residual, bins=60, color="#2563eb", alpha=.8)
    axes[2].axvline(0, color="black", linestyle="--")
    axes[2].set(xlabel="Residual (Mg/ha)", ylabel="Valid pixel count")
    figure.suptitle(f"{split}: n={metrics['n']}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, bias={metrics['bias']:.2f} Mg/ha")
    figure.savefig(output / f"predictions_{split}.png", dpi=160)
    plt.close(figure)


def plot_patch_errors(output, patch_rows, splits):
    figure, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), squeeze=False, constrained_layout=True)
    vmax = max(max(row["rmse"] for row in patch_rows), 1e-6)
    for axis, split in zip(axes.flat, splits):
        rows = [r for r in patch_rows if r["split"] == split]
        image = axis.scatter([r["center_x_m"] for r in rows], [r["center_y_m"] for r in rows],
                             c=[r["rmse"] for r in rows], s=[25 + math.sqrt(r["n"]) * 3 for r in rows],
                             cmap="magma", vmin=0, vmax=vmax, alpha=.85)
        axis.set(title=f"{split}: labeled patch centers", xlabel="Projected easting (m)", ylabel="Projected northing (m)")
        axis.set_aspect("equal", adjustable="datalim")
    figure.colorbar(image, ax=list(axes.flat), label="Patch RMSE (Mg/ha)")
    figure.savefig(output / "patch_error_map.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/agbd"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--country-prefix")
    parser.add_argument("--hf-model-path", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.num_workers < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("Threads and batch size must be positive; workers must be nonnegative.")
    started = time.perf_counter()
    run = args.run_dir.expanduser().resolve()
    summary, original_metrics = load_run(run)
    data_root = resolve_data_root(summary, args.data_root)
    patches, audit = audit_dataset(summary, data_root, args.country_prefix)
    splits = list(dict.fromkeys(args.splits))
    selected = [p for p in patches if p.split in splits and p.valid_pixels > 0]
    for split in splits:
        if not any(p.split == split for p in selected):
            raise ValueError(f"No labeled patches in selected split {split}.")

    import torch
    from torch.utils.data import DataLoader
    from dataset.sentinel_gedi_canopy import GediSentinelDataset, PatchPair
    from rasterio.warp import transform

    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("medium")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    path = checkpoint_path(run, original_metrics, args.checkpoint)
    model, checkpoint_epoch = load_model(path, summary, args.hf_model_path)
    if args.checkpoint is None:
        history, _ = read_history(run)
        best_epoch = min((r for r in history if "val_RMSE" in r), key=lambda r: r["val_RMSE"])["epoch"]
        if checkpoint_epoch != best_epoch:
            raise ValueError("The default checkpoint epoch does not match the best epoch in the CSV.")
    model.to(device).eval()
    batch_size = args.batch_size or summary["config"]["dataset"]["batch_size"]
    by_id = {p.saved_id: p for p in selected}
    pairs = [PatchPair(region=str(PurePosixPath(p.saved_id).parent), name=p.name,
                       sentinel_path=p.directory / "input" / p.name, gedi_path=p.directory / "gedi" / p.name,
                       input_valid_mask_path=p.directory / "input_valid_mask" / p.name,
                       metadata_path=p.directory / "metadata" / Path(p.name).with_suffix(".json"),
                       bounds=p.bounds, valid_pixels=p.valid_pixels) for p in selected]
    dataset = GediSentinelDataset(pairs, np.asarray(summary["normalization_mean"], dtype=np.float32),
                                 np.asarray(summary["normalization_std"], dtype=np.float32),
                                 max_agbd=summary["config"]["dataset"].get("max_agbd"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device == "cuda")
    output = (args.output_dir or run / "evaluation").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pixel_fields = ["split", "saved_patch_id", "resolved_patch_id", "country", "province", "site", "period",
                    "patch_row", "patch_col", "source_row", "source_col", "source_x", "source_y", "source_crs",
                    "longitude", "latitude", "target_agbd", "prediction_agbd", "residual"]
    temporary_csv = output / "predictions.csv.tmp"
    grouped, group_metadata, patch_rows = {s: [] for s in splits}, {}, []
    baseline, processed = float(summary["train_target_mean"]), 0
    print(f"Evaluating checkpoint epoch {checkpoint_epoch} on {device}: {len(selected)} labeled patches, splits={splits}", flush=True)
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream, torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(pixel_fields)
        for batch_index, batch in enumerate(loader):
            inference_batch = {**batch, "image": batch["image"].to(device)}
            predictions = model.predict_step(inference_batch, batch_index)["prediction_agbd"].cpu().numpy()
            targets = batch["mask"].numpy()
            masks = batch["valid_mask"].numpy()
            for index, saved_id in enumerate(batch["patch_id"]):
                patch = by_id[saved_id]
                valid = masks[index] & np.isfinite(targets[index]) & (targets[index] >= 0)
                truth = targets[index][valid].astype(np.float64)
                prediction = predictions[index][valid].astype(np.float64)
                if truth.size != patch.valid_pixels or not np.isfinite(prediction).all():
                    raise ValueError(f"Invalid predictions or changed labels for {saved_id}.")
                grouped[patch.split].append((truth, prediction))
                metadata = patch.metadata
                tags = {name: metadata.get(key, "") for name, key in (("country", "country_id"), ("province", "province_id"),
                                                                       ("site", "site_id"), ("period", "window_id"))}
                for name, value in tags.items():
                    group_metadata.setdefault((patch.split, name, str(value)), []).append((truth, prediction))
                patch_rows.append({"split": patch.split, "saved_patch_id": saved_id, "resolved_patch_id": patch.resolved_id,
                                   **tags, "center_x_m": (patch.bounds[0] + patch.bounds[2]) / 2,
                                   "center_y_m": (patch.bounds[1] + patch.bounds[3]) / 2,
                                   **regression_metrics(truth, prediction, baseline)})
                rows, cols = np.nonzero(valid)
                a, b, c, d, e, f = metadata["transform"]
                x = a * (cols + .5) + b * (rows + .5) + c
                y = d * (cols + .5) + e * (rows + .5) + f
                crs = metadata["crs"]
                crs = crs["value"] if isinstance(crs, dict) else crs
                longitude, latitude = transform(crs, "EPSG:4326", x.tolist(), y.tolist())
                window = metadata["source_window"]
                prefix = [patch.split, saved_id, patch.resolved_id, *tags.values()]
                for pixel in range(truth.size):
                    writer.writerow(prefix + [int(rows[pixel]), int(cols[pixel]), int(rows[pixel] + window["row_off"]),
                        int(cols[pixel] + window["col_off"]), float(x[pixel]), float(y[pixel]), crs,
                        longitude[pixel], latitude[pixel], truth[pixel], prediction[pixel], prediction[pixel] - truth[pixel]])
                processed += 1
            if (batch_index + 1) % 5 == 0 or processed == len(selected):
                print(f"Predicted {processed}/{len(selected)} patches", flush=True)

    overall, biomass_rows = {}, []
    for split in splits:
        truth = np.concatenate([x for x, _ in grouped[split]])
        prediction = np.concatenate([y for _, y in grouped[split]])
        result = regression_metrics(truth, prediction, baseline)
        if result["n"] != audit["counts"][split]["valid_label_pixels"]:
            raise ValueError(f"Evaluated pixel count does not match the saved split {split}.")
        subset = [r for r in patch_rows if r["split"] == split]
        result.update(labeled_patches=len(subset), mean_patch_rmse=float(np.mean([r["rmse"] for r in subset])),
                      median_patch_rmse=float(np.median([r["rmse"] for r in subset])),
                      largest_patch_label_share_pct=max(r["n"] for r in subset) / result["n"] * 100)
        overall[split] = result
        for lower, upper in ((0, 30), (30, 100), (100, 200), (200, math.inf)):
            valid = (truth >= lower) & (truth < upper)
            scores = regression_metrics(truth[valid], prediction[valid], baseline)
            biomass_rows.append({"split": split, "agbd_bin": f"[{lower}, {upper})", **scores,
                                 "squared_error_share_pct": scores["squared_error_sum"] / result["squared_error_sum"] * 100 if result["squared_error_sum"] else 0.})
        plot_predictions(output, split, truth, prediction, result)
    metadata_rows = []
    for (split, kind, value), parts in sorted(group_metadata.items()):
        truth, prediction = np.concatenate([x for x, _ in parts]), np.concatenate([y for _, y in parts])
        metadata_rows.append({"split": split, "group_type": kind, "group": value,
                              "labeled_patches": len(parts), **regression_metrics(truth, prediction, baseline)})
    patch_rows.sort(key=lambda r: (r["split"], -r["rmse"]))
    write_csv(output / "per_patch.csv", patch_rows)
    write_csv(output / "by_agbd_bin.csv", biomass_rows)
    write_csv(output / "by_region_period.csv", metadata_rows)
    plot_patch_errors(output, patch_rows, splits)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run), "checkpoint": str(path),
              "checkpoint_sha256": digest.hexdigest(), "checkpoint_epoch_index": checkpoint_epoch,
              "splits_evaluated": splits, "target_units": "Mg/ha", "residual_definition": "prediction - target",
              "model_mode": "eval, inference_mode, float32; no training or prediction clipping",
              "device": device, "torch_version": torch.__version__, "original_torch_version": original_metrics.get("runtime", {}).get("torch_version"),
              "target_scale": summary["target_scale"], "train_target_mean": baseline, "data_audit": audit,
              "metrics": overall, "elapsed_seconds": time.perf_counter() - started,
              "notes": ["Mean/median patch RMSE are descriptive macro summaries, not the pooled pixel-weighted RMSE.",
                        "Pixels may share GEDI footprints or spatial dependence; n is not an independent sample count.",
                        "This evaluates agreement with GEDI L4A estimates, not independent field biomass measurements."]}
    if "val" in splits and args.checkpoint is None:
        expected = original_metrics["training_summary"]["best_val_RMSE"]
        difference = overall["val"]["rmse"] - expected
        report["validation_reproduction"] = {"recorded_best_val_RMSE": expected, "difference_mg_ha": difference,
                                             "within_0_05_mg_ha": abs(difference) <= .05}
        if abs(difference) > .05:
            print(f"Validation differs from the recorded checkpoint score by {difference:.4f} Mg/ha; inspect runtime/configuration differences.", flush=True)
    temporary_csv.replace(output / "predictions.csv")
    write_json(output / "evaluation_metrics.json", report)
    lines = ["# Fixed-checkpoint evaluation", "", f"Checkpoint: `{path}` (epoch index {checkpoint_epoch}).", "",
             f"Evaluated splits: {', '.join(splits)}. All labeled patches in each selected saved split were evaluated.", "",
             "| Split | Labeled patches | Pixels | RMSE | MAE | Bias | R2 | Baseline RMSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, scores in overall.items():
        r2 = f"{scores['r2']:.4f}" if scores["r2"] is not None else "n/a"
        lines.append(f"| {split} | {scores['labeled_patches']} | {scores['n']} | {scores['rmse']:.4f} | {scores['mae']:.4f} | {scores['bias']:.4f} | {r2} | {scores['baseline_rmse']:.4f} |")
    lines += ["", "RMSE/MAE/bias are in Mg/ha. Bias is prediction minus target.",
              "All results use the saved split, normalization, target scale and trained weights.",
              "Training metrics here use fixed weights in eval mode; they differ from online metrics in the training CSV.",
              "No predictions on unselected splits were produced. Region/period groups can overlap in their observations."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluation saved to {output}", flush=True)
    for split, scores in overall.items():
        print(f"{split}: n={scores['n']}, RMSE={scores['rmse']:.4f}, MAE={scores['mae']:.4f}, bias={scores['bias']:.4f} Mg/ha", flush=True)


if __name__ == "__main__":
    main()
