"""Analyze saved training history and audit the original geographic split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test", "excluded")
COLORS = dict(zip(SPLITS, ("#2563eb", "#d97706", "#16803c", "#888888")))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def load_run(run_dir):
    summary = json.loads((run_dir / "data_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if summary.get("target") != "agbd" or summary.get("target_units") != "Mg/ha":
        raise ValueError("Expected an AGBD run in Mg/ha.")
    return summary, metrics


def read_history(run_dir):
    epochs, learning_rates = {}, []
    with (run_dir / "history" / "metrics.csv").open(newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if value:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(f"Non-finite metric in history: {key}={value}.")
                    row[key] = number
            if "lr" in row and "step" in row:
                learning_rates.append({key: row[key] for key in ("epoch", "step", "lr") if key in row})
            if "epoch" in row:
                epoch = int(row["epoch"])
                epochs.setdefault(epoch, {"epoch": epoch}).update({k: v for k, v in row.items() if k != "epoch"})
    result = [epochs[index] for index in sorted(epochs)]
    if not any("val_RMSE" in row for row in result):
        raise ValueError("History has no finite validation RMSE.")
    return result, learning_rates


def replay_early_stopping(history, patience, min_delta):
    best, wait, best_epoch = math.inf, 0, None
    for row in history:
        if "val_RMSE" not in row:
            continue
        if row["val_RMSE"] < best - min_delta:
            best, wait, best_epoch = row["val_RMSE"], 0, int(row["epoch"])
        else:
            wait += 1
        if wait >= patience:
            return {"stop_epoch_index": int(row["epoch"]), "epochs_completed": int(row["epoch"]) + 1,
                    "best_significant_epoch_index": best_epoch, "best_significant_val_RMSE": best}
    return None


def label_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    result = {"n": int(values.size)}
    if not values.size:
        return result
    mean, std = float(values.mean()), float(values.std())
    result.update(mean=mean, std_population=std, min=float(values.min()), max=float(values.max()),
                  skewness=float(np.mean(((values - mean) / std) ** 3)) if std else None)
    for name, quantile in (("p01", .01), ("p05", .05), ("p10", .1), ("p25", .25),
                           ("median", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99)):
        result[name] = float(np.quantile(values, quantile))
    result["zero_count"] = int((values == 0).sum())
    for threshold in (30, 100, 200, 300, 400, 1000):
        result[f"above_{threshold}_count"] = int((values > threshold).sum())
    return result


def regression_metrics(target, prediction, baseline):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("Expected equally shaped 1D target and prediction arrays.")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite valid target or prediction.")
    if not target.size:
        return {"n": 0, "rmse": None, "mae": None, "bias": None, "r2": None,
                "squared_error_sum": 0.0, "baseline_rmse": None}
    error = prediction - target
    squared_error = float(np.square(error).sum())
    total_variation = float(np.square(target - target.mean()).sum())
    rmse = math.sqrt(squared_error / target.size)
    baseline_rmse = float(np.sqrt(np.mean(np.square(target - baseline))))
    return {"n": int(target.size), "rmse": rmse, "mae": float(np.abs(error).mean()),
            "bias": float(error.mean()), "r2": 1 - squared_error / total_variation if total_variation > 0 else None,
            "squared_error_sum": squared_error, "baseline_rmse": baseline_rmse,
            "baseline_mae": float(np.abs(target - baseline).mean()),
            "rmse_improvement_pct": 100 * (1 - rmse / baseline_rmse) if baseline_rmse else None,
            "target_mean": float(target.mean()), "prediction_mean": float(prediction.mean()),
            "target_std": float(target.std()), "prediction_std": float(prediction.std()),
            "below_zero_pct": float((prediction < 0).mean() * 100)}


@dataclass
class SavedPatch:
    saved_id: str
    resolved_id: str
    split: str
    directory: Path
    name: str
    bounds: tuple
    metadata: dict
    labels: np.ndarray

    @property
    def valid_pixels(self):
        return int(self.labels.size)


def resolve_data_root(summary, override=None):
    path = Path(override or summary["config"]["dataset"]["data_root"]).expanduser()
    if not path.is_absolute():
        path = REPOSITORY / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Data root not found: {path}. Set --data-root to the local processed folder.")
    return path


def summarize_periods(patches):
    """Include empty splits so missing labels in a period remain visible."""
    groups = {}
    for patch in patches:
        key = tuple(str(patch.metadata.get(field, "")) for field in
                    ("country_id", "province_id", "site_id", "window_id"))
        groups.setdefault(key, []).append(patch)
    rows, warnings = [], []
    for key, group in sorted(groups.items()):
        for split in SPLITS:
            subset = [p for p in group if p.split == split]
            labels = np.concatenate([p.labels for p in subset]) if subset else np.empty(0)
            rows.append({**dict(zip(("country", "province", "site", "period"), key)),
                         "split": split, "all_patches": len(subset),
                         "labeled_patches": sum(p.valid_pixels > 0 for p in subset),
                         "unique_spatial_patches": len({p.bounds for p in subset}),
                         **label_statistics(labels)})
            if split != "excluded" and not labels.size:
                warnings.append(f"{'/'.join(key)}: no valid {split} labels.")
    return rows, warnings


def audit_dataset(summary, data_root, country_prefix=None):
    """Resolve original IDs; verify geometry, label counts and saved train statistics.

    No new split is computed. An optional single country level may precede old IDs.
    Counts/statistics are checks, not a byte-level fingerprint of the old dataset.
    """
    from rasterio.warp import transform_bounds
    from dataset.sentinel_gedi_canopy import valid_gedi_mask

    saved = summary["split"]["patches"]
    if not saved or len({row["patch_id"] for row in saved}) != len(saved):
        raise ValueError("Saved split has empty or duplicate patch IDs.")
    channels = int(summary["config"]["dataset"]["expected_channels"])
    max_agbd = summary["config"]["dataset"].get("max_agbd")
    channel_sum, channel_square_sum = np.zeros(channels), np.zeros(channels)
    input_count, patches, resolved = 0, [], set()
    for index, row in enumerate(saved):
        relative = PurePosixPath(row["patch_id"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npy":
            raise ValueError(f"Invalid saved patch ID: {relative}")
        if row["split"] not in SPLITS:
            raise ValueError(f"Unknown saved split: {row['split']}")
        suffix = relative.parent / "metadata" / relative.with_suffix(".json").name
        candidates = [data_root / suffix]
        if country_prefix:
            prefix = PurePosixPath(country_prefix)
            if prefix.is_absolute() or ".." in prefix.parts:
                raise ValueError("--country-prefix must be relative to --data-root.")
            candidates.append(data_root / prefix / suffix)
        else:
            candidates.extend(data_root.glob("*/" + suffix.as_posix()))
        candidates = sorted({path.resolve() for path in candidates if path.is_file()})
        if len(candidates) != 1:
            raise ValueError(f"Expected one match for {relative}, found {len(candidates)}. "
                             "Use --data-root/--country-prefix to select the original data.")
        metadata_path = candidates[0]
        metadata_path.relative_to(data_root)
        if metadata_path in resolved:
            raise ValueError(f"Two saved patches resolve to {metadata_path}.")
        resolved.add(metadata_path)
        directory = metadata_path.parent.parent
        name = relative.name
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("target"), metadata.get("target_units"), metadata.get("label_product")) != ("agbd", "Mg/ha", "GEDI_L4A"):
            raise ValueError(f"Incorrect label contract: {metadata_path}")
        crs = metadata["crs"]
        crs = crs["value"] if isinstance(crs, dict) else crs
        bounds = tuple(metadata["bounds"][key] for key in ("left", "bottom", "right", "top"))
        projected = transform_bounds(crs, summary["split"]["split_crs"], *bounds, densify_pts=21)
        if not np.allclose(projected, row["bounds_m"], atol=1e-3, rtol=0):
            raise ValueError(f"Footprint changed for {relative}.")
        image = np.load(directory / "input" / name, mmap_mode="r", allow_pickle=False)
        target = np.load(directory / "gedi" / name, mmap_mode="r", allow_pickle=False)
        mask = np.load(directory / "input_valid_mask" / name, mmap_mode="r", allow_pickle=False)
        if image.ndim != 3 or image.shape[0] != channels or image.shape[1:] != target.shape or target.ndim != 2:
            raise ValueError(f"Invalid array shapes for {relative}.")
        if mask.shape != target.shape or mask.dtype != np.bool_:
            raise ValueError(f"Invalid input mask for {relative}.")
        valid_input = np.asarray(mask) & np.isfinite(image).all(axis=0)
        labels = np.asarray(target[valid_gedi_mask(target, max_agbd) & valid_input], dtype=np.float64)
        if labels.size != row["valid_label_pixels"]:
            raise ValueError(f"Label count changed for {relative}: {labels.size} != {row['valid_label_pixels']}.")
        if bool(row["fit_eligible"]) != (row["split"] != "excluded" and labels.size > 0):
            raise ValueError(f"Saved eligibility disagrees with split/count for {relative}.")
        if row["split"] == "train":
            values = np.asarray(image, dtype=np.float64)[:, valid_input]
            channel_sum += values.sum(axis=1)
            channel_square_sum += np.square(values).sum(axis=1)
            input_count += values.shape[1]
        patches.append(SavedPatch(str(relative), (directory.relative_to(data_root) / name).as_posix(),
                                  row["split"], directory, name, tuple(row["bounds_m"]), metadata, labels))
        if (index + 1) % 100 == 0:
            print(f"Audited {index + 1}/{len(saved)} saved patches", flush=True)
    if not input_count:
        raise ValueError("No valid training inputs.")
    mean = channel_sum / input_count
    std = np.sqrt(np.maximum(channel_square_sum / input_count - mean ** 2, 1e-12))
    for name, current in (("normalization_mean", mean), ("normalization_std", std)):
        if not np.allclose(current, summary[name], atol=1e-6, rtol=1e-5):
            raise ValueError(f"Training {name} no longer matches the saved run.")
    counts, statistics = {}, {}
    for split in SPLITS:
        subset = [patch for patch in patches if patch.split == split]
        labels = np.concatenate([patch.labels for patch in subset]) if subset else np.empty(0)
        counts[split] = {"all_patches": len(subset), "labeled_patches": sum(p.valid_pixels > 0 for p in subset),
                         "valid_label_pixels": int(labels.size)}
        if counts[split] != summary["split"]["counts"][split]:
            raise ValueError(f"Saved/current split counts differ for {split}.")
        statistics[split] = label_statistics(labels)
        statistics[split]["largest_patch_label_share_pct"] = max((p.valid_pixels for p in subset), default=0) / labels.size * 100 if labels.size else None
    training_labels = np.concatenate([p.labels for p in patches if p.split == "train"])
    quantile = summary["config"]["dataset"].get("target_scale_quantile", .95)
    scale = max(float(np.quantile(training_labels, quantile)), 1.)
    if not np.isclose(scale, summary["target_scale"], rtol=1e-6, atol=1e-6) or not np.isclose(training_labels.mean(), summary["train_target_mean"], rtol=1e-6, atol=1e-6):
        raise ValueError("Training label statistics no longer match target_scale/train_target_mean.")
    period_statistics, period_warnings = summarize_periods(patches)
    return patches, {"data_root": str(data_root), "mapped_patches": len(patches), "max_agbd": max_agbd,
                     "renamed_patch_ids": sum(p.saved_id != p.resolved_id for p in patches),
                     "counts": counts, "label_statistics": statistics,
                     "period_statistics": period_statistics, "period_warnings": period_warnings,
                     "train_normalization_verified": True, "train_target_statistics_verified": True,
                     "verification_limit": "Geometry, masks/counts and aggregate training statistics checked; no original file hashes were saved."}


def plot_history(output, history, rates, best_epoch):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, label, color in (("train_RMSE", "Train (online)", COLORS["train"]), ("val_RMSE", "Validation", COLORS["val"]),
                              ("val_mean_baseline_RMSE", "Mean baseline", "#777777")):
        rows = [row for row in history if key in row]
        if rows:
            axes[0, 0].plot([r["epoch"] + 1 for r in rows], [r[key] for r in rows], label=label, color=color)
    axes[0, 0].axvline(best_epoch + 1, color="black", linestyle=":", label="Best validation")
    axes[0, 0].set(xlabel="Epoch (1-based)", ylabel="RMSE (Mg/ha)")
    axes[0, 0].legend()
    for key, label in (("train_Loss(MSE)_epoch", "Train"), ("val_Loss(MSE)", "Validation")):
        rows = [row for row in history if key in row]
        if rows:
            axes[0, 1].plot([r["epoch"] + 1 for r in rows], [r[key] for r in rows], label=label)
    axes[0, 1].set(xlabel="Epoch (1-based)", ylabel="Normalized MSE")
    axes[0, 1].legend()
    positive = [r for r in rates if r["lr"] > 0]
    axes[1, 0].plot([r["step"] for r in positive], [r["lr"] for r in positive], color="#7c3aed")
    axes[1, 0].set(xlabel="Logged optimizer step", ylabel="Learning rate", yscale="log")
    paired = [r for r in history if "train_RMSE" in r and "val_RMSE" in r]
    axes[1, 1].plot([r["epoch"] + 1 for r in paired], [r["val_RMSE"] - r["train_RMSE"] for r in paired])
    axes[1, 1].set(xlabel="Epoch (1-based)", ylabel="Val - online train RMSE (Mg/ha)")
    for axis in axes.flat:
        axis.grid(alpha=.2)
    figure.savefig(output / "learning_curves.png", dpi=160)
    plt.close(figure)


def plot_split_map(output, patches, summary):
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for split in SPLITS:
        subset = [p for p in patches if p.split == split]
        axis.scatter([(p.bounds[0] + p.bounds[2]) / 2 for p in subset], [(p.bounds[1] + p.bounds[3]) / 2 for p in subset],
                     s=[12 if not p.valid_pixels else 35 for p in subset], color=COLORS[split], alpha=.65, label=split)
    for boundary in summary["split"].get("boundaries_x_m", []):
        half = summary["split"]["buffer_m"] / 2
        axis.axvspan(boundary - half, boundary + half, color="black", alpha=.12)
    axis.set(xlabel="Projected easting (m)", ylabel="Projected northing (m)",
             title=f"Saved split: {summary['split']['split_crs']}; larger markers have GEDI labels")
    axis.set_aspect("equal")
    axis.legend()
    figure.savefig(output / "split_map.png", dpi=160)
    plt.close(figure)


def label_distribution_markdown(statistics):
    lines = ["", "## AGBD label distribution", "",
             "Values are in Mg/ha, except skewness (unitless). Statistics use all valid labels without clipping.", "",
             "| Statistic | Train | Validation | Test | Excluded |", "|---|---:|---:|---:|---:|"]
    for label, key in (("Mean", "mean"), ("Std (population)", "std_population"), ("Skewness", "skewness"),
                       ("Min", "min"), ("P01", "p01"), ("P05", "p05"), ("P10", "p10"),
                       ("P25", "p25"), ("Median", "median"), ("P75", "p75"), ("P90", "p90"),
                       ("P95", "p95"), ("P99", "p99"), ("Max", "max")):
        values = [statistics[s].get(key) for s in SPLITS]
        lines.append("| " + label + " | " + " | ".join(f"{v:.2f}" if v is not None else "n/a" for v in values) + " |")
    lines += ["", "### Label counts by threshold", "",
              "Each cell gives the pixel count and its percentage within that partition. Thresholds use strict >.", "",
              "| Condition | Train | Validation | Test | Excluded |", "|---|---:|---:|---:|---:|"]
    conditions = [("AGBD = 0", "zero_count")] + [(f"AGBD > {t}", f"above_{t}_count") for t in (30, 100, 200, 300, 400, 1000)]
    for label, key in conditions:
        values = []
        for split in SPLITS:
            n = statistics[split]["n"]
            count = statistics[split].get(key, 0)
            values.append(f"{count:,} ({100 * count / n:.2f}%)" if n else "n/a")
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines += ["", "Excluded patches are outside fitting and evaluation. Pixel counts do not imply independent GEDI observations."]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/agbd"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--country-prefix")
    parser.add_argument("--history-only", action="store_true", help="Analyze logs without reading processed arrays.")
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    summary, metrics = load_run(run)
    history, rates = read_history(run)
    validation = [r for r in history if "val_RMSE" in r]
    best, last = min(validation, key=lambda r: r["val_RMSE"]), validation[-1]
    recorded = metrics.get("training_summary", {}).get("best_val_RMSE")
    if recorded is not None and not np.isclose(recorded, best["val_RMSE"], atol=1e-4, rtol=0):
        raise ValueError("metrics.json and CSV disagree about best validation RMSE.")
    early = summary["config"].get("early_stopping", {})
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run),
              "target_units": "Mg/ha", "config_used_for_training": summary["config"],
              "history": {"validation_epochs": len(validation), "logged_lr_points": len(rates),
                          "best_epoch_index": best["epoch"], "best_epoch_number": best["epoch"] + 1,
                          "best_val_RMSE": best["val_RMSE"], "train_RMSE_online_at_best_epoch": best.get("train_RMSE"),
                          "last_val_RMSE": last["val_RMSE"], "last_train_RMSE_online": last.get("train_RMSE"),
                          "baseline_val_RMSE": best.get("val_mean_baseline_RMSE"),
                          "last_vs_best_increase_pct": 100 * (last["val_RMSE"] / best["val_RMSE"] - 1) if best["val_RMSE"] else None},
              "early_stopping_replay": {"patience": early.get("patience", 5), "min_delta": early.get("min_delta", .1),
                  "result": replay_early_stopping(history, early.get("patience", 5), early.get("min_delta", .1)),
                  "note": "Counterfactual replay of logged validation checks, not a rerun."},
              "notes": ["Training RMSE is collected during optimizer updates; evaluate one checkpoint in eval mode for a fixed-weight comparison.",
                        "Learning rates are sampled log entries, not a complete scheduler trace.",
                        "Patch/pixel counts do not establish independent GEDI footprint counts."]}
    patches = None
    if not args.history_only:
        patches, report["data_audit"] = audit_dataset(summary, resolve_data_root(summary, args.data_root), args.country_prefix)
    output = (args.output_dir or run / "analysis").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = ["epoch"] + sorted({key for row in history for key in row if key != "epoch"})
    write_csv(output / "history_by_epoch.csv", history, fields)
    plot_history(output, history, rates, best["epoch"])
    if patches is not None:
        write_csv(output / "label_statistics.csv", [{"split": s, **report["data_audit"]["label_statistics"][s]} for s in SPLITS],
                  ["split"] + sorted({key for stats in report["data_audit"]["label_statistics"].values() for key in stats}))
        write_csv(output / "label_statistics_by_period.csv", report["data_audit"]["period_statistics"])
        rows = [{"saved_patch_id": p.saved_id, "resolved_patch_id": p.resolved_id, "split": p.split,
                 "valid_label_pixels": p.valid_pixels, "fit_eligible": p.split != "excluded" and p.valid_pixels > 0,
                 "country": p.metadata.get("country_id", ""), "province": p.metadata.get("province_id", ""),
                 "site": p.metadata.get("site_id", ""), "period": p.metadata.get("window_id", ""),
                 "label_max": float(p.labels.max()) if p.valid_pixels else None} for p in patches]
        write_csv(output / "patch_audit.csv", rows)
        plot_split_map(output, patches, summary)
    write_json(output / "analysis_report.json", report)
    lines = ["# Saved run analysis", "", f"Run: `{run}`", "",
             f"Best validation RMSE: **{best['val_RMSE']:.4f} Mg/ha**, epoch index {best['epoch']} (epoch {best['epoch'] + 1}).",
             f"Last validation RMSE: **{last['val_RMSE']:.4f} Mg/ha**.", "",
             "Training metrics in the history are online metrics, not a fixed-checkpoint evaluation.", "",
             "## Early stopping replay", "", f"`{report['early_stopping_replay']['result']}`", "",
             "This replay uses the old validation trace; it does not predict a new run exactly."]
    if patches is not None:
        lines += ["", "## Original split (verified against current arrays)", "",
                  "| Split | All patches | Labeled patches | Valid pixels | Largest patch label share |", "|---|---:|---:|---:|---:|"]
        for split in SPLITS:
            counts = report["data_audit"]["counts"][split]
            share = report["data_audit"]["label_statistics"][split]["largest_patch_label_share_pct"]
            share_text = f"{share:.1f}%" if share is not None else "n/a"
            lines.append(f"| {split} | {counts['all_patches']} | {counts['labeled_patches']} | {counts['valid_label_pixels']} | {share_text} |")
        lines += ["", "The original split and train statistics were preserved. No new geographic split was calculated.",
                  "Counts and aggregate checks cannot prove byte-identical inputs without original file hashes."]
        lines += label_distribution_markdown(report["data_audit"]["label_statistics"])
        ceiling = report["data_audit"]["max_agbd"]
        lines += ["", "## Labels by period", "",
                  f"Saved AGBD upper label limit: {ceiling if ceiling is not None else 'none'} Mg/ha.", "",
                  "| Region | Period | Split | Patches | Spatial footprints | Labeled patches | Valid pixels | Mean | P95 |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in report["data_audit"]["period_statistics"]:
            region = "/".join(row[k] for k in ("country", "province", "site"))
            mean = f"{row['mean']:.2f}" if row["n"] else "n/a"
            p95 = f"{row['p95']:.2f}" if row["n"] else "n/a"
            lines.append(f"| {region} | {row['period']} | {row['split']} | {row['all_patches']} | "
                         f"{row['unique_spatial_patches']} | {row['labeled_patches']} | {row['n']} | {mean} | {p95} |")
        lines += ["", "Spatial footprints count distinct saved bounding boxes, not independent GEDI shots."]
        if report["data_audit"]["period_warnings"]:
            lines += ["", "### Missing period labels", ""]
            lines += [f"- {warning}" for warning in report["data_audit"]["period_warnings"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    obsolete_plot = output / "label_distributions.png"
    if obsolete_plot.is_file():
        obsolete_plot.unlink()
    print(f"Analysis saved to {output}", flush=True)


if __name__ == "__main__":
    main()
