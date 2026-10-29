from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import lightning as L
import numpy as np
import rasterio
import torch
from rasterio.warp import transform_bounds
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class PatchPair:
    region: str
    name: str
    sentinel_path: Path
    gedi_path: Path
    input_valid_mask_path: Path
    metadata_path: Path
    bounds: tuple[float, float, float, float]
    valid_pixels: int

    @property
    def patch_id(self) -> str:
        return f"{self.region}/{self.name}"


def valid_gedi_mask(gedi: np.ndarray) -> np.ndarray:
    """AGBD is valid when finite and nonnegative; zero biomass is a label."""
    return np.isfinite(gedi) & (gedi >= 0)


def _input_mask(image: np.ndarray, path: Path) -> np.ndarray:
    mask = np.load(path, mmap_mode="r", allow_pickle=False)
    if mask.shape != image.shape[-2:] or mask.dtype != np.bool_:
        raise ValueError(f"{path}: expected a boolean input-valid mask of shape {image.shape[-2:]}.")
    # Filled zero bands are invalid only where the explicit mask says so.
    return np.asarray(mask) & np.all(np.isfinite(image), axis=0)


def _validated_bounds(metadata: dict, path: Path, split_crs: rasterio.crs.CRS) -> tuple:
    if (metadata.get("target") != "agbd" or metadata.get("target_units") != "Mg/ha"
            or metadata.get("label_product") != "GEDI_L4A"):
        raise ValueError(f"{path}: require target=agbd, target_units=Mg/ha, label_product=GEDI_L4A.")
    quality = metadata.get("quality_filter", {})
    flags = {"algorithm_run_flag": 1, "l4_quality_flag": 1, "degrade_flag": 0}
    if (not isinstance(quality, dict) or quality.get("applied") is not True
            or quality.get("required_flags") != flags
            or not isinstance(quality.get("method"), str) or not quality["method"].strip()
            or not isinstance(quality.get("provenance"), str) or not quality["provenance"].strip()):
        raise ValueError(f"{path}: missing verified GEDI L4A quality-filter provenance ({flags}).")
    try:
        crs_value = metadata["crs"]
        crs = rasterio.crs.CRS.from_user_input(crs_value["value"] if isinstance(crs_value, dict) else crs_value)
        bounds = tuple(float(metadata["bounds"][k]) for k in ("left", "bottom", "right", "top"))
        if not all(np.isfinite(bounds)) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("invalid footprint bounds")
        projected = tuple(transform_bounds(crs, split_crs, *bounds, densify_pts=21))
        if not all(np.isfinite(projected)) or projected[0] >= projected[2] or projected[1] >= projected[3]:
            raise ValueError("invalid projected footprint bounds")
        return projected
    except (KeyError, TypeError, ValueError, rasterio.errors.CRSError) as exc:
        raise ValueError(f"{path}: require valid full-patch CRS and bounds for geographic splitting.") from exc


def discover_patch_pairs(
    data_root: str | Path, regions: str | Sequence[str] | None = None,
    expected_channels: int = 14, split_crs: str = "EPSG:32648",
) -> list[PatchPair]:
    """Read processed province/site/period patches, including unlabeled patches."""
    root = Path(data_root).expanduser().resolve()
    metric_crs = rasterio.crs.CRS.from_user_input(split_crs)
    if not metric_crs.is_projected or not np.isclose(metric_crs.linear_units_factor[1], 1.0):
        raise ValueError("split_crs must be a projected CRS whose units are metres.")
    if regions is None:
        directories = sorted(path.parent for path in root.rglob("input") if path.is_dir())
    else:
        names = [regions] if isinstance(regions, str) else list(regions)
        directories = [root / name for name in names]
    pairs: list[PatchPair] = []
    seen_ids: set[str] = set()
    for directory in directories:
        region = directory.relative_to(root).as_posix()
        folders = {name: directory / name for name in ("input", "gedi", "input_valid_mask", "metadata")}
        for folder in folders.values():
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected processed data folder {folder}.")
        files = {name: {p.stem: p for p in folder.glob("*.json" if name == "metadata" else "*.npy")}
                 for name, folder in folders.items()}
        stems = set(files["input"])
        for name, paths in files.items():
            if set(paths) != stems:
                raise ValueError(f"Unpaired processed files in {directory}/{name}: "
                                 f"missing={sorted(stems - set(paths))}, extra={sorted(set(paths) - stems)}")
        for stem in sorted(stems):
            image = np.load(files["input"][stem], mmap_mode="r", allow_pickle=False)
            target = np.load(files["gedi"][stem], mmap_mode="r", allow_pickle=False)
            if image.ndim != 3 or image.shape[0] != expected_channels:
                raise ValueError(f"{files['input'][stem]} has shape {image.shape}; expected ({expected_channels}, H, W).")
            if target.ndim != 2 or image.shape[-2:] != target.shape:
                raise ValueError(f"Spatial mismatch for {region}/{stem}: input={image.shape}, AGBD={target.shape}.")
            valid = valid_gedi_mask(target) & _input_mask(image, files["input_valid_mask"][stem])
            metadata_path = files["metadata"][stem]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            bounds = _validated_bounds(metadata, metadata_path, metric_crs)
            pair = PatchPair(region, f"{stem}.npy", files["input"][stem], files["gedi"][stem],
                             files["input_valid_mask"][stem], metadata_path, bounds, int(valid.sum()))
            if pair.patch_id in seen_ids:
                raise ValueError(f"Duplicate patch selection: {pair.patch_id}.")
            seen_ids.add(pair.patch_id)
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"No processed .npy patches found below {root}.")
    return pairs


def geographic_split(
    pairs: Sequence[PatchPair], train_fraction: float, val_fraction: float, buffer_m: float
) -> tuple[dict[str, list[PatchPair]], dict]:
    """Fixed west/east strips based on all footprints, never labels or dates.

    Whole patches intersecting a boundary buffer are excluded from fitting.
    """
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Require train_fraction > 0, val_fraction > 0, and their sum < 1 (independent test).")
    if not np.isfinite(buffer_m) or buffer_m <= 0:
        raise ValueError("buffer_m must be positive metres.")
    west = min(pair.bounds[0] for pair in pairs)
    east = max(pair.bounds[2] for pair in pairs)
    boundary1 = west + (east - west) * train_fraction
    boundary2 = west + (east - west) * (train_fraction + val_fraction)
    half_buffer = buffer_m / 2
    split = {name: [] for name in ("train", "val", "test", "excluded")}
    for pair in pairs:
        left, _, right, _ = pair.bounds
        if right <= boundary1 - half_buffer:
            name = "train"
        elif left >= boundary1 + half_buffer and right <= boundary2 - half_buffer:
            name = "val"
        elif left >= boundary2 + half_buffer:
            name = "test"
        else:
            name = "excluded"
        split[name].append(pair)
    # Disjoint conservative projected boxes guarantee the buffer for full
    # footprints, including repeated locations in different acquisition years.
    separations = {}
    for left_name, right_name in (("train", "val"), ("val", "test"), ("train", "test")):
        if split[left_name] and split[right_name]:
            distance = min(p.bounds[0] for p in split[right_name]) - max(p.bounds[2] for p in split[left_name])
            if distance < buffer_m - 1e-6:
                raise ValueError(f"Geographic split violates {buffer_m} m footprint buffer.")
            separations[f"{left_name}_{right_name}"] = float(distance)
    return split, {"axis": "projected_x", "boundaries_x_m": [boundary1, boundary2],
                   "buffer_m": buffer_m, "footprint_x_separation_m": separations}


def compute_channel_stats(pairs: Iterable[PatchPair], expected_channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Training input statistics use the explicit input-valid mask."""
    channel_sum = np.zeros(expected_channels, dtype=np.float64)
    channel_squared_sum = np.zeros(expected_channels, dtype=np.float64)
    pixel_count = 0
    for pair in pairs:
        image = np.asarray(np.load(pair.sentinel_path, mmap_mode="r", allow_pickle=False), dtype=np.float64)
        values = image[:, _input_mask(image, pair.input_valid_mask_path)]
        channel_sum += values.sum(axis=1)
        channel_squared_sum += np.square(values).sum(axis=1)
        pixel_count += values.shape[1]
    if pixel_count == 0:
        raise ValueError("No valid training input pixels are available for normalization.")
    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - np.square(mean)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)


def compute_target_stats(pairs: Iterable[PatchPair], quantile: float) -> tuple[float, float, int]:
    if not 0 < quantile <= 1:
        raise ValueError("target_scale_quantile must be in (0, 1].")
    values = []
    for pair in pairs:
        image = np.load(pair.sentinel_path, mmap_mode="r", allow_pickle=False)
        target = np.load(pair.gedi_path, mmap_mode="r", allow_pickle=False)
        valid = valid_gedi_mask(target) & _input_mask(image, pair.input_valid_mask_path)
        values.append(np.asarray(target[valid], dtype=np.float64))
    labels = np.concatenate(values)
    if not labels.size:
        raise ValueError("No valid training AGBD labels are available for biomass scaling.")
    # The scale never clips or removes high biomass. Zero-heavy training data
    # still needs a positive denominator.
    scale = max(float(np.quantile(labels, quantile)), 1.0)
    return scale, float(labels.mean()), int(labels.size)


class GediSentinelDataset(Dataset):
    def __init__(self, pairs: Sequence[PatchPair], mean: np.ndarray, std: np.ndarray) -> None:
        self.pairs = list(pairs)
        self.mean = torch.from_numpy(mean).view(-1, 1, 1)
        self.std = torch.from_numpy(std).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        image_np = np.asarray(np.load(pair.sentinel_path, allow_pickle=False), dtype=np.float32)
        target_np = np.asarray(np.load(pair.gedi_path, allow_pickle=False), dtype=np.float32)
        input_valid = _input_mask(image_np, pair.input_valid_mask_path)
        valid = valid_gedi_mask(target_np) & input_valid
        image = torch.from_numpy(image_np.copy())
        image = torch.where(torch.isfinite(image), image, self.mean)
        image = (image - self.mean) / self.std
        image[:, ~torch.from_numpy(input_valid.copy())] = 0.0
        return {"image": image, "mask": torch.from_numpy(target_np.copy()),
                "valid_mask": torch.from_numpy(valid.copy()), "patch_id": pair.patch_id}


class GediSentinelDataModule(L.LightningDataModule):
    def __init__(
        self, data_root: str, regions: str | Sequence[str] | None = None,
        expected_channels: int = 14, batch_size: int = 1, num_workers: int = 0,
        train_fraction: float = 0.6, val_fraction: float = 0.2, buffer_m: float = 1000.0,
        split_crs: str = "EPSG:32648", target_scale_quantile: float = 0.95,
    ) -> None:
        super().__init__()
        self.data_root, self.regions = data_root, regions
        self.expected_channels, self.batch_size, self.num_workers = expected_channels, batch_size, num_workers
        self.train_fraction, self.val_fraction, self.buffer_m = train_fraction, val_fraction, buffer_m
        self.split_crs, self.target_scale_quantile = split_crs, target_scale_quantile
        self._is_ready = False

    def setup(self, stage: str | None = None) -> None:
        if self._is_ready:
            return
        all_pairs = discover_patch_pairs(self.data_root, self.regions, self.expected_channels, self.split_crs)
        geographic, audit = geographic_split(all_pairs, self.train_fraction, self.val_fraction, self.buffer_m)
        labeled = {name: [p for p in geographic[name] if p.valid_pixels > 0] for name in ("train", "val", "test")}
        for name, pairs in labeled.items():
            if not pairs:
                raise ValueError(f"Geographic {name} region has no valid AGBD labels after the {self.buffer_m:g} m buffer. "
                                 "Select larger regions or revise geographic fractions/buffer; no random split fallback is used.")
        # Include unlabeled inputs only from the training geographic region;
        # held-out inputs and labels never influence normalization/scaling.
        mean, std = compute_channel_stats(geographic["train"], self.expected_channels)
        self.target_scale, self.train_target_mean, train_labels = compute_target_stats(labeled["train"], self.target_scale_quantile)
        for name, pairs in labeled.items():
            setattr(self, f"{name}_dataset", GediSentinelDataset(pairs, mean, std))
        self.predict_dataset = GediSentinelDataset(all_pairs, mean, std)
        self.normalization_mean, self.normalization_std = mean, std
        self.split_summary = {
            **audit, "split_crs": self.split_crs, "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction, "target": "agbd", "target_units": "Mg/ha",
            "target_scale": self.target_scale, "target_scale_quantile": self.target_scale_quantile,
            "train_target_mean": self.train_target_mean, "train_label_pixels": train_labels,
            "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
            "counts": {name: {"all_patches": len(pairs), "labeled_patches": sum(p.valid_pixels > 0 for p in pairs),
                              "valid_label_pixels": sum(p.valid_pixels for p in pairs)} for name, pairs in geographic.items()},
            "patches": [{"patch_id": p.patch_id, "split": name, "bounds_m": list(p.bounds),
                         "valid_label_pixels": p.valid_pixels,
                         "fit_eligible": name != "excluded" and p.valid_pixels > 0}
                        for name, pairs in geographic.items() for p in pairs],
            "prediction_patch_ids": [p.patch_id for p in all_pairs],
        }
        self._is_ready = True
        print(f"AGBD patches: predict={len(all_pairs)}, train={len(labeled['train'])}, "
              f"val={len(labeled['val'])}, test={len(labeled['test'])}, "
              f"buffer-excluded={len(geographic['excluded'])}; train scale={self.target_scale:.3f} Mg/ha.")

    def _loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available(), persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_dataset)
