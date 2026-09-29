from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class PatchPair:
    region: str
    name: str
    sentinel_path: Path
    gedi_path: Path
    valid_pixels: int


def valid_gedi_mask(
    gedi: np.ndarray, min_height: float = 0.0, max_height: float = 30.0
) -> np.ndarray:
    """Return pixels that represent usable GEDI canopy-height observations."""
    return np.isfinite(gedi) & (gedi > min_height) & (gedi <= max_height)


def _sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)

def discover_patch_pairs(
    data_root: str | Path,
    regions: str | Sequence[str],
    expected_channels: int,
    min_height: float,
    max_height: float,
) -> list[PatchPair]:
    """Discover and validate Sentinel/GEDI pairs by region and file name."""
    root = Path(data_root).expanduser().resolve()
    region_names = regions.split("-") if isinstance(regions, str) else list(regions)
    pairs: list[PatchPair] = []

    for region in region_names:
        sentinel_dir = root / "Sentinel" / region
        gedi_dir = root / "GEDI" / region
        if not sentinel_dir.is_dir() or not gedi_dir.is_dir():
            raise FileNotFoundError(
                f"Expected data folders {sentinel_dir} and {gedi_dir}."
            )

        sentinel_files = {p.name: p for p in sentinel_dir.glob("*.npy")}
        gedi_files = {p.name: p for p in gedi_dir.glob("*.npy")}
        missing_sentinel = sorted(set(gedi_files) - set(sentinel_files))
        missing_gedi = sorted(set(sentinel_files) - set(gedi_files))
        if missing_sentinel or missing_gedi:
            raise ValueError(
                f"Unpaired files in region {region}: "
                f"missing Sentinel={missing_sentinel}, missing GEDI={missing_gedi}"
            )

        for name in sorted(sentinel_files, key=lambda item: _sort_key(Path(item))):
            sentinel_path = sentinel_files[name]
            gedi_path = gedi_files[name]
            sentinel = np.load(sentinel_path, mmap_mode="r")
            gedi = np.load(gedi_path, mmap_mode="r")
            if sentinel.ndim != 3 or sentinel.shape[0] != expected_channels:
                raise ValueError(
                    f"{sentinel_path} has shape {sentinel.shape}; expected "
                    f"({expected_channels}, H, W)."
                )
            if gedi.ndim != 2 or tuple(sentinel.shape[-2:]) != tuple(gedi.shape):
                raise ValueError(
                    f"Spatial mismatch: Sentinel {sentinel.shape}, GEDI {gedi.shape} "
                    f"for {region}/{name}."
                )
            valid_pixels = int(
                valid_gedi_mask(gedi, min_height, max_height).sum()
            )
            pairs.append(
                PatchPair(
                    region=region,
                    name=name,
                    sentinel_path=sentinel_path,
                    gedi_path=gedi_path,
                    valid_pixels=valid_pixels,
                )
            )

    if not pairs:
        raise ValueError(f"No paired .npy patches found below {root}.")
    return pairs


def compute_channel_stats(
    pairs: Iterable[PatchPair], expected_channels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel statistics, excluding all-zero/no-data pixels."""
    channel_sum = np.zeros(expected_channels, dtype=np.float64)
    channel_squared_sum = np.zeros(expected_channels, dtype=np.float64)
    pixel_count = 0

    for pair in pairs:
        image = np.asarray(np.load(pair.sentinel_path, mmap_mode="r"), dtype=np.float64)
        usable = np.all(np.isfinite(image), axis=0) & np.any(image != 0, axis=0)
        values = image[:, usable]
        channel_sum += values.sum(axis=1)
        channel_squared_sum += np.square(values).sum(axis=1)
        pixel_count += values.shape[1]

    if pixel_count == 0:
        raise ValueError("No finite, non-zero Sentinel pixels are available for normalization.")

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class GediSentinelDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PatchPair],
        mean: np.ndarray,
        std: np.ndarray,
        min_height: float = 0.0,
        max_height: float = 30.0,
    ) -> None:
        self.pairs = list(pairs)
        self.mean = torch.from_numpy(mean).view(-1, 1, 1)
        self.std = torch.from_numpy(std).view(-1, 1, 1)
        self.min_height = min_height
        self.max_height = max_height

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        image_np = np.asarray(np.load(pair.sentinel_path), dtype=np.float32)
        target_np = np.asarray(np.load(pair.gedi_path), dtype=np.float32)
        valid_np = valid_gedi_mask(target_np, self.min_height, self.max_height)

        image = torch.from_numpy(image_np.copy())
        target = torch.from_numpy(target_np.copy())
        valid_mask = torch.from_numpy(valid_np.copy())

        no_data = (~torch.isfinite(image)).any(dim=0) | (image != 0).logical_not().all(dim=0)
        image = torch.where(torch.isfinite(image), image, self.mean)
        image = (image - self.mean) / self.std
        image[:, no_data] = 0.0

        return {
            "image": image,
            "mask": target,
            "valid_mask": valid_mask,
            "patch_id": f"{pair.region}/{pair.name}",
        }


class GediSentinelDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        regions: str | Sequence[str] = "NgocHien",
        expected_channels: int = 14,
        batch_size: int = 1,
        num_workers: int = 0,
        train_fraction: float = 0.8,
        split_seed: int = 42,
        min_height: float = 0.0,
        max_height: float = 30.0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.regions = regions
        self.expected_channels = expected_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.split_seed = split_seed
        self.min_height = min_height
        self.max_height = max_height
        self._is_ready = False

    def setup(self, stage: str | None = None) -> None:
        if self._is_ready:
            return

        all_pairs = discover_patch_pairs(
            self.data_root,
            self.regions,
            self.expected_channels,
            self.min_height,
            self.max_height,
        )
        labeled_pairs = [pair for pair in all_pairs if pair.valid_pixels > 0]
        if len(labeled_pairs) < 2:
            raise ValueError(
                "At least two patches containing valid GEDI pixels are required "
                "for a non-overlapping train/validation split."
            )

        order = np.random.default_rng(self.split_seed).permutation(len(labeled_pairs))
        split = int(self.train_fraction * len(labeled_pairs))
        split = min(max(split, 1), len(labeled_pairs) - 1)
        train_pairs = [labeled_pairs[i] for i in order[:split]]
        val_pairs = [labeled_pairs[i] for i in order[split:]]

        mean, std = compute_channel_stats(train_pairs, self.expected_channels)
        common = dict(
            mean=mean,
            std=std,
            min_height=self.min_height,
            max_height=self.max_height,
        )
        self.train_dataset = GediSentinelDataset(train_pairs, **common)
        self.val_dataset = GediSentinelDataset(val_pairs, **common)
        self.test_dataset = self.val_dataset
        self.predict_dataset = GediSentinelDataset(all_pairs, **common)
        self.normalization_mean = mean
        self.normalization_std = std
        self._is_ready = True

        print(
            f"Data: {len(all_pairs)} paired patches, {len(labeled_pairs)} labeled; "
            f"train={len(train_pairs)} ({sum(p.valid_pixels for p in train_pairs)} GEDI pixels), "
            f"val={len(val_pairs)} ({sum(p.valid_pixels for p in val_pairs)} GEDI pixels)."
        )

    def _loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_dataset)
