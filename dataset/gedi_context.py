"""Deterministic, nested GEDI context subsets selected by exact shot ID."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np


@dataclass(frozen=True)
class GediContext:
    enabled: bool = False
    context_fraction: float = 0.2
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("gedi_context.enabled must be boolean.")
        if not (math.isfinite(self.context_fraction) and 0 <= self.context_fraction < 1):
            raise ValueError("context_fraction must be in [0, 1); some shots must remain as labels.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**64:
            raise ValueError("gedi_context.seed must be an unsigned 64-bit integer.")


def role_masks(directory: Path, name: str, valid: np.ndarray,
               config: GediContext) -> tuple[np.ndarray, np.ndarray]:
    if not config.enabled:
        return np.zeros_like(valid), valid.copy()
    path = directory / "shot_number" / name
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run dataset/prepare_shots.py before enabling GEDI context.")
    shots = np.load(path, mmap_mode="r", allow_pickle=False)
    return masks_from_shots(shots, valid, config)


def masks_from_shots(shots: np.ndarray, valid: np.ndarray,
                     config: GediContext) -> tuple[np.ndarray, np.ndarray]:
    if shots.shape != valid.shape or shots.dtype.kind not in "iu" or shots.dtype.itemsize != 8:
        raise ValueError("Shot IDs must be an aligned int64/uint64 array; floating point IDs lose precision.")
    # SplitMix64: never cast the original ID to float or use Python's salted hash.
    # The same ID retains its role across pixels, patches, dates and workers.
    with np.errstate(over="ignore"):
        hashed = shots.astype(np.uint64) + np.uint64(config.seed) + np.uint64(0x9E3779B97F4A7C15)
        hashed = (hashed ^ (hashed >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        hashed = (hashed ^ (hashed >> 27)) * np.uint64(0x94D049BB133111EB)
        hashed ^= hashed >> 31
    uniform = (hashed >> 11).astype(np.float64) * (1.0 / 2**53)
    eligible = valid & (shots > 0)
    context = eligible & (uniform < config.context_fraction)
    query = eligible & ~context
    return context, query
