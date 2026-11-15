"""Add aligned int64 shot-ID patches to existing processed Sentinel/GEDI data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def prepare(raw_root: Path, processed_root: Path, overwrite: bool = False) -> list[dict]:
    reports = []
    for manifest_path in sorted(processed_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        main_name = Path(manifest["source_tif"]).name
        shot_name = main_name.replace("_S1_S2_GEDI.tif", "_GEDI_shot_number.tif")
        mains, shots = list(raw_root.rglob(main_name)), list(raw_root.rglob(shot_name))
        if len(mains) != 1 or len(shots) != 1:
            raise ValueError(f"Require one source stack and one shot raster for {manifest_path}.")
        directory = manifest_path.parent
        metadata_paths = sorted((directory / "metadata").glob("*.json"))
        if len(metadata_paths) != manifest["patch_count"]:
            raise ValueError(f"Incomplete processed data: {directory}.")
        with rasterio.open(mains[0]) as main, rasterio.open(shots[0]) as shot:
            if (main.shape != shot.shape or main.crs != shot.crs or main.transform != shot.transform
                    or np.dtype(shot.dtypes[0]).kind not in "iu" or np.dtype(shot.dtypes[0]).itemsize != 8):
                raise ValueError(f"Shot raster must be int64/uint64 and exactly aligned: {shots[0]}.")
            output = directory / "shot_number"
            output.mkdir(exist_ok=True)
            for metadata_path in metadata_paths:
                metadata = json.loads(metadata_path.read_text())
                w = metadata["source_window"]
                window = Window(w["col_off"], w["row_off"], w["width"], w["height"])
                if not np.allclose(list(shot.window_transform(window))[:6], metadata["transform"], rtol=0, atol=1e-8):
                    raise ValueError(f"Patch transform disagrees with source raster: {metadata_path}.")
                values = shot.read(1, window=window)
                if values.shape != (w["height"], w["width"]):
                    raise ValueError(f"Truncated shot window: {metadata_path}.")
                path = output / metadata_path.with_suffix(".npy").name
                if path.exists() and not overwrite:
                    if not np.array_equal(np.load(path, allow_pickle=False), values):
                        raise FileExistsError(f"Different shot data already exists: {path}.")
                else:
                    np.save(path, values, allow_pickle=False)
            report = {"directory": str(directory), "source": str(shots[0]), "patch_count": len(metadata_paths), "dtype": shot.dtypes[0]}
            reports.append(report)
            print(json.dumps(report), flush=True)
    if not reports:
        raise ValueError(f"No processed manifests below {processed_root}.")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.raw_root.resolve(), args.processed_root.resolve(), args.overwrite)
