from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


DATA_ROOT = Path.home() / "data" / "CarbonStock"

TIF_PATH = DATA_ROOT / ".raw" / "NgocHien.tif"
INPUT_DIR = DATA_ROOT / "Sentinel" / "NgocHien"
GEDI_DIR = DATA_ROOT / "GEDI" / "NgocHien"

PATCH_SIZE = 252
INPUT_BANDS = list(range(1, 15))
GEDI_BAND = 15


INPUT_DIR.mkdir(parents=True, exist_ok=True)
GEDI_DIR.mkdir(parents=True, exist_ok=True)


def remove_existing_npy(directory: Path) -> int:
    npy_files = list(directory.glob("*.npy"))
    for npy_file in npy_files:
        npy_file.unlink()
    return len(npy_files)


with rasterio.open(TIF_PATH) as src:
    assert src.count == 15, f"Expected 15 bands, found {src.count}"
    assert src.descriptions[14] == "GEDI_rh98"

    removed_count = remove_existing_npy(INPUT_DIR) + remove_existing_npy(GEDI_DIR)
    if removed_count:
        print(f"Removed existing .npy files: {removed_count}")

    candidate_count = 0
    no_gedi_count = 0
    patch_id = 0

    for row in range(0, src.height - PATCH_SIZE + 1, PATCH_SIZE):
        for col in range(0, src.width - PATCH_SIZE + 1, PATCH_SIZE):
            candidate_count += 1

            window = Window(
                col_off=col,
                row_off=row,
                width=PATCH_SIZE,
                height=PATCH_SIZE,
            )

            # Sentinel-2 + Sentinel-1: (14, 252, 252)
            x_masked = src.read(
                INPUT_BANDS,
                window=window,
                masked=True,
            )

            # GEDI: (252, 252)
            y_masked = src.read(
                GEDI_BAND,
                window=window,
                masked=True,
            )

            y_data = np.asarray(y_masked.data, dtype=np.float32)
            y_invalid = np.ma.getmaskarray(y_masked)

            valid_gedi = (
                (~y_invalid)
                & np.isfinite(y_data)
                & (y_data > 0)
            )

            if not valid_gedi.any():
                no_gedi_count += 1

            # Input NoData to 0; GEDI NoData to NaN
            x = np.asarray(x_masked.filled(0), dtype=np.float32)
            x = np.nan_to_num(
                x,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            y = np.full(
                (PATCH_SIZE, PATCH_SIZE),
                np.nan,
                dtype=np.float32,
            )
            y[valid_gedi] = y_data[valid_gedi]

            np.save(INPUT_DIR / f"{patch_id}.npy", x)
            np.save(GEDI_DIR / f"{patch_id}.npy", y)

            patch_id += 1

    print(f"Candidate patches: {candidate_count}")
    print(f"Patches without GEDI: {no_gedi_count}")
    print(f"Saved paired patches: {patch_id}")
