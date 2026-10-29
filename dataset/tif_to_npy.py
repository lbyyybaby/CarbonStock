"""Export Sentinel/GEDI L4A AGBD TIFFs to NPY patches, retaining unlabeled patches."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "CarbonStock"
RAW_DIR = DATA_ROOT / "raw"
PATCH_SIZE = 252
INPUT_BANDS = 14
GEDI_NODATA = -9999.0  # Fallback even when the TIFF has no nodata metadata.
OUTPUT_FOLDERS = ("input", "gedi", "input_valid_mask", "metadata")
QUALITY_FLAGS = {"algorithm_run_flag": 1, "l4_quality_flag": 1, "degrade_flag": 0}


def agbd_metadata(provenance: str, method: str = "upstream_user_confirmed") -> dict:
    """Describe the label contract in both patch metadata and the manifest.

    Upstream confirmation is an explicit assertion that the GEDI L4A quality
    policy was applied before any temporal aggregation; it cannot be inferred
    from AGBD values or from a band name alone.
    """
    if not provenance or not provenance.strip():
        raise ValueError("Quality-filter provenance must be nonempty.")
    return {
        "target": "agbd", "target_units": "Mg/ha", "label_product": "GEDI_L4A",
        "quality_filter": {
            "applied": True, "method": method,
            "required_flags": dict(QUALITY_FLAGS), "provenance": provenance.strip(),
        },
    }


def label_band_contract(src, quality_filtered_provenance: str | None) -> tuple[int, dict, dict]:
    """Require explicit AGBD and either per-pixel QA or confirmed upstream QA."""
    names = [(name or "").lower().removeprefix("gedi_") for name in src.descriptions]
    if any("rh98" in name for name in names):
        raise ValueError("RH98 is canopy height, not AGBD. Export GEDI L4A agbd in Mg/ha.")
    if src.count < INPUT_BANDS + 1 or names[INPUT_BANDS] != "agbd":
        raise ValueError("Expected 14 Sentinel bands followed by GEDI_AGBD (Mg/ha).")
    extras = names[INPUT_BANDS + 1:]
    supported = {*QUALITY_FLAGS, "predictor_limit_flag", "response_limit_flag"}
    if len(extras) != len(set(extras)) or set(extras) - supported:
        raise ValueError(f"Unsupported/duplicate GEDI quality bands: {extras}.")
    flag_indices = {name: INPUT_BANDS + 1 + extras.index(name) for name in extras}
    if all(name in flag_indices for name in QUALITY_FLAGS):
        metadata = agbd_metadata(
            "Validated per-pixel quality bands in source TIFF: " + ", ".join(extras),
            method="raw_flags",
        )
        # Optional extrapolation flags are conservative when provided.
        metadata["quality_filter"]["optional_flags"] = {
            name: 0 for name in ("predictor_limit_flag", "response_limit_flag")
            if name in flag_indices
        }
    elif quality_filtered_provenance is not None:
        metadata = agbd_metadata(quality_filtered_provenance)
    else:
        raise ValueError(
            "GEDI L4A quality flags are missing. Include algorithm_run_flag, "
            "l4_quality_flag, degrade_flag, or explicitly confirm filtering before "
            "temporal aggregation with --quality-filtered-provenance."
        )
    return INPUT_BANDS, flag_indices, metadata


def parse_source_ids(tif_path: Path) -> tuple[str, str, str]:
    match = re.fullmatch(
        r"([^_]+)_([^_]+)_(\d{4}_\d{4})_S1_S2_GEDI", tif_path.stem
    )
    if match is None:
        raise ValueError(
            f"Invalid filename: {tif_path.name}. Expected "
            "<province>_<forest>_<YYYY_YYYY>_S1_S2_GEDI.tif"
        )
    return match.groups()


def write_json(path: Path, metadata: dict) -> None:
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_output(directory: Path, overwrite: bool) -> None:
    # Only inspect this province/site/window; other windows are untouched.
    files = [
        path
        for folder in OUTPUT_FOLDERS
        for path in (directory / folder).glob("*.json" if folder == "metadata" else "*.npy")
        if path.is_file()
    ]
    if (directory / "manifest.json").is_file():
        files.append(directory / "manifest.json")
    if files and not overwrite:
        raise FileExistsError(f"Output exists: {directory}. Use --overwrite to replace it.")
    for path in files:
        path.unlink()
    if files:
        print(f"Removed {len(files)} generated files from {directory}")
    for folder in OUTPUT_FOLDERS:
        (directory / folder).mkdir(parents=True, exist_ok=True)


def patch_metadata(src, window, common, patch_id, input_invalid, gedi_invalid):
    bounds = src.window_bounds(window)
    center_x, center_y = src.window_transform(window) * (window.width / 2, window.height / 2)
    lon, lat = transform(src.crs, "EPSG:4326", [center_x], [center_y])
    input_valid = ~input_invalid.any(axis=0)
    pixel_count = int(window.width * window.height)
    gedi_valid_count = int((~gedi_invalid).sum())
    return {
        **common,
        "patch_id": patch_id,
        "files": {folder: f"{folder}/{patch_id}.npy" for folder in OUTPUT_FOLDERS[:-1]},
        "shape": {"input": [INPUT_BANDS, *gedi_invalid.shape], "gedi": list(gedi_invalid.shape)},
        "source_window": {
            "row_off": int(window.row_off), "col_off": int(window.col_off),
            "height": int(window.height), "width": int(window.width),
        },
        # Affine coefficients a,b,c,d,e,f: x=a*col+b*row+c; y=d*col+e*row+f.
        "transform": list(src.window_transform(window))[:6],
        "bounds": dict(zip(("left", "bottom", "right", "top"), bounds)),
        "center": {"x": center_x, "y": center_y},
        "wgs84_bounds": dict(zip(
            ("left", "bottom", "right", "top"),
            transform_bounds(src.crs, "EPSG:4326", *bounds),
        )),
        "wgs84_center": {"longitude": lon[0], "latitude": lat[0]},
        "pixel_counts": {
            "pixels_per_band": pixel_count,
            "input_invalid_by_band": {
                name or f"band_{i}": int(count)
                for i, (name, count) in enumerate(zip(
                    common["bands"]["input"], input_invalid.sum(axis=(1, 2))
                ), start=1)
            },
            "input_all_bands_valid": int(input_valid.sum()),
            "input_any_band_invalid": int((~input_valid).sum()),
            "gedi_valid": gedi_valid_count,
            "gedi_invalid": pixel_count - gedi_valid_count,
        },
    }


def convert_tif(
    tif_path: Path,
    output_root: Path = DATA_ROOT / "processed",
    patch_size: int = PATCH_SIZE,
    overwrite: bool = False,
    quality_filtered_provenance: str | None = None,
) -> dict:
    tif_path = Path(tif_path).expanduser().resolve()
    province, site, period = parse_source_ids(tif_path)
    directory = Path(output_root).expanduser().resolve() / province / site / period
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    with rasterio.open(tif_path) as src:
        target_index, flag_indices, label_metadata = label_band_contract(
            src, quality_filtered_provenance
        )
        if src.crs is None or min(src.width, src.height) < patch_size:
            raise ValueError("The TIFF needs a CRS and must be at least one patch in size.")
        prepare_output(directory, overwrite)
        nodata = [str(v) if v is not None and not np.isfinite(v) else v for v in src.nodatavals]
        common = {
            **label_metadata,
            "province_id": province, "site_id": site, "window_id": period,
            "source_tif": str(tif_path),
            "crs": {"value": src.crs.to_string(), "epsg": src.crs.to_epsg()},
            "resolution": {
                "x": src.res[0], "y": src.res[1],
                "unit": src.crs.linear_units if src.crs.is_projected else "degree",
            },
            "bands": {
                "input": list(src.descriptions[:INPUT_BANDS]),
                "gedi": src.descriptions[target_index],
                "quality": {name: src.descriptions[index] for name, index in flag_indices.items()},
            },
            "nodata": {
                "source_input": nodata[:INPUT_BANDS], "source_gedi": nodata[target_index],
                "gedi_fallback_checked": GEDI_NODATA,
                "saved_input_fill": 0.0, "saved_gedi_fill": "NaN",
            },
        }
        counts = dict.fromkeys((
            "patch_count", "patches_without_gedi", "patches_with_input_nodata",
            "total_input_invalid_band_pixels", "total_gedi_valid_pixels",
        ), 0)

        for row in range(0, src.height - patch_size + 1, patch_size):
            for col in range(0, src.width - patch_size + 1, patch_size):
                patch_id = f"r{row:06d}_c{col:06d}"
                window = Window(col, row, patch_size, patch_size)
                data = src.read(window=window, masked=True, out_dtype="float32")
                invalid = np.ma.getmaskarray(data) | ~np.isfinite(data.data)
                x, y = data.data[:INPUT_BANDS], data.data[target_index]
                input_invalid = invalid[:INPUT_BANDS]
                gedi_invalid = invalid[target_index] | (y == GEDI_NODATA) | (y < 0)
                for name, index in flag_indices.items():
                    expected = QUALITY_FLAGS.get(name, 0)
                    gedi_invalid |= invalid[index] | (data.data[index] != expected)
                if input_invalid.any():
                    x[input_invalid] = 0.0
                if gedi_invalid.any():
                    y[gedi_invalid] = np.nan

                np.save(directory / "input" / f"{patch_id}.npy", x)
                np.save(directory / "gedi" / f"{patch_id}.npy", y)
                np.save(directory / "input_valid_mask" / f"{patch_id}.npy", ~input_invalid.any(axis=0))
                write_json(directory / "metadata" / f"{patch_id}.json", patch_metadata(
                    src, window, common, patch_id, input_invalid, gedi_invalid
                ))

                counts["patch_count"] += 1
                counts["patches_without_gedi"] += int(gedi_invalid.all())
                counts["patches_with_input_nodata"] += int(input_invalid.any())
                counts["total_input_invalid_band_pixels"] += int(input_invalid.sum())
                counts["total_gedi_valid_pixels"] += int((~gedi_invalid).sum())

        manifest = {
            **label_metadata,
            **counts,
            "province_id": province, "site_id": site, "window_id": period,
            "source_tif": str(tif_path),
            "source_shape": {"bands": src.count, "height": src.height, "width": src.width},
            "patch_size": patch_size,
            "omitted_edge_pixels": {"rows": src.height % patch_size, "columns": src.width % patch_size},
            "crs": common["crs"]["value"],
            "resolution": {"x": src.res[0], "y": src.res[1]},
            "source_nodata_by_band": nodata, "band_names": list(src.descriptions),
            "input_nodata_fill": 0.0, "gedi_nodata_fill": "NaN",
            "gedi_fallback_nodata": GEDI_NODATA,
        }
        write_json(directory / "manifest.json", manifest)
        print(f"{directory}: {counts}")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tif_paths", nargs="*", type=Path)
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT / "processed")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in the matching province/site/window.")
    parser.add_argument(
        "--quality-filtered-provenance",
        help="Explicit provenance confirming GEDI L4A quality filtering before temporal aggregation, for TIFFs without QA bands.",
    )
    args = parser.parse_args()
    tif_paths = args.tif_paths or sorted(RAW_DIR.glob("*_S1_S2_GEDI.tif"))
    if not tif_paths:
        parser.error(f"No matching TIFF files found in {RAW_DIR}.")
    for tif_path in tif_paths:
        convert_tif(
            tif_path, args.output_root, args.patch_size, args.overwrite,
            args.quality_filtered_provenance,
        )


if __name__ == "__main__":
    main()
