"""Conversion regression checks for province-scoped Hugging Face downloads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin


spec = importlib.util.spec_from_file_location(
    "tif_to_npy", Path(__file__).resolve().parents[1] / "dataset" / "tif_to_npy.py"
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
BANDS = ["S2_" + band for band in
         ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")]
BANDS += ["S1_VV", "S1_VH", "GEDI_AGBD"]


class DataPreparationTest(unittest.TestCase):
    def make_tif(self, path: Path, value: float = 42.0) -> np.ndarray:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.arange(15 * 8 * 8, dtype=np.float32).reshape(15, 8, 8)
        data[14] = value
        data[0, 0, 0] = -9999
        data[14, 1, 1] = np.nan
        data[14, 2, 2] = -9999
        data[14, 3, 3] = -1
        with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=15,
                           dtype="float32", crs="EPSG:32648",
                           transform=from_origin(465100, 982250, 10, 10), nodata=-9999) as dst:
            dst.write(data)
            dst.descriptions = tuple(BANDS)
        return data

    def test_conversion_keeps_pixel_alignment_without_forest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "VietNam_CaMau_2021_2022_S1_S2_GEDI.tif"
            data = self.make_tif(source)
            manifest = converter.convert_tif(source, root / "processed", patch_size=4)
            output = root / "processed" / "VietNam" / "CaMau" / "2021_2022"
            self.assertEqual(manifest["patch_count"], 4)
            self.assertNotIn("site_id", manifest)
            for row in (0, 4):
                for col in (0, 4):
                    stem = f"r{row:06d}_c{col:06d}"
                    metadata = json.loads((output / "metadata" / f"{stem}.json").read_text())
                    image = np.load(output / "input" / f"{stem}.npy")
                    target = np.load(output / "gedi" / f"{stem}.npy")
                    valid = np.load(output / "input_valid_mask" / f"{stem}.npy")
                    original = data[:, row:row + 4, col:col + 4]
                    expected_input = np.where(original[:14] == -9999, 0.0, original[:14])
                    expected_target = original[14].copy()
                    expected_target[expected_target < 0] = np.nan
                    np.testing.assert_array_equal(image, expected_input)
                    np.testing.assert_array_equal(target, expected_target)
                    np.testing.assert_array_equal(valid, (original[:14] != -9999).all(axis=0))
                    self.assertEqual(metadata["transform"], [10, 0, 465100 + col * 10, 0, -10, 982250 - row * 10])
                    self.assertEqual(metadata["bands"]["input"], BANDS[:14])
                    self.assertNotIn("site_id", metadata)
            # A repeat conversion must fail before deleting generated patches.
            with self.assertRaises(FileExistsError):
                converter.convert_tif(source, root / "processed", patch_size=4)

    def test_recursive_discovery_separates_provinces_and_periods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for province, period, value in (("CaMau", "2021_2022", 42),
                                            ("CaMau", "2022_2023", 43),
                                            ("CanGio", "2021_2022", 44)):
                self.make_tif(root / "raw" / province / "S1_S2_GEDI" /
                              f"VietNam_{province}_{period}_S1_S2_GEDI.tif", value)
            # This is not an input raster and must never be opened by discovery.
            shot = root / "raw" / "CaMau" / "shot_number" / "VietNam_CaMau_2021_2022_GEDI_shot_number.tif"
            shot.parent.mkdir(parents=True)
            shot.write_text("not a Sentinel stack")
            argv = ["tif_to_npy.py", "--raw-root", str(root / "raw"),
                    "--output-root", str(root / "processed"), "--patch-size", "4"]
            with patch("sys.argv", argv):
                converter.main()
            manifests = sorted((root / "processed").rglob("manifest.json"))
            self.assertEqual(len(manifests), 3)
            self.assertEqual({p.parent.relative_to(root / "processed").as_posix() for p in manifests},
                             {"VietNam/CaMau/2021_2022", "VietNam/CaMau/2022_2023", "VietNam/CanGio/2021_2022"})


if __name__ == "__main__":
    unittest.main()
