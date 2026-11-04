# CarbonStock

Estimate aboveground biomass density (AGBD, Mg/ha) from 14 Sentinel-1/2 bands
using Depth Anything ViT-S and sparse GEDI L4A labels. Adapted from
[Depth Any Canopy](https://arxiv.org/abs/2408.04523) by Daniele Rege Cambrin,
Isaac Corley and Paolo Garza.

Run commands from this repository's root (`code/CarbonStock`).

## Setup

Install a PyTorch build suitable for your CPU/CUDA environment, then:

```bash
python -m pip install -r requirements.txt
```

Place the offline model's `config.json` and `model.safetensors` in
`offline_models/depth-any-canopy-small/`. The model loads locally without
network access. Settings are in [configs/default.yaml](configs/default.yaml).

## Data

Raw TIFF names follow
`<country>_<province>_<site>_<YYYY_YYYY>_S1_S2_GEDI.tif`.
The first 14 bands are Sentinel inputs in a consistent order; a later band
must be named `AGBD` or `GEDI_AGBD` (Mg/ha).

```bash
python dataset/tif_to_npy.py
```

The converter reads `../../data/CarbonStock/raw` and creates:

```text
../../data/CarbonStock/processed/<country>/<province>/<site>/<period>/
|-- input/<patch>.npy             # [14, H, W]
|-- gedi/<patch>.npy              # [H, W]; NaN = unlabeled
|-- input_valid_mask/<patch>.npy  # [H, W], boolean
|-- metadata/<patch>.json         # CRS, footprint, period and QA
`-- manifest.json
```

Copy all folders, including metadata. Supplied GEDI QA bands are applied;
missing QA is recorded as unverified. Use `--quality-filtered-provenance`
to document known upstream filtering.

- `dataset.regions=null` reads all periods recursively under `data_root`.
  To select one forest, set `data_root` to its folder; an explicit `regions`
  list must name individual period folders.
- Spatial train/validation/test strips use 60/20/20 of geographic width,
  with a 1 km buffer. Repeated locations stay in the same split across years.
  Normalization and target scaling use training data only. The default CRS,
  EPSG:32648, should be reviewed when changing study areas.
- `dataset.max_agbd=500` excludes labels above 500 while reading arrays;
  zero and exactly 500 remain valid. No reconversion is needed. Set it to
  `null` to disable filtering. Predictions are not capped.

## Train

```bash
python main.py
# Optionally evaluate the best validation checkpoint on test:
python main.py run_test=true
# Submit to Slurm after adjusting the module and .venv in train.slurm:
bash submit_train.sh
```

Defaults: 100 epochs, batch size 4, input size 252, learning rate 5e-6.
Override settings on the command line, e.g. `python main.py seed=43`.
Each run saves checkpoints, history, `data_summary.json` and `metrics.json`
under `runs/agbd/<timestamp>/`; the Slurm launcher also puts job logs there.
`runs/` and `offline_models/` are ignored by Git; transfer them separately.

## Analyze

Replace `<run-dir>` with a specific run folder:

```bash
python -m analysis.analyze_run --run-dir <run-dir>
python -m analysis.evaluate_checkpoint --run-dir <run-dir> --device auto
# Test evaluation after model selection:
python -m analysis.evaluate_checkpoint --run-dir <run-dir> --splits test --output-dir <run-dir>/evaluation-test
```

Analysis writes learning curves, split maps and label statistics by split
and period to `<run-dir>/analysis/`, with warnings for missing period labels.
Evaluation writes RMSE/MAE/bias/R2, predictions and group tables to
`<run-dir>/evaluation/`. Both reuse the saved split, normalization and label
filter; older runs without `max_agbd` retain their original label policy.
Use `--data-root` if data moved, or analysis `--history-only` without arrays.
Pixel counts do not represent independent GEDI footprints.

## License

[Apache 2.0](LICENSE). Original model:
[Depth Any Canopy Small](https://huggingface.co/DarthReca/depth-any-canopy-small).
