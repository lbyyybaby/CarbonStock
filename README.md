# CarbonStock

Estimate aboveground biomass density (AGBD, Mg/ha) from 14 Sentinel-1/2 bands
using Depth Anything ViT-S and sparse GEDI L4A labels. Adapted from
[Depth Any Canopy](https://arxiv.org/abs/2408.04523) by Daniele Rege Cambrin,
Isaac Corley and Paolo Garza.

Run commands from this repository's root (`code/CarbonStock`).

## Setup

Use Python 3.11 and a PyTorch build suitable for your CPU/CUDA environment, then:

```bash
python -m pip install -r requirements.txt
```

Place the offline model's `config.json` in
`offline_models/depth-any-canopy-small/`. By default, `model.pretrained=false`
initializes the entire model randomly, including the encoder and regression
head; no pretrained weights are loaded. Keep `model.use_huggingface=true`
to build the same architecture from this local config without network access.
Settings are in [configs/default.yaml](configs/default.yaml).

## Data

Raw TIFF names follow
`<country>_<province>_<YYYY_YYYY>_S1_S2_GEDI.tif`, for example
`VietNam_CaMau_2021_2022_S1_S2_GEDI.tif`. There is no forest/site component.
The first 14 bands are Sentinel inputs in a consistent order; a later band
must be named `AGBD` or `GEDI_AGBD` (Mg/ha).
Training validates their names and order against `dataset.expected_input_bands`:
B1/B2/B3/B4/B5/B6/B7/B8/B8A/B9/B11/B12, then VV/VH.

```bash
python dataset/tif_to_npy.py
# Or select a downloaded dataset root; source TIFFs are discovered recursively:
python dataset/tif_to_npy.py --raw-root /path/to/raw --output-root /path/to/processed
```

The converter reads `../../data/CarbonStock/raw` and creates:

```text
../../data/CarbonStock/processed/<country>/<province>/<period>/
|-- input/<patch>.npy             # [14, H, W]
|-- gedi/<patch>.npy              # [H, W]; NaN = unlabeled
|-- input_valid_mask/<patch>.npy  # [H, W], boolean
|-- metadata/<patch>.json         # CRS, footprint, period and QA
`-- manifest.json
```

Copy all folders, including metadata. Supplied GEDI QA bands are applied;
missing QA is recorded as unverified. Use `--quality-filtered-provenance`
to document known upstream filtering.

- The default `dataset.data_root` selects `processed/VietNam/CaMau`;
  `dataset.regions=null` reads all periods recursively under that province.
  To select Can Gio, override `dataset.data_root` with
  `../../data/CarbonStock/processed/VietNam/CanGio`. An explicit `regions`
  list must name individual period folders.
- Raw downloads keep `<province>/S1_S2_GEDI/` separate from
  `<province>/shot_number/` (shot rasters, CSVs and check reports). The converter
  only discovers `*_S1_S2_GEDI.tif`, so shot rasters are not treated as inputs.
- Spatial train/validation/test strips use 60/20/20 of geographic width,
  with a 1 km buffer. Repeated locations stay in the same split across years.
  Normalization and target scaling use training data only. The default CRS,
  EPSG:32648, should be reviewed when changing study areas.
- `dataset.max_agbd=500` excludes labels above 500 while reading arrays;
  zero and exactly 500 remain valid. No reconversion is needed. Set it to
  `null` to disable filtering. Predictions are not capped.

The current raw dataset is
[WindyTheNguyen/camau-s1-s2-gedi](https://huggingface.co/datasets/WindyTheNguyen/camau-s1-s2-gedi).
Use its `CaMau/` and `CanGio/` folders. The legacy `rasters/` directory duplicates
the Ca Mau TIFFs under older names and should not be included in the raw root.

Run the data preparation regression checks with:

```bash
python -m unittest discover -s tests -v
```

## Train

```bash
python main.py
# Optionally evaluate the best validation checkpoint on test:
python main.py run_test=true
# Slurm uses ~/envs/carbonstock/bin/python by default:
bash submit_train.sh
# Override the environment when needed, before submitting:
export CARBONSTOCK_PYTHON=/path/to/environment/bin/python
# Explicitly select the default ViT Plateau configuration:
bash submit_train.sh --config-name default
# Short server check with the full model, before a complete training run:
bash submit_train.sh trainer.max_epochs=2 trainer.limit_train_batches=2 trainer.limit_val_batches=1
```

Defaults: ViT-S from scratch, 100 epochs, batch size 4, input size 252,
learning rate 3e-5 and pixel-weighted MSE. ReduceLROnPlateau monitors validation
RMSE (factor 0.5, patience 2, minimum LR 1e-7); early stopping uses patience 12.
Override settings on the command line, e.g. `python main.py seed=43`.

Each run saves checkpoints, history, `data_summary.json` and `metrics.json`
under `runs/agbd/<timestamp>/`; the Slurm launcher also puts job logs there.
`runs/` and `offline_models/` are ignored by Git; transfer them separately.
Each new run also records package versions, setup timings and source hashes.

Pixel weighting divides each batch's squared-error sum by its patch count
times the training-derived mean valid-label count per patch. It does not turn
raster pixels into independent GEDI observations.

Only the ViT architecture and the standalone `default.yaml` configuration are
maintained. Historical experiment reports and run artifacts remain available.
Training uses MSE with a single learning rate and the model's standard ReLU
output. Experimental RGB projection adaptation, head resets, alternative
losses, separate backbone learning rates and augmentation have been removed.
The configured buffer remains 1 km; to reproduce the 8 November experiments,
explicitly pass `dataset.buffer_m=500` (those experiments used a 500 m buffer).

### Run independently of SSH

`bash submit_train.sh` submits a Slurm batch job and returns after printing
the job ID and run directory. Once submission succeeds, the job is managed
by Slurm and continues if you close the terminal or disconnect SSH.

Replace `12345` and `<run-dir>` with the values printed by the launcher:

```bash
squeue -j 12345
# Inspect completed jobs or jobs that disappeared from squeue:
sacct -j 12345 --format=JobID,JobName,State,Elapsed,ExitCode
tail -f <run-dir>/train-12345.out <run-dir>/train-12345.err
# Stop viewing logs with Ctrl+C; the Slurm job keeps running.
# Cancel the training job when needed:
scancel 12345
```

`PD` means queued for resources; `R` means running. The job ID is also saved
in `<run-dir>/slurm_job_id.txt`. In `sacct`, `COMPLETED` means finished;
`CANCELLED`, `FAILED`, `TIMEOUT` and `OUT_OF_MEMORY` mean the job stopped.
The current `train.slurm` requests an 8-hour
time limit. If training needs more time, increase `#SBATCH --time` before
submission, within the GPU partition's allowed limit. Configuration changes
apply to new jobs; an existing training process retains its loaded settings.

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
