<div align="center">
  
## Depth Any Canopy: Leveraging Depth Foundation Models for Canopy Height Estimation

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Isaac Corley**](https://isaacc.dev/)<sup>2</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy&emsp;&emsp;&emsp;&emsp;<sup>2</sup>University of Texas at San Antonio, USA

**[ECCV 2024 CV4E Workshop](https://cv4e.netlify.app/)**

<a href="https://arxiv.org/abs/2408.04523"><img src='https://img.shields.io/badge/arXiv-Depth%20Any%20Canopy-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/DarthReca/depth-any-canopy-small'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Small%20Version-yellow'></a>
</div>

**In this paper, we propose transferring the representations learned by recent depth estimation foundation models to the remote sensing domain for measuring canopy height.** Our findings suggest that our proposed Depth Any Canopy, the result of fine-tuning the Depth Anything v2 model for canopy height estimation, provides a performant and efficient solution, surpassing the current state-of-the-art with superior or comparable performance using only a fraction of the computational resources and parameters. Furthermore, our approach requires less than \$1.30 in compute and results in an estimated carbon footprint of 0.14 kgCO2.

### Sentinel-GEDI L4A biomass training

The configured model remains the pretrained 14-channel Depth Anything ViT-S.
The baseline uses this same model to learn AGBD, without adding another model.
The active pipeline reads the existing processed layout:

```text
data/CarbonStock/processed/<country>/<province>/<site>/<period>/
|-- input/<patch>.npy             # float32 [14, H, W]
|-- gedi/<patch>.npy              # AGBD [H, W], Mg/ha; NaN = unlabeled
|-- input_valid_mask/<patch>.npy  # boolean [H, W]
|-- metadata/<patch>.json         # Country, CRS, footprint, AGBD and available QA
`-- manifest.json
```

Raw filenames now include the country, for example
`VietNam_CaMau_CaMauMangrove_2021_2022_S1_S2_GEDI.tif`. Convert files from
`../../data/CarbonStock/raw` with:

```bash
python dataset/tif_to_npy.py
```

The first 14 TIFF bands are Sentinel inputs. One later band must be named
`AGBD` or `GEDI_AGBD` (Mg/ha); unrelated extra bands are ignored. There is no
RH98 check. Quality bands are optional: supplied L4A flags are still applied
(`algorithm_run_flag=1`, `l4_quality_flag=1`, `degrade_flag=0`, optional predictor
and response limit flags equal to 0). Missing flags do not block conversion or
training, and metadata records `quality_filter.applied=null` when the full
policy is unverified. To record known upstream filtering, optionally pass
`--quality-filtered-provenance "<export filtering description>"`.
Existing processed folders without the country level remain readable; they
are not moved automatically. Do not keep duplicate old and new exports under
the same training data root. The L4A band definitions and units are documented in the
[Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI04_A_002_MONTHLY).

Train/validation/test occupy separate west-to-east geographic strips in
`dataset.split_crs` (default EPSG:32648, metres), with extent fractions 60/20/20.
Full patch footprints intersecting the 1 km boundary buffers are excluded from
fitting and evaluation. Fractions describe geographic width, not patch counts.
Repeated locations across years cannot cross partitions. An empty labeled
partition raises an error; validation is never reused as test.
Adding a country folder does not change this split into a per-country split.
For additional countries, review `dataset.split_crs` and the geographic extent
before training; the default projected CRS is intended for the current region.

Loss uses only finite, nonnegative AGBD labels with valid Sentinel inputs.
`dataset.max_agbd: 500.0` excludes labels above 500 Mg/ha from training,
validation, test and label statistics; exactly 500 and zero remain valid.
The filter is applied while reading processed arrays, so conversion does not
need to be rerun and the stored labels are preserved. Set
`dataset.max_agbd=null` to disable it, or change the threshold with a Hydra
override, for example `python main.py dataset.max_agbd=500`.
Predictions are not capped by this label filter.
All patches, including unlabeled and buffer patches,
remain in `predict_dataloader()` for later mapping. Input normalization uses
only the geographic training region. The biomass scale is the 95th percentile
of valid training labels (minimum scale 1 Mg/ha); it neither removes nor clips
labels or predictions. Loss is normalized MSE, while RMSE and `prediction_agbd`
are in Mg/ha with no upper output limit. Validation/test also report RMSE for
a constant predictor equal to mean training AGBD.

From `code/CarbonStock`, a basic local check uses the same configured model
with a smaller input and one train/validation batch:

```powershell
python main.py trainer.accelerator=cpu dataset.num_workers=0 dataset.batch_size=1 model.image_size=56 trainer.max_epochs=1 trainer.limit_train_batches=1 trainer.limit_val_batches=1 trainer.enable_checkpointing=false output_dir=runs/agbd-smoke
```

Local learnability check (2026-10-29): the configured pretrained ViT-S, with all
parameters trainable, learned one 56x56 training crop containing 20 valid AGBD
pixels in 30 CPU AdamW steps at the configured learning rate 5e-6. Normalized
MSE fell from 0.23694 to 0.00310; crop RMSE fell from 50.56 to 5.79 Mg/ha.
This confirms optimization on a small training subset, not generalization.
The separate `main.py` one-batch training/validation and checkpoint smoke check
also passed. The current independent test contains only 2 labeled patches
(410 valid pixels), so a full test score will have limited geographic support.

For the full server run, install a compatible CUDA-enabled PyTorch build and
the requirements, copy the processed data **including its JSON metadata**, and
run:

```bash
python -m pip install -r requirements.txt
python main.py run_test=true
```

Default training remains 100 epochs, batch size 4, input size 252, and maximum
learning rate 5e-6 with AdamW/OneCycleLR. It loads the local RGB Depth Any Canopy
snapshot at `offline_models/depth-any-canopy-small` with networking disabled;
the input projection is reinitialized for 14 bands. Each run's `data_summary.json`
records geographic assignments, buffers and training normalization/scale;
`metrics.json` records results. The best validation checkpoint is selected
before independent test evaluation. `run_test=false` (default) leaves test
evaluation until explicitly requested.

### Slurm runs and generated files

Keep `train.slurm` and `submit_train.sh` in Git. The latter creates a unique
directory using Bangkok time before submission, so Slurm can open its log
files there. From the repository root on the server:

```bash
bash submit_train.sh
# Optional Hydra overrides:
bash submit_train.sh seed=43 run_test=true
```

The launcher prints the job ID and directory. For example:

```text
runs/agbd/2026-11-03_14-30-00-ABC123/
|-- train-<jobid>.out
|-- train-<jobid>.err
|-- train.slurm                  # Copy of the submitted script
|-- slurm_job_id.txt
|-- .hydra/                      # Configuration and command-line overrides
|-- main.log
|-- checkpoints/
|-- history/
|-- data_summary.json
`-- metrics.json
```

`CARBONSTOCK_RUN_DIR` links the launcher's log directory to `output_dir` in
`configs/default.yaml`. Avoid overriding `output_dir` with the launcher if
you want all artifacts together. Direct `python main.py` runs also default to
a timestamped directory, using the process's local timezone. Direct
`sbatch train.slurm` remains possible, but its Slurm logs use the fallback
`train-<jobid>.out/.err` paths in the submission directory.

The module and `.venv` in `train.slurm` must exist on the server. Activating
`.venv` selects that environment's interpreter; loading a module does not
upgrade an existing virtual environment. The script prints the actual Python,
PyTorch and CUDA status before training.

`runs/` is intentionally not ignored, so run results can be committed and
transferred between the server and local checkout. Review the selected run
files before committing. Root-level fallback `train-*.out/.err` and
`slurm-*.out/.err` logs are ignored; logs inside `runs/` remain eligible for Git.
The two logs in `runs/agbd/job-7349/` are versioned examples of a server run.
Its timestamp and other artifacts were not supplied, so the folder contains
only the available logs. The existing global `*.log` rule still ignores
`main.log`; CSV history, JSON summaries, Slurm logs and checkpoints in `runs/`
are not ignored.

### Analyze a saved run

Install the updated requirements (analysis plots use Matplotlib), then run:

```bash
python -m analysis.analyze_run --run-dir runs/agbd
python -m analysis.evaluate_checkpoint --run-dir runs/agbd --device auto
```

The analysis commands live together in the `analysis/` package. Run them from
the repository root with `python -m` as above.

`analysis/analyze_run.py` merges sparse CSV log rows by epoch, plots RMSE/loss and the
logged learning rates, and replays early stopping on the recorded validation
trace. It also checks the saved geographic split against the processed arrays,
reports label distributions and patch counts, and plots the split. Label
statistics, percentiles and threshold counts/percentages are printed as tables
in `report.md`. Results go to `runs/agbd/analysis/`: `report.md`,
`analysis_report.json`, `history_by_epoch.csv`, `label_statistics.csv`,
`label_statistics_by_period.csv`,
`patch_audit.csv`, and just two plots: `learning_curves.png` and `split_map.png`.
The period table groups labels by country/province/site/period and split,
including empty splits. It reports patch counts and AGBD distributions, and
warns when a period has no train, validation or test labels. The same table
and warnings are included in `report.md` and `analysis_report.json`.
Rerunning the script removes the obsolete `label_distributions.png` if present.
Use `--history-only` when the processed data is unavailable.

`analysis/evaluate_checkpoint.py` loads the trained checkpoint with strict weight
matching and evaluates all labeled patches in the saved train/validation
split. It uses the saved normalization, target scale and training target mean;
it does not fit a model or recompute the split. Results go to
`runs/agbd/evaluation/`: `report.md`, `evaluation_metrics.json`,
`predictions.csv`, `per_patch.csv`, `by_agbd_bin.csv`, `by_region_period.csv`,
and prediction/residual/patch-error PNGs. Residual and bias mean prediction
minus target. RMSE, MAE and bias use Mg/ha; the mean of patch RMSEs is reported
separately from pooled pixel-weighted RMSE. Bin intervals are left-inclusive
and right-exclusive: `[0,30)`, `[30,100)`, `[100,200)`, `[200,inf)`.

Both scripts accept `--data-root /path/to/processed`. Old IDs without a country
prefix can be mapped to a unique country folder. If multiple countries match,
select one with `--country-prefix VietNam`; no match is silently guessed.
The audit checks geometry, array/mask shapes, per-patch label counts, and saved
training normalization/label statistics. Original data hashes were not saved,
so these checks cannot prove that every file is byte-identical to the training
data. Extra current patches are not added to the saved split.
Both commands use the label ceiling recorded in the saved run's dataset
configuration. Older runs without `max_agbd` retain their original unbounded
label policy; changing the current default does not reinterpret old runs.

The evaluator resolves the copied checkpoint under the run's `checkpoints/`
directory before using its recorded server path. Override paths with
`--checkpoint /path/to/model.ckpt` and
`--hf-model-path /path/to/offline_models/depth-any-canopy-small` if needed.
`--device auto` selects CUDA when available, otherwise CPU. CPU evaluation
defaults to four threads; use `--cpu-threads`, `--batch-size` and `--num-workers`
to adjust resources. Only local model configuration and the trained checkpoint
are loaded; the original RGB pretrained weights are not reloaded.

Test prediction is opt-in. After model selection, evaluate it separately:

```bash
python -m analysis.evaluate_checkpoint --run-dir runs/agbd --splits test --output-dir runs/agbd/evaluation-test
```

Dataset label statistics include all saved partitions, but the default evaluator
does not predict the test partition. Pixel counts do not imply independent
footprint counts. Sparse groups (especially a two-patch test) need to be
interpreted with their spatial coverage and label concentration.

To transfer the model to an offline server, copy and extract the prepared model
archive so that `offline_models/depth-any-canopy-small/config.json` and
`model.safetensors` exist. The prepared archive is
`offline_models/depth-any-canopy-small-5da90568.tar.gz` (revision
`5da905680afb5943a72966cfcc33a3e6d79e3c49`). On the server:

```bash
cd /path/to/CarbonStock/offline_models
sha256sum -c depth-any-canopy-small-5da90568.tar.gz.sha256
tar -xzf depth-any-canopy-small-5da90568.tar.gz
```

You can then verify strict offline loading with:

```bash
cd /path/to/CarbonStock
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py
```

To initialize from a local checkpoint, set
`model.use_huggingface=false model.pretrained=true model.checkpoint_path=C:/path/to/model.safetensors`.
Only tensors with matching names and shapes are loaded, so a checkpoint trained
with fewer input channels will leave the 14-channel patch embedding initialized
from scratch.

### Getting Started

All runtime settings now live in `configs/default.yaml`; command-line Hydra
overrides avoid editing the file for one-off runs.

### Pre-Trained Models

Pre-trained checkpoints are available on HuggingFace.

| Model | Parameters | Checkpoint | 
|:---|:---:|:---:|
| Depth-Any-Canopy-Small | 24.8M | [Download](https://huggingface.co/DarthReca/depth-any-canopy-small) |
| Depth-Any-Canopy-Base  | 97.5M | [Download](https://huggingface.co/DarthReca/depth-any-canopy-base) |

You can easily load them with *pipelines* or *AutoModel*:

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("depth-estimation", model="DarthReca/depth-any-canopy-base")

# Load model directly
from transformers import AutoModelForDepthEstimation

model = AutoModelForDepthEstimation.from_pretrained("DarthReca/depth-any-canopy-base")
```

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex
@inbook{RegeCambrin2025,
  title = {Depth Any Canopy: Leveraging Depth Foundation Models for Canopy Height Estimation},
  ISBN = {9783031923876},
  ISSN = {1611-3349},
  url = {http://dx.doi.org/10.1007/978-3-031-92387-6_5},
  DOI = {10.1007/978-3-031-92387-6_5},
  booktitle = {Computer Vision – ECCV 2024 Workshops},
  publisher = {Springer Nature Switzerland},
  author = {Rege Cambrin,  Daniele and Corley,  Isaac and Garza,  Paolo},
  year = {2025},
  pages = {71–86}
}
```
