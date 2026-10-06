<div align="center">
  
## Depth Any Canopy: Leveraging Depth Foundation Models for Canopy Height Estimation

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Isaac Corley**](https://isaacc.dev/)<sup>2</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy&emsp;&emsp;&emsp;&emsp;<sup>2</sup>University of Texas at San Antonio, USA

**[ECCV 2024 CV4E Workshop](https://cv4e.netlify.app/)**

<a href="https://arxiv.org/abs/2408.04523"><img src='https://img.shields.io/badge/arXiv-Depth%20Any%20Canopy-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/DarthReca/depth-any-canopy-small'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Small%20Version-yellow'></a>
</div>

**In this paper, we propose transferring the representations learned by recent depth estimation foundation models to the remote sensing domain for measuring canopy height.** Our findings suggest that our proposed Depth Any Canopy, the result of fine-tuning the Depth Anything v2 model for canopy height estimation, provides a performant and efficient solution, surpassing the current state-of-the-art with superior or comparable performance using only a fraction of the computational resources and parameters. Furthermore, our approach requires less than \$1.30 in compute and results in an estimated carbon footprint of 0.14 kgCO2.

### Sentinel-GEDI smoke test

The active pipeline expects data outside this repository with matching file names:

```text
data/CarbonStock/
|-- Sentinel/NgocHien/<patch>.npy  # float32 [14, H, W]
`-- GEDI/NgocHien/<patch>.npy      # float32 [H, W], NaN where unlabeled
```

From `code/CarbonStock`, install the dependencies with a CUDA-enabled PyTorch
build compatible with the target machine, verify CUDA, and run:

```powershell
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python main.py
```

The default config runs the 14-channel Depth Anything ViT-S model for 100 epochs
on one GPU. It discovers and validates matching patches, computes normalization
statistics from the training split, uses every train/validation batch, saves a
checkpoint, and reports validation RMSE in metres using only finite GEDI pixels
in `(0, 30]`.

The optimizer and learning-rate schedule match the previous `tqk` training code:
AdamW with a `5e-6` maximum learning rate and OneCycleLR stepped after every
optimization step. The batch size is 4. The runtime input size is 252, which is
divisible by the model's 14-pixel patch size.

By default, compatible weights are loaded only from the local RGB Depth Any
Canopy ViT-S snapshot at `offline_models/depth-any-canopy-small`; Hugging Face
network access is disabled. Its input projection is reinitialized for 14 bands.
GEDI remains in metres in the dataset so filtering stays in `(0, 30]`, while the
training target is divided by 30 to match the pretrained normalized output.
Loss is normalized MSE; RMSE and predictions are converted back to metres.

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

For a quick one-batch pipeline check, run:

```powershell
python main.py model.architecture=tiny trainer.max_epochs=1 trainer.limit_train_batches=1 trainer.limit_val_batches=1 trainer.enable_checkpointing=false
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
