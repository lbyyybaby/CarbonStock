<div align="center">
  
## Depth Any Canopy: Leveraging Depth Foundation Models for Canopy Height Estimation

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Isaac Corley**](https://isaacc.dev/)<sup>2</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy&emsp;&emsp;&emsp;&emsp;<sup>2</sup>University of Texas at San Antonio, USA

**[ECCV 2024 CV4E Workshop](https://cv4e.netlify.app/)**

<a href="https://arxiv.org/abs/2408.04523"><img src='https://img.shields.io/badge/arXiv-Depth%20Any%20Canopy-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/DarthReca/depth-any-canopy-small'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Small%20Version-yellow'></a>
</div>

**In this paper, we propose transferring the representations learned by recent depth estimation foundation models to the remote sensing domain for measuring canopy height.** Our findings suggest that our proposed Depth Any Canopy, the result of fine-tuning the Depth Anything v2 model for canopy height estimation, provides a performant and efficient solution, surpassing the current state-of-the-art with superior or comparable performance using only a fraction of the computational resources and parameters. Furthermore, our approach requires less than \$1.30 in compute and results in an estimated carbon footprint of 0.14 kgCO2.

### Ngoc Hien Sentinel--GEDI smoke test

The active pipeline expects data outside this repository with matching file names:

```text
data/CarbonStock/
|-- Sentinel/NgocHien/<patch>.npy  # float32 [14, H, W]
`-- GEDI/NgocHien/<patch>.npy      # float32 [H, W], NaN where unlabeled
```

From `code/CarbonStock`, install the dependencies once and run:

```powershell
python -m pip install -r requirements.txt
python main.py
```

The default config is deliberately a one-batch CPU smoke test. It discovers and
validates matching patches, computes 14-channel normalization statistics from the
training split, runs one optimization step, and reports validation RMSE in metres
using only finite GEDI pixels in `(0, 30]`. The tiny convolutional model checks the
pipeline only; its RMSE is not a model-quality result.

For a full randomly initialized 14-channel Depth Anything run, override only the
settings that differ from the smoke test:

```powershell
python main.py model.architecture=depth_anything model.image_size=518 trainer.accelerator=auto trainer.max_epochs=100 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.enable_checkpointing=true
```

To initialize from a local checkpoint, also set
`model.pretrained=true model.checkpoint_path=C:/path/to/model.safetensors`.
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
