# WFDM MRI synthesis workflows

This repository contains training and inference workflows for the Wavelet-Fusion Diffusion Model (WFDM).
It depends on [BrainMint](https://pypi.org/project/brainmint/) for reusable model,
data, training, and inference components.

## Installation

Install [Conda](https://docs.conda.io/).

The provided environment targets NVIDIA CUDA 12.4 and installs Python, PyTorch,
BrainMint, MedMetric, MONAI, and the workflow utilities.

```bash
# From the repository root.
conda env create --file environment.yaml
conda activate brain-mri-synthesis
```

To update an existing environment after pulling changes:

```bash
conda env update --file environment.yaml --prune
```


## Required paths

Set these values in `configs/paths/paths.yaml` or provide equivalent Hydra command-line overrides. The public defaults intentionally use `???` and contain no local
paths.

| Key | Purpose |
| --- | --- |
| `paths.brainscape_root` | Root directory containing preprocessed BrainScape datasets. |
| `paths.brainscape_manifest` | BrainScape manifest with records, splits, paths, and metadata. |
| `paths.hydra_output_dir` | Parent directory for Hydra outputs, logs, and composed configs. |
| `paths.wf_sa_vae_checkpoint` | Trained WF-SA-VAE checkpoint. |
| `paths.wf_sa_vae_latent_dataset_root` | Output/input root for the exported WF-SA-VAE latent dataset. |
| `paths.wfdm_rflow_l2_ckpt` | Trained WFDM RFlow-L2 checkpoint. |
| `paths.wfdm_synthetic_root` | Parent directory for generated WFDM RFlow-L2 datasets. |

## Workflow

1. Train the WF-SA-VAE, or provide a compatible checkpoint.
2. Export a BrainScape MRI latent dataset with the WF-SA-VAE.
3. Train WFDM using the exported latent dataset and its manifest.
4. Generate individual conditioned samples or a synthetic cohort.
5. Evaluate the generated cohort with FID, MMD, and MS-SSIM.

```bash
# Train and evaluate the WF-SA-VAE compressor.
python experiments/train/compression/wavelet_fusion/train.py
python experiments/metrics/compression/report_metrics.py

# Export the latent dataset.
python experiments/latent_dataset/export_latents.py

# Train WFDM RFlow-L2.
python experiments/train/generation/wfdm/train.py

# Generate a conditioned sample. Override any conditioning values as needed.
python experiments/inference/generation/wfdm/generate_conditioned_images.py \
  inference.modalities='[t1w]' \
  inference.images_per_modality=1

# Generate the held-out synthetic cohort.
python experiments/synthetic_dataset/generate_synthetic_dataset.py

# Calculate FID, MMD, and MS-SSIM.
python experiments/metrics/generation/report_metrics.py
```

The generation-metrics workflow uses the public `medmetric` package and downloads
the MedicalNet weights on first use if they are not already cached.
