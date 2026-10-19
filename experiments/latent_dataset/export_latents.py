import json
from pathlib import Path

import torch
import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

REPO_ROOT   = Path(__file__).resolve().parents[2]      # Main Repo
CONFIG_PATH = str(REPO_ROOT / "configs")               # Hydra Configs

def _clone_json_with_latents(
    src_json: Path,
    dst_json: Path,
    dataset_root: Path,
    add_sigma: bool = False,
) -> None:
    src_json = src_json.resolve()
    dst_json = dst_json.resolve()
    dataset_root = dataset_root.resolve()

    if src_json == dst_json:
        raise ValueError(
            "dst_json must differ from src_json to avoid overwriting the source dataset JSON"
        )
    if dataset_root == src_json.parent:
        raise ValueError(
            "dataset_root must differ from the source dataset root to avoid overwriting source data"
        )
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    with src_json.open("r") as f:
        data = json.load(f)

    def _postfix(p: str, suffix: str) -> str:
        # p is relative path under <DATASET>/preprocessed/
        if p.endswith(".nii.gz"):
            return p[:-7] + suffix + ".nii.gz"
        return p + suffix + ".nii.gz"

    output_splits = {}
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        new_recs = []
        for rec in data[split]:
            pre = rec.get("preprocessed", {}) or {}
            lat, recn, sigma_paths = {}, {}, {}
            skip = False
            for mod, rel in pre.items():
                if mod == "seg":  # Skip Segmentation Labels
                    continue

                full_pre = dataset_root / rec['dataset'] / "preprocessed"/ rel
                if not full_pre.exists():
                    # To be sure whatever is in JSON is also part of the dataset.
                    raise FileNotFoundError(f"Missing preprocessed file: {full_pre}")
                    skip = True
                    break

                lat_path = _postfix(rel, "_emb_mu")
                recon_path = _postfix(rel, "_recon")
                
                lat_abs = dataset_root / rec["dataset"] / "preprocessed" / lat_path
                recon_abs = dataset_root / rec["dataset"] / "preprocessed" / recon_path

                # Validate files exist in dataset_root
                if not lat_abs.exists():
                    raise FileNotFoundError(f"Missing latent file: {dataset_root / lat_path}")
                if not recon_abs.exists():
                    raise FileNotFoundError(f"Missing recon file: {dataset_root / recon_path}")
                
                if not lat_abs.exists() or not recon_abs.exists():
                    skip = True
                    break
                
                lat[mod] = lat_path
                recn[mod] = recon_path

                if add_sigma:
                    sigma_path = _postfix(rel, "_emb_sigma")
                    full_sigma = dataset_root / rec["dataset"] / "preprocessed" /sigma_path
                    if not full_sigma.exists():
                        raise FileNotFoundError(f"Missing sigma file: {full_sigma}")
                    sigma_paths[mod] = sigma_path
                

            if skip or not lat or not recn:
                continue
            
            new_rec = dict(rec)
            new_rec["latent"] = lat
            new_rec["recon"]  = recn
            if add_sigma and sigma_paths:
                new_rec["latent_sigma"] = sigma_paths

            new_recs.append(new_rec)
        
        if new_recs:
            output_splits[split] = new_recs


    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with dst_json.open("w") as f:
        json.dump(output_splits, f, indent=2)

@hydra.main(
    config_path=CONFIG_PATH,
    config_name="exp/latent_dataset/compression/wavelet_fusion/wf_sa_vae",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    
    run_dir = HydraConfig.get().runtime.output_dir  # absolute path
    print(f"Run directory : {run_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(cfg.seed, workers=True)

    dm = instantiate(cfg.dataset.brainscape)
    module = instantiate(cfg.lightning.export_latents_module)  # Exporter module

    # Trainer (predict only)
    callbacks = [instantiate(cb) for cb in cfg.get("callbacks", [])]
    loggers = [instantiate(lg) for lg in cfg.get("loggers", [])]
    trainer = pl.Trainer(
        **cfg.trainer.pl_trainer,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(run_dir),
        fast_dev_run=cfg.get("fast_dev_run", False),
    )

    # Run prediction for each split; module accumulates z_mu statistics internally
    module._sum_sq.zero_()
    module._count.zero_()
    splits = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }
    for name, loader in splits.items():
        trainer.predict(module, dataloaders=loader)
        num_batches = len(loader) if hasattr(loader, "__len__") else "?"
        print(f"[export] finished split={name}: batches={num_batches}")

    total_sum_sq = module._sum_sq.detach().to(dtype=torch.float64, device="cpu")
    total_count = module._count.detach().to(dtype=torch.float64, device="cpu")
    if total_count.item() <= 0:
        raise RuntimeError("No voxels processed; total_count is zero")

    # Compute scale_factor = 1/std(z_mu)
    std = torch.sqrt(total_sum_sq / total_count)
    scale_factor = float(1.0 / max(std.item(), 1e-12))

    # Write meta + cloned JSON with latent/recon keys
    out_root = Path(cfg.lightning.export_latents_module.out_dataset_dir)
    meta_dir = out_root 
    meta_dir.mkdir(parents=True, exist_ok=True)

    weight_loads = cfg.lightning.export_latents_module.hparams.weight_loads
    autoencoder_weights = next(
        (wl for wl in weight_loads if wl.get("target") == "autoencoder"),
        weight_loads[0] if weight_loads else {},
    )
    autoencoder_ckpt = autoencoder_weights.get("path", "")

    meta = {
        "scale_factor": scale_factor,
        "z_source": "mu",
        "autoencoder_ckpt": autoencoder_ckpt,
        "count_voxels": int(total_count.item()),
        "sum_sq": float(total_sum_sq.item()),
    }
    with (meta_dir / "latents_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    _clone_json_with_latents(
        src_json=Path(cfg.dataset.brainscape.json_path),
        dst_json=Path(cfg.out_dataset_json),
        dataset_root=Path(cfg.lightning.export_latents_module.out_dataset_dir),
        add_sigma=bool(cfg.lightning.export_latents_module.get("save_sigma", False)),
    )

    print(f"[export] scale_factor={scale_factor:.8f}")
    print(f"[export] wrote: {(meta_dir / 'latents_meta.json')}")
    print(f"[export] wrote: {cfg.out_dataset_json}")


if __name__ == "__main__":
    main()
