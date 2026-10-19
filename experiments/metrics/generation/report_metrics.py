from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import csv
import math

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from brainmint.metrics.diffusion_metrics import DiffusionMetricAggregator, DiffusionMetricCalculator

from medmetric.extractors.medicalnet import MedicalNetFeatureExtractor

# WFDM RFlow-L2 is the release default.
EXP_CONFIG_NAME = "exp/metrics/generation/wfdm/rflow_l2"


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = str(REPO_ROOT / "configs")


@dataclass(frozen=True)
class ReportMetadata:
    model_name: str = ""
    study_name: str = ""
    conditioning: str = ""
    notes: str = ""

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ReportMetadata":
        return cls(
            model_name=str(cfg.get("model_name", "")),
            study_name=str(cfg.get("study_name", "")),
            conditioning=str(cfg.get("conditioning", "")),
            notes=str(cfg.get("notes", "")),
        )


def _resolve_device(device_cfg: Any) -> torch.device:
    if device_cfg is None or str(device_cfg).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_cfg))


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [value.item()]
        return value.detach().cpu().tolist()
    return [value]


def _string_list(value: Any) -> List[str]:
    items: List[str] = []
    for item in _to_list(value):
        if item is None:
            items.append("")
        elif isinstance(item, bytes):
            items.append(item.decode("utf-8"))
        else:
            items.append(str(item))
    return items


# column widths
def fmt_num(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if math.isnan(v):
        return "nan"
    return f"{v:.6g}"


def _write_table_txt(path: Path, rows: List[Dict[str, Any]], metadata: ReportMetadata | None = None) -> None:
    headers = ["Modality", "Count", "FID", "MMD", "MS-SSIM"]
    lines: List[str] = []

    if metadata is not None and any([metadata.model_name, metadata.study_name, metadata.conditioning, metadata.notes]):
        lines.append("Report")
        if metadata.model_name:
            lines.append(f"  model_name: {metadata.model_name}")
        if metadata.study_name:
            lines.append(f"  study_name: {metadata.study_name}")
        if metadata.conditioning:
            lines.append(f"  conditioning: {metadata.conditioning}")
        if metadata.notes:
            lines.append(f"  notes: {metadata.notes}")
        lines.append("")

    table_rows = [
        [
            str(r.get("modality", "")),
            str(r.get("count", "")),
            fmt_num(r.get("fid")),
            fmt_num(r.get("mmd")),
            fmt_num(r.get("ms_ssim")),
        ]
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for tr in table_rows:
        for i, cell in enumerate(tr):
            widths[i] = max(widths[i], len(cell))

    def join_row(cells: Sequence[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines.append(join_row(headers))
    lines.append("-+-".join("-" * w for w in widths))
    for tr in table_rows:
        lines.append(join_row(tr))

    path.parent.mkdir(parents=True, exist_ok=True)
    print("\n".join(lines) + "\n")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_brainscape_stats(cfg: DictConfig) -> Dict[str, Dict[str, float]]:
    stats_container = OmegaConf.to_container(cfg, resolve=True)
    modalities = (
        stats_container.get("dataset", {})
        .get("stats", {})
        .get("brainscape", {})
        .get("modalities", None)
    )
    if not isinstance(modalities, dict):
        raise KeyError(
            "Expected dataset.stats.brainscape.modalities mapping in config, "
            f"got {type(modalities).__name__}."
        )
    stats: Dict[str, Dict[str, float]] = {}
    for modality, values in modalities.items():
        if not isinstance(values, dict):
            raise TypeError(f"Value must be dict, but got:{type(values)}")
        if str(modality).strip().lower() == "all":
            continue
        if "mean" not in values or "std" not in values:
            raise KeyError(f"Missing mean/std for modality '{modality}' in dataset stats config.")
        stats[str(modality).strip().lower()] = {
            "mean": float(values["mean"]),
            "std": float(values["std"]),
        }
    return stats


@hydra.main(config_path=CONFIG_PATH, config_name=EXP_CONFIG_NAME, version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Hydra run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(int(cfg.get("seed", 42)), workers=True)
    device = _resolve_device(cfg.get("device"))
    print(f"Using device: {device}")

    # Instantiate datamodule
    datamodule = instantiate(cfg.dataset.brainscape)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    if test_loader is None:
        raise RuntimeError("Datamodule did not provide a test dataloader.")

    # Feature extractor (MedicalNet ResNet50 by default)
    feat_extractor = MedicalNetFeatureExtractor.from_pretrained(
        depth=50,
        use_23dataset=True,
        device=device,
    )
    feat_extractor.eval()
    feat_extractor.to(device)

    # Prefer cfg.metrics.modalities, fallback to dataset.brainscape.modalities
    modalities_cfg = getattr(cfg.metrics, "modalities", None)
    if modalities_cfg is None or len(_to_list(modalities_cfg)) == 0:
        modalities_cfg = getattr(cfg.dataset.brainscape, "modalities", [])
    report_modalities = [str(m).strip().lower() for m in _to_list(modalities_cfg)]

    # Feature extraction intensity normalization (used before FID/MMD).
    intensity_norm = getattr(cfg.metrics, "intensity_normalization", "zscore-dataset")

    brainscape_stats = _load_brainscape_stats(cfg)

    calc = DiffusionMetricCalculator(
        feature_extractor=feat_extractor,
        modalities=report_modalities,
        intensity_normalization=intensity_norm,
        modality_stats=brainscape_stats,
        mask_quantile=0.10,
        mask_eps=1e-3,
    )
    ms_cfg = getattr(cfg.metrics, "ms_ssim", {})
    agg = DiffusionMetricAggregator(
        modalities=report_modalities,
        seed=int(getattr(cfg, "seed", 42)),
        device=device,
        ms_ssim_data_range=float(getattr(ms_cfg, "data_range", 1.0)),
        ms_ssim_intensity_normalization=getattr(ms_cfg, "intensity_normalization", None),
        ms_ssim_kernel_type=getattr(ms_cfg, "kernel_type", None),
        ms_ssim_kernel_sigma=getattr(ms_cfg, "kernel_sigma", None),
        ms_ssim_k1=getattr(ms_cfg, "k1", None),
        ms_ssim_k2=getattr(ms_cfg, "k2", None),
        ms_ssim_max_pairs=int(getattr(ms_cfg, "max_pairs", 5000)),
        ms_ssim_pair_batch_size=int(getattr(ms_cfg, "pair_batch_size", 16)),
    )

    keys_cfg = cfg.metrics.get("keys")
    real_key = str(keys_cfg.get("real"))
    syn_key  = str(keys_cfg.get("synthetic"))
    mod_key  = str(keys_cfg.get("modality"))

    tqdm_total = None
    if isinstance(test_loader, Sized):
        try:
            tqdm_total = len(test_loader)
        except TypeError:
            tqdm_total = None

    total_samples = 0
    for batch in tqdm(test_loader, total=tqdm_total, desc="Diffusion metrics", leave=True):
        if real_key not in batch or syn_key not in batch:
            raise KeyError(
                f"Batch missing required keys. Need '{real_key}' and '{syn_key}'. Got: {list(batch.keys())}"
            )
        if mod_key not in batch:
            raise KeyError(f"Batch missing modality key '{mod_key}'. Got: {list(batch.keys())}")

        real = batch[real_key].to(device=device, dtype=torch.float32)
        syn = batch[syn_key].to(device=device, dtype=torch.float32)

        batch_size = int(real.shape[0])
        total_samples += batch_size

        mods = [m.strip().lower() for m in _string_list(batch[mod_key])]

        if len(mods) != batch_size:
            raise ValueError(f"Modality list length {len(mods)} != batch size {batch_size}.")

        real_feats, syn_feats, syn_images = calc.process_batch(syn=syn, real=real, modalities_batch=mods)

        # print(f"REAL FEATURE SHAPE: {real_feats.shape}")
        # print(f"SYN FEATURE SHAPE: {syn_feats.shape}")
        # print(f"MS-SSIM FEATURE SHAPE: {ms_div}")

        agg.add_features_batch(modalities=mods, real_feats=real_feats, syn_feats=syn_feats, syn_images=syn_images)

    rows = agg.summary_rows()
    print(f"Processed {total_samples} volumes.")

    # Write summary CSV
    out_summary = run_dir / str(cfg.metrics.output.summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["modality", "count", "fid", "mmd", "ms_ssim"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write a human-readable table
    metadata = ReportMetadata.from_config(cfg.get("report", {}))
    out_table = run_dir / str(cfg.metrics.output.table_txt)
    _write_table_txt(out_table, rows, metadata=metadata)

    print(f"[done] wrote: {out_summary}")
    print(f"[done] wrote: {out_table}")


if __name__ == "__main__":
    main()
