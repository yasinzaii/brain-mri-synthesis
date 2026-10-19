from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import csv
import math
import torch

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from brainmint.metrics.reconstruction import MetricAggregator, ReconstructionMetricCalculator

# WF-SA-VAE reconstruction metrics is the release default.
EXP_CONFIG_NAME = "exp/metrics/compression/wavelet_fusion/wf_sa_vae"

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = str(REPO_ROOT / "configs")

#METRIC_ORDER: Sequence[str] = ("lpips", "ssim", "ms_ssim", "psnr", "mse")
METRIC_ORDER: Sequence[str] = ReconstructionMetricCalculator.METRIC_NAMES

METRIC_TITLES: Mapping[str, str] = {
    "lpips": "LPIPS",
    "ssim": "SSIM",
    "ms_ssim": "MS-SSIM",
    "psnr": "PSNR",
    "mse": "MSE",
}
METRIC_PRECISION: Mapping[str, int] = {
    "lpips": 4,
    "ssim": 4,
    "ms_ssim": 4,
    "psnr": 2,
    "mse": 6,
}
BASE_HEADERS: Sequence[str] = (
    "Model (Architecture)",
    "Study",
    "Compression (Linear)",
)

@dataclass
class ReportMetadata:
    model_name: str
    study_name: str
    compression: str

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ReportMetadata":
        return cls(
            model_name=str(cfg.get("model_name", "")).strip(),
            study_name=str(cfg.get("study_name", "")).strip(),
            compression=str(cfg.get("compression", "")).strip(),
        )

    def row_prefix(self, modality_label: str) -> List[str]:
        model_label = self.model_name or ""
        if modality_label:
            suffix = modality_label
            if model_label:
                model_label = f"{model_label} ({suffix})"
            else:
                model_label = suffix
        return [model_label, self.study_name, self.compression]

def _resolve_device(device_cfg: Any) -> torch.device:
    if device_cfg is None or str(device_cfg).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_cfg))


def _reconstruction_module_cfg(cfg: DictConfig) -> Any:
    recon_cfg = cfg.get("reconstruction", None)
    if recon_cfg is not None and "module" in recon_cfg:
        return recon_cfg.module

    lightning_cfg = cfg.get("lightning", None)
    if lightning_cfg is not None and "vae_lightning" in lightning_cfg:
        return lightning_cfg.vae_lightning

    raise ValueError("Expected reconstruction.module or lightning.vae_lightning in config.")


def _setup_for_inference(module: Any, *, stage: str) -> None:
    setup = getattr(module, "setup", None)
    if callable(setup):
        setup(stage=stage)
    module.eval()


def _reconstruct(module: Any, batch: Dict[str, Any]) -> torch.Tensor:
    run_inference = getattr(module, "run_inference", None)
    if callable(run_inference):
        out = run_inference(batch)
    else:
        out = module(batch["image"])

    if isinstance(out, dict) and "reconstruction" in out:
        recon = out["reconstruction"]
        return recon[-1] if isinstance(recon, (list, tuple)) else recon
    if isinstance(out, (list, tuple)):
        return out[0]
    return out

def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
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

def _format_metric(name: str, value: Any) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    if math.isinf(numeric):
        return "inf"
    precision = METRIC_PRECISION.get(name, 4)
    return f"{numeric:.{precision}f}"

def _build_table(rows: Sequence[Mapping[str, Any]], metadata: ReportMetadata) -> List[str]:
    metric_order = [name for name in METRIC_ORDER if any(name in row for row in rows)]

    table_rows: List[List[str]] = []
    for row in rows:
        prefix = metadata.row_prefix(str(row.get("modality", "")))
        data_row = list(prefix)
        for name in metric_order:
            data_row.append(_format_metric(name, row.get(name)))
        table_rows.append(data_row)

    table_headers = (*BASE_HEADERS, *[METRIC_TITLES[name] for name in metric_order])
    widths = [len(header) for header in table_headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format_line(cells: Iterable[str]) -> str:
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_format_line(table_headers), separator]
    lines.extend(_format_line(row) for row in table_rows)
    return lines

@hydra.main(config_path=CONFIG_PATH, config_name=EXP_CONFIG_NAME, version_base="1.3")
def main(cfg: DictConfig) -> None:
    
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Hydra run directory: {run_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(int(cfg.get("seed", 42)), workers=True)

    device = _resolve_device(cfg.get("device"))
    print(f"Using device: {device}")

    datamodule = instantiate(cfg.dataset.brainscape)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    if test_loader is None:
        raise RuntimeError("Datamodule did not provide a test dataloader.")
    
    module = instantiate(_reconstruction_module_cfg(cfg))
    module.to(device)
    _setup_for_inference(module, stage="test")

    metrics_cfg = cfg.metrics
    lpips_cfg = metrics_cfg.get("lpips", {})
    metric_calc = ReconstructionMetricCalculator(
        max_value=float(metrics_cfg.intensity.max_value),
        lpips_network=str(lpips_cfg.get("network", "alex")),
        lpips_is_fake_3d=bool(lpips_cfg.get("is_fake_3d", True)),
        lpips_fake_3d_ratio=float(lpips_cfg.get("fake_3d_ratio", 0.5)),
    )

    modalities_cfg = metrics_cfg.get("report_modalities")
    if not modalities_cfg:
        modalities_cfg = cfg.dataset.brainscape.get("modalities", [])
    modalities = [str(mod) for mod in modalities_cfg]
    aggregator = MetricAggregator(metric_calc.metric_names, modalities)

    metadata = ReportMetadata.from_config(cfg.get("report", {}))

    per_sample_records: List[Dict[str, Any]] = []
    total_batches = 0
    sample_index = 0

    tqdm_total = None
    if isinstance(test_loader, Sized):
        try:
            tqdm_total = len(test_loader)
        except TypeError:
            tqdm_total = None
    for batch in tqdm(
        test_loader,
        total=tqdm_total,
        desc="Evaluating reconstructions",
        unit="batch",
    ):
        total_batches += 1
        images = batch["image"].float().to(device)
        infer_batch = dict(batch)
        infer_batch["image"] = images

        with torch.no_grad():
            reconstruction = _reconstruct(module, infer_batch)
        batch_metrics = metric_calc.compute_batch(reconstruction, images)
        modalities_batch = _string_list(batch.get("modality"))
        dataset_ids = _string_list(batch.get("dataset"))
        subjects = _string_list(batch.get("subject"))

        # Sanity check: we expect one metadata entry per sample.
        if not (
            len(batch_metrics) == len(modalities_batch) ==
            len(dataset_ids) == len(subjects)
        ):
            raise ValueError(
                "Inconsistent batch metadata lengths: "
                f"metrics={len(batch_metrics)}, "
                f"modalities={len(modalities_batch)}, "
                f"datasets={len(dataset_ids)}, "
                f"subjects={len(subjects)}"
            )

        for idx, metrics in enumerate(batch_metrics):
            modality = modalities_batch[idx]
            dataset_id = dataset_ids[idx]
            subject = subjects[idx]

            aggregator.add_sample(modality, metrics)

            record: Dict[str, Any] = {
                "sample_index": sample_index,
                "dataset": dataset_id,
                "subject": subject,
                "modality": modality,
            }
            for name in metric_calc.metric_names:
                record[name] = metrics.get(name, float("nan"))
            per_sample_records.append(record)
            sample_index += 1

    print(f"Processed {sample_index} volumes across {total_batches} batches.")

    summary_rows = aggregator.summary_rows()
    table_lines = _build_table(summary_rows, metadata)
    print("\n".join(table_lines))

    output_cfg = cfg.metrics.output
    summary_path = run_dir / output_cfg.summary_csv
    sample_path = run_dir / output_cfg.sample_csv
    table_path = run_dir / output_cfg.table_txt

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    summary_fields = ["modality", "count", *metric_calc.metric_names]
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in summary_fields})

    sample_fields = ["sample_index", "dataset", "subject", "modality", *metric_calc.metric_names]
    with sample_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=sample_fields)
        writer.writeheader()
        for record in per_sample_records:
            writer.writerow(record)

    with table_path.open("w") as fp:
        fp.write("\n".join(table_lines) + "\n")

    print(f"Summary metrics saved to: {summary_path}")
    print(f"Per-sample metrics saved to: {sample_path}")
    print(f"Formatted table written to: {table_path}")

if __name__ == "__main__":
    main()
