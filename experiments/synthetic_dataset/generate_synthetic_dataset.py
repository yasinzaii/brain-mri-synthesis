from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Sequence, Tuple

import hydra
import nibabel as nib
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

from brainmint.inference.core.runner import ContextPipelineRunner

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = str(REPO_ROOT / "configs")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.utils.batch import infer_batch_size, to_list
from experiments.utils.brainscape_manifest import build_brainscape_index
from experiments.utils.paths import add_postfix, ensure_clean_dir, relpath_from_abs
from experiments.utils.runtime import resolve_device, sync_if_cuda

# WFDM RFlow-L2 is the release default.
EXP_CONFIG_NAME = "exp/synthetic_dataset/generation/wfdm/rflow_l2"


def _instantiate_inference_runner(cfg: DictConfig, *, device: torch.device) -> Any:
    runner = str(cfg.synthetic_generation.get("runner", "lightning_module"))

    if runner == "lightning_module":
        module = hydra.utils.instantiate(cfg.lightning.diffusion_inference)
        if hasattr(module, "setup"):
            module.setup(stage="test")  # loads weights via StateDictLoaderMixin

    elif runner == "direct_context":
        scalars_cfg = cfg.generation.get("scalars", {})
        scalars = (
            OmegaConf.to_container(scalars_cfg, resolve=True)
            if OmegaConf.is_config(scalars_cfg)
            else dict(scalars_cfg)
        )
        module = ContextPipelineRunner(
            pipeline=hydra.utils.instantiate(cfg.generation.pipeline),
            modules={
                str(name): hydra.utils.instantiate(module_cfg)
                for name, module_cfg in cfg.generation.get("modules", {}).items()
            },
            scalars=dict(scalars or {}),
            auto_load_wrapper_weights=bool(cfg.synthetic_generation.get("auto_load_wrapper_weights", False)),
        )

    elif runner == "context_pipeline":
        module = hydra.utils.instantiate(cfg.generation.runner)

    else:
        raise ValueError(
            "Unsupported synthetic_generation.runner "
            f"{runner!r}. Expected one of: lightning_module, direct_context, context_pipeline."
        )

    module.to(device)
    module.eval()
    return module


def _compute_metrics_summary(
    *,
    latencies_s: Sequence[float],
    peak_mem_bytes: Sequence[int],
    output_tensor_size: Tuple[int, int, int],
    model_name: str,
    modality: str | None,
    steps: int | None,
    warmup_runs: int,
    timed_runs: int,
    device: torch.device,
) -> Dict[str, Any]:
    if len(latencies_s) == 0:
        raise ValueError("No latency measurements were recorded in metrics mode.")

    std_seconds = statistics.stdev(latencies_s) if len(latencies_s) > 1 else 0.0
    peak_max = max(peak_mem_bytes) if peak_mem_bytes else 0
    peak_mean = statistics.mean(peak_mem_bytes) if peak_mem_bytes else 0.0
    gib = float(1024**3)
    return {
        "model_name": model_name,
        "benchmark_modality": modality,
        "steps": steps,
        "batch_size": 1,
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "device": str(device),
        "output_tensor_size": list(output_tensor_size),
        "latency_seconds": {
            "mean": float(statistics.mean(latencies_s)),
            "median": float(statistics.median(latencies_s)),
            "std": float(std_seconds),
            "min": float(min(latencies_s)),
            "max": float(max(latencies_s)),
        },
        "peak_gpu_memory_bytes": {
            "max": int(peak_max),
            "mean": float(peak_mean),
        },
        "peak_gpu_memory_gb": {
            "max": float(peak_max / gib),
            "mean": float(peak_mean / gib),
        },
    }


def _run_metrics_mode(
    *,
    test_loader: Any,
    inference_module: Any,
    device: torch.device,
    path_key: str,
    modality_key: str,
    expected_out_size: Any,
    metrics_cfg: DictConfig,
    cfg: DictConfig,
) -> Dict[str, Any]:
    warmup_runs = int(metrics_cfg.warmup_runs)
    timed_runs = int(metrics_cfg.target_samples)
    reset_peak_each_run = bool(metrics_cfg.reset_peak_memory_each_run)

    if warmup_runs < 0 or timed_runs <= 0:
        raise ValueError("metrics.warmup_runs must be >= 0 and metrics.target_samples must be > 0.")

    total_runs = warmup_runs + timed_runs
    latencies_s: List[float] = []
    peak_mem_bytes: List[int] = []
    output_tensor_size: Tuple[int, int, int] | None = None
    benchmark_modality: str | None = None

    pbar = tqdm(test_loader, desc="Metrics benchmark", dynamic_ncols=True)
    n_seen = 0
    for batch in pbar:
        if n_seen >= total_runs:
            break

        bsz = infer_batch_size(batch.get(path_key), key=path_key)
        if bsz != 1:
            raise ValueError(
                "metrics mode requires test batch size = 1. "
                f"Got batch size {bsz} for key '{path_key}'."
            )

        if modality_key in batch:
            modality_values = to_list(batch.get(modality_key), bsz)
            batch_modality = str(modality_values[0])
            if benchmark_modality is None:
                benchmark_modality = batch_modality
            elif benchmark_modality != batch_modality:
                continue
                # raise ValueError(
                #     f"Inconsistent modalities in metrics run. First={benchmark_modality}, got={batch_modality}"
                # )

        if device.type == "cuda" and torch.cuda.is_available() and reset_peak_each_run and n_seen >= warmup_runs:
            torch.cuda.reset_peak_memory_stats(device)

        sync_if_cuda(device)
        t0 = time.perf_counter()
        out = inference_module.run(batch, device=device)
        if "sample" not in out:
            raise KeyError("Inference output missing key 'sample'")
        sample = out["sample"]
        sync_if_cuda(device)
        elapsed = time.perf_counter() - t0

        got_shape = tuple(int(x) for x in sample.shape[-3:])
        if expected_out_size is not None:
            exp = tuple(int(x) for x in expected_out_size)
            if got_shape != exp:
                raise ValueError(f"Unexpected output size {got_shape}, expected {exp}")

        if output_tensor_size is None:
            output_tensor_size = got_shape
        elif got_shape != output_tensor_size:
            raise ValueError(
                f"Inconsistent output shape across runs. First={output_tensor_size}, got={got_shape}"
            )

        if n_seen >= warmup_runs:
            latencies_s.append(float(elapsed))
            if device.type == "cuda" and torch.cuda.is_available():
                peak_mem_bytes.append(int(torch.cuda.max_memory_allocated(device)))
            else:
                peak_mem_bytes.append(0)
            run_idx = len(latencies_s) - 1
            peak_gb = float(peak_mem_bytes[-1]) / float(1024**3)
            log.info(
                "metrics_run=%d latency_s=%.6f peak_gpu_mem_bytes=%d peak_gpu_mem_gb=%.4f output_tensor_size=%s",
                run_idx,
                latencies_s[-1],
                peak_mem_bytes[-1],
                peak_gb,
                got_shape,
            )
            pbar.set_postfix({"timed_runs": len(latencies_s), "target": timed_runs})
        else:
            pbar.set_postfix({"warmup": n_seen + 1, "target": warmup_runs})
        n_seen += 1

        if len(latencies_s) >= timed_runs:
            break

    if len(latencies_s) < timed_runs:
        raise RuntimeError(
            f"Insufficient samples for metrics mode: requested timed_runs={timed_runs}, got={len(latencies_s)}"
        )

    summary = _compute_metrics_summary(
        latencies_s=latencies_s,
        peak_mem_bytes=peak_mem_bytes,
        output_tensor_size=output_tensor_size if output_tensor_size is not None else (0, 0, 0),
        model_name=str(cfg.get("project", "unknown_model")),
        modality=benchmark_modality,
        steps=int(cfg.get("inference_steps")) if cfg.get("inference_steps", None) is not None else None,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        device=device,
    )
    return summary


@hydra.main(config_path=CONFIG_PATH, config_name=EXP_CONFIG_NAME, version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info("Hydra run dir: %s", run_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    seed_everything(int(cfg.get("seed", 0)), workers=True)
    device = resolve_device(cfg.get("device", "auto"))
    log.info("Using device: %s", device)

    # I/O settings
    gen_cfg = cfg.synthetic_generation
    out_root = Path(gen_cfg.out_dataset_root)
    out_json = Path(gen_cfg.out_json)

    real_root = Path(cfg.dataset.brainscape.dataset_root)
    real_json = Path(cfg.dataset.brainscape.json_path)
    split = str(gen_cfg.split)

    match_group = str(gen_cfg.get("match_group", "preprocessed"))
    out_group = str(gen_cfg.get("out_group", "preprocessed"))
    match_group_dir = str(gen_cfg.get("match_group_dir", match_group))
    out_group_dir = str(gen_cfg.get("out_group_dir", out_group))
    postfix = str(gen_cfg.sample_postfix)

    # Batch keys
    path_key = str(gen_cfg.get("path_key", "image_path"))
    modality_key = str(gen_cfg.get("modality_key", "modality"))
    metrics_mode = bool(gen_cfg.get("compute_metrics", False))
    metrics_cfg = gen_cfg.metrics

    # Strict record matching keys: meta_keys + modality
    brainscape_match_keys = list(gen_cfg.brainscape_match_keys)
    match_keys = brainscape_match_keys + [modality_key]

    overwrite = bool(gen_cfg.get("overwrite", False))
    resume = bool(gen_cfg.get("resume", False))
    max_items = gen_cfg.get("max_items", None)
    max_items_int = int(max_items) if max_items is not None else None

    expected_out_size = gen_cfg.get("sample_out_size", None)

    # Data
    datamodule = hydra.utils.instantiate(cfg.dataset.brainscape)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # Pipeline-based inference. Defaults to the existing Lightning shell until
    # individual synthetic-generation configs opt into direct_context/context_pipeline.
    inference_module = _instantiate_inference_runner(cfg, device=device)

    if metrics_mode:
        log.info("Running in synthetic_generation.compute_metrics mode.")
        summary = _run_metrics_mode(
            test_loader=test_loader,
            inference_module=inference_module,
            device=device,
            path_key=path_key,
            modality_key=modality_key,
            expected_out_size=expected_out_size,
            metrics_cfg=metrics_cfg,
            cfg=cfg,
        )
        log.info(
            "METRICS SUMMARY | model=%s modality=%s mean=%.6fs median=%.6fs std=%.6fs "
            "peak_mem_gb_max=%.4f output_size=%s",
            summary["model_name"],
            summary["benchmark_modality"],
            summary["latency_seconds"]["mean"],
            summary["latency_seconds"]["median"],
            summary["latency_seconds"]["std"],
            summary["peak_gpu_memory_gb"]["max"],
            tuple(summary["output_tensor_size"]),
        )
        return

    if resume and overwrite:
        raise ValueError("synthetic_generation.resume=true is incompatible with overwrite=true.")

    if out_json.exists() and not (overwrite or resume):
        raise FileExistsError(
            f"Refusing to overwrite existing json: {out_json} "
            f"(set overwrite=true or synthetic_generation.resume=true)"
        )
    if out_root.resolve() == real_root.resolve():
        raise ValueError("out_dataset_root must be different from real_root")

    if resume:
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        ensure_clean_dir(out_root, overwrite=overwrite)

    # Load source JSON from the real datalist every run.
    source = json.loads(real_json.read_text())
    if split not in source:
        raise KeyError(f"Split '{split}' missing from real_json: {real_json}")

    records: List[MutableMapping[str, Any]] = list(source[split])
    index = build_brainscape_index(
        records, match_group=match_group, match_keys=match_keys, modality_key=modality_key
    )

    # Generate
    n_written = 0
    n_skipped_existing = 0
    pbar = tqdm(test_loader, desc="Generating", dynamic_ncols=True)

    for batch in pbar:
        bsz = infer_batch_size(batch.get(path_key), key=path_key)

        # Collect per-item metadata needed for strict key matching.
        meta: Dict[str, List[Any]] = {}
        for k in match_keys:
            if k not in batch:
                raise KeyError(f"Batch missing required meta key '{k}'.")
            meta[k] = to_list(batch.get(k), bsz)

        paths = to_list(batch.get(path_key), bsz)
        if len(paths) != bsz:
            raise ValueError(f"{path_key} length {len(paths)} != batch size {bsz}")

        batch_items: List[Dict[str, Any]] = []
        for i in range(bsz):
            # Build strict key tuple for this item.
            key_tuple = tuple(str(meta[k][i]) for k in match_keys)
            dataset_id = str(meta.get("dataset", [""] * bsz)[i])
            modality = str(meta[modality_key][i])

            abs_in = paths[i]
            if abs_in is None:
                raise KeyError(f"Batch missing '{path_key}' (needed for JSON mapping/writing)")

            real_group_base = real_root / dataset_id / match_group_dir
            rel_in = relpath_from_abs(abs_in, root=real_group_base)

            # Record lookup
            if key_tuple not in index:
                raise KeyError(
                    f"No matching record for key={key_tuple}. "
                    f"(match_keys={match_keys}, match_group={match_group})"
                )

            rec_idx, rel_from_json = index[key_tuple]
            if str(rel_in) != rel_from_json:
                raise ValueError(
                    "Path mismatch between batch and JSON record for key="
                    f"{key_tuple}.\n"
                    f"  batch rel: {rel_in}\n"
                    f"  json  rel: {rel_from_json}"
                )

            out_rel = add_postfix(rel_in, postfix)
            out_path = out_root / dataset_id / out_group_dir / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            batch_items.append(
                {
                    "i": i,
                    "rec_idx": rec_idx,
                    "modality": modality,
                    "out_rel": out_rel,
                    "out_path": out_path,
                }
            )

        all_exist = all(item["out_path"].exists() for item in batch_items)
        if resume and all_exist:
            for item in batch_items:
                rec = records[item["rec_idx"]]
                syn = rec.setdefault("synthetic", {})
                syn[str(item["modality"])] = str(item["out_rel"])
            n_skipped_existing += len(batch_items)
            pbar.set_postfix({"written": n_written, "skipped_existing": n_skipped_existing})
            continue

        out = inference_module.run(batch, device=device)
        if "sample" not in out:
            raise KeyError("Inference output missing key 'sample'")

        sample = out["sample"].detach().float().cpu()  # (B, C, Z, Y, X) or (B,1,...)
        if int(sample.shape[0]) != bsz:
            raise ValueError(f"Sample batch size {int(sample.shape[0])} != input batch size {bsz}")

        if expected_out_size is not None:
            exp = tuple(int(x) for x in expected_out_size)
            got = tuple(int(x) for x in sample.shape[-3:])
            if got != exp:
                raise ValueError(f"Unexpected output size {got}, expected {exp}")

        for item in batch_items:
            i = int(item["i"])
            out_path = item["out_path"]

            if out_path.exists() and not overwrite:
                if resume:
                    n_skipped_existing += 1
                else:
                    raise ValueError(
                        f"Output file already exists, and overwrite=false, file:{str(out_path)}"
                    )
            else:
                affine = np.eye(4, dtype=np.float32)
                vol = sample[i]
                if vol.ndim == 4:
                    if vol.shape[0] != 1:
                        raise ValueError("Only sigle channel output supported for NOW!")
                    vol = vol[0]
                vol_np = np.asarray(vol, dtype=np.float32)
                img = nib.Nifti1Image(vol_np, affine=affine)
                nib.save(img, str(out_path))
                n_written += 1

            # Update JSON: store path relative to dataset_id root (consistent with match_group paths).
            rec = records[item["rec_idx"]]
            syn = rec.setdefault("synthetic", {})
            syn[str(item["modality"])] = str(item["out_rel"])

            if max_items_int is not None and n_written >= max_items_int:
                break

        if max_items_int is not None and n_written >= max_items_int:
            break

        pbar.set_postfix({"written": n_written, "skipped_existing": n_skipped_existing})

    # --- Save updated JSON ---
    source[split] = records
    summary = {
        "project": str(cfg.get("project", "")),
        "split": split,
        "real_json": str(real_json),
        "real_root": str(real_root),
        "synthetic_root": str(out_root),
        "synthetic_json": str(out_json),
        "match_group": match_group,
        "out_group": out_group,
        "match_group_dir": match_group_dir,
        "out_group_dir": out_group_dir,
        "postfix": postfix,
        "path_key": path_key,
        "match_keys": match_keys,
        "written": n_written,
        "skipped_existing": n_skipped_existing,
        "resume": resume,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(source, indent=2))
    log.info("Wrote %d synthetic volumes", n_written)
    log.info("Wrote synthetic datalist JSON: %s", out_json)
    log.info("Synthetic dataset root: %s", out_root)

    summary_path = run_dir / "synthetic_generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Run summary: %s", summary_path)


if __name__ == "__main__":
    main()
