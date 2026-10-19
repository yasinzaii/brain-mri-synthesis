import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import hydra
import numpy as np
import torch

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from brainmint.inference.io.writers import WriterBase, VolumeWriter


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = str(REPO_ROOT / "configs")


CONFIG_NAME = "exp/inference/generation/wfdm/rflow_l2"


def _resolve_device(device_cfg: str) -> torch.device:
    d = str(device_cfg).lower()
    if d in ("auto", "cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if d in ("cpu",):
        return torch.device("cpu")
    return torch.device(device_cfg)


def _ensure_list_modalities(modalities: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(modalities, str):
        return [modalities]
    return list(modalities)


def _resolve_images_per_modality(images_per_modality: Any, modality: str) -> int:
    if isinstance(images_per_modality, int):
        return int(images_per_modality)
    if isinstance(images_per_modality, Mapping):
        if modality in images_per_modality:
            return int(images_per_modality[modality])
        m2 = str(modality).lower()
        if m2 in images_per_modality:
            return int(images_per_modality[m2])
    raise ValueError(
        "inference.images_per_modality must be an int or a mapping {modality: int}; "
        f"got {type(images_per_modality)}"
    )


def _expand_single_sample_to_batch(sample: Dict[str, Any], *, batch_size: int) -> Dict[str, Any]:
    """Broadcast single-sample conditioning tensors to (B, ...) for inference."""
    out: Dict[str, Any] = dict(sample)
    out["batch_size"] = int(batch_size)

    if "modality_map" in out and torch.is_tensor(out["modality_map"]):
        lab = out["modality_map"].view(-1)[0].long()
        out["modality_map"] = lab.repeat(batch_size)

    if "demo_values" in out and torch.is_tensor(out["demo_values"]):
        v = out["demo_values"].view(1, -1)
        out["demo_values"] = v.repeat(batch_size, 1)

    if "demo_missing" in out and torch.is_tensor(out["demo_missing"]):
        m = out["demo_missing"].view(1, -1)
        out["demo_missing"] = m.repeat(batch_size, 1)

    return out


def _instantiate_inference_runner(cfg: DictConfig, *, device: torch.device) -> Any:
    runner = str(cfg.inference.get("runner", "lightning_module"))
    if runner == "context_pipeline":
        module = instantiate(cfg.generation.runner)
        if not callable(getattr(module, "run", None)):
            raise TypeError(f"generation.runner must instantiate an object with run(...), got {type(module)!r}")
    elif runner == "lightning_module":
        module = instantiate(cfg.lightning.diffusion_inference)
        module.setup("predict")
    else:
        raise ValueError(f"Unsupported inference.runner={runner!r}")

    module.to(device)
    module.eval()
    return module


@hydra.main(
    config_path=CONFIG_PATH,
    config_name=CONFIG_NAME,
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Run directory: {run_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(int(cfg.seed), workers=True)
    device = _resolve_device(str(cfg.device))

    modality_tf = instantiate(cfg.dataset.conditioning.modality_conditioning)
    demographics_tf = instantiate(cfg.dataset.conditioning.demographics_conditioning)

    infer_mod = _instantiate_inference_runner(cfg, device=device)

    out_root = run_dir / str(cfg.inference.output_dirname)
    out_root.mkdir(parents=True, exist_ok=True)

    writer_cfg = cfg.inference.get("output", {}).get("writer", None)
    writer: WriterBase = instantiate(writer_cfg) if writer_cfg is not None else VolumeWriter()

    filename_prefix = str(cfg.inference.get("filename_prefix", "sample"))
    output_ext = str(cfg.inference.get("output_ext", ".nii.gz"))
    if not output_ext.startswith("."):
        output_ext = f".{output_ext}"

    modalities = [str(m).lower() for m in _ensure_list_modalities(cfg.inference.modalities)]
    demo = dict(cfg.inference.demographics or {})

    ipm_cfg = cfg.inference.images_per_modality
    ipm_for_manifest = OmegaConf.to_container(ipm_cfg, resolve=True) if OmegaConf.is_config(ipm_cfg) else ipm_cfg

    manifest: Dict[str, Any] = {
        "modalities": modalities,
        "images_per_modality": ipm_for_manifest,
        "demographics": demo,
        "outputs": [],
    }

    for modality in modalities:
        n = _resolve_images_per_modality(cfg.inference.images_per_modality, modality)
        if n <= 0:
            raise ValueError(f"images_per_modality for modality={modality} must be > 0, got {n}")

        sample: Dict[str, Any] = {"modality": modality, "demographics": demo}
        sample = modality_tf(sample)
        sample = demographics_tf(sample)
        batch = _expand_single_sample_to_batch(sample, batch_size=n)

        out = infer_mod.run(batch, device=device)
        if "sample" not in out:
            raise KeyError(f"Pipeline output missing 'sample'. Keys: {sorted(out.keys())}")

        x = out["sample"]
        if not torch.is_tensor(x) or x.ndim != 5:
            raise ValueError(f"Expected 'sample' to be 5D (B,C,Z,Y,X), got {type(x)} {getattr(x,'shape',None)}")

        x_np = x.detach().cpu().numpy()

        mod_dir = out_root / modality
        mod_dir.mkdir(parents=True, exist_ok=True)

        for i in range(x_np.shape[0]):
            vol = x_np[i, 0]
            out_path = mod_dir / f"{filename_prefix}_{i:04d}{output_ext}"
            saved_path = writer.write(vol, out_path)
            manifest["outputs"].append(
                {"modality": modality, "index": int(i), "path": str(saved_path.relative_to(run_dir))}
            )

        if bool(cfg.inference.get("save_latents", False)) and "latent" in out and torch.is_tensor(out["latent"]):
            z = out["latent"].detach().cpu().numpy()  # (B, C, Z, Y, X)

            for i in range(z.shape[0]):
                out_path = mod_dir / f"latent_{i:04d}.nii.gz"
                writer.write_4d(z[i], out_path, channels_first=True)

    with (out_root / "manifest.json").open("w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"Wrote samples to: {out_root}")


if __name__ == "__main__":
    main()
