from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything

torch.set_float32_matmul_precision('medium')

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = str(REPO_ROOT / "configs")

@hydra.main(config_path=CONFIG_PATH, config_name="exp/train/compression/wavelet_fusion/wf_sa_vae", version_base="1.3")
def train(cfg: DictConfig) -> None:
    runtime = HydraConfig.get().runtime
    run_dir = Path(runtime.output_dir)
    print(f"Run directory : {run_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.dataset.brainscape)
    module = instantiate(cfg.lightning.vae_lightning)

    callbacks = [instantiate(cb) for cb in cfg.callbacks.compression_callbacks]
    loggers = [instantiate(lg) for lg in cfg.get("loggers", [])]

    trainer = Trainer(
        **cfg.trainer.pl_trainer,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(run_dir),
        fast_dev_run=cfg.get("fast_dev_run", False),
    )

    trainer.fit(module, datamodule=datamodule)

    if cfg.get("test_after_training", False):
        trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    train()