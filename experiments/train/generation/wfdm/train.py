import hydra
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything

REPO_ROOT   = Path(__file__).resolve().parents[4]      # Main Repo
CONFIG_PATH = str(REPO_ROOT / "configs")               # Hydra Configs

torch.set_float32_matmul_precision("medium")

@hydra.main(config_path=CONFIG_PATH, config_name="exp/train/generation/wfdm/rflow_l2", version_base="1.3")
def train(cfg: DictConfig) -> None:

    run_dir = HydraConfig.get().runtime.output_dir  # absolute path
    print(f"Run directory : {run_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.dataset.brainscape)  # DataModule (BrainScape)
    module = instantiate(cfg.lightning.diffusion_lightning)  # LightningModule

    # Callbacks / loggers
    callbacks = [instantiate(c) for c in cfg.callbacks.diff_callbacks]
    loggers   = [instantiate(lg) for lg in cfg.get("loggers", [])]

    # Trainer
    trainer = Trainer(
        **cfg.trainer.pl_trainer,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=run_dir,
        fast_dev_run=cfg.get("fast_dev_run", False),
    )

    trainer.fit(module, datamodule=datamodule)

    if cfg.get("test_after_training", False):
        trainer.test(module, datamodule=datamodule)

if __name__ == "__main__":
    train()
