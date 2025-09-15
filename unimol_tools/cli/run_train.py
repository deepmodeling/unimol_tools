import hydra
from omegaconf import DictConfig

from ..train import MolTrain


@hydra.main(version_base=None, config_path="../config", config_name="train_config")
def main(cfg: DictConfig):
    data_path = cfg.get("train_path")
    if not data_path:
        raise ValueError("train_path must be specified")
    trainer = MolTrain(cfg=cfg)
    trainer.fit(data=data_path)


if __name__ == "__main__":
    main()
