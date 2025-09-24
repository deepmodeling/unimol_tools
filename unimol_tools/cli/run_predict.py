import hydra
from omegaconf import DictConfig

from ..predict import MolPredict


@hydra.main(version_base=None, config_path="../config", config_name="predict_config")
def main(cfg: DictConfig):
    data_path = cfg.get("data_path")
    predictor = MolPredict(cfg=cfg)
    predictor.predict(data=data_path, save_path=cfg.get("save_path"))


if __name__ == "__main__":
    main()
