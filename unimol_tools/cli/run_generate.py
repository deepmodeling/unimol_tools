import hydra
from omegaconf import DictConfig, OmegaConf

from unimol_tools.generate import MolGeneration

@hydra.main(version_base=None, config_path="../config", config_name="generate_config")
def main(cfg: DictConfig):
    # Convert DictConfig to a normal dictionary to be unpacked
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    train_path = cfg_dict.pop("train_path", None)
    if not train_path:
        raise ValueError("train_path must be specified")
        
    generator = MolGeneration(train_path=train_path, **cfg_dict)
    generator.train()

if __name__ == "__main__":
    main()
