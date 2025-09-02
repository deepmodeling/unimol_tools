import os
import hydra
import numpy as np
from omegaconf import DictConfig

from ..predictor import UniMolRepr


@hydra.main(version_base=None, config_path="../config", config_name="repr_config")
def main(cfg: DictConfig):
    data_path = cfg.get("data_path")
    return_atomic_reprs = cfg.get("return_atomic_reprs", False)
    return_tensor = cfg.get("return_tensor", False)
    encoder = UniMolRepr(cfg=cfg)
    reprs = encoder.get_repr(data=data_path, return_atomic_reprs=return_atomic_reprs, return_tensor=return_tensor)
    repr_path = cfg.get("repr_path")
    save_dir = cfg.get("save_path")
    if repr_path is None and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        repr_path = os.path.join(save_dir, "repr.npy")
    if repr_path:
        np.save(repr_path, reprs)
    else:
        print(reprs)


if __name__ == "__main__":
    main()
