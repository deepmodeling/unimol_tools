import os
import hydra
import numpy as np
from omegaconf import DictConfig

from ..predictor import UniMolRepr


@hydra.main(version_base=None, config_path="../config", config_name="repr_config")
def main(cfg: DictConfig):
    data_path = cfg.get("data_path")
    return_tensor = cfg.get("return_tensor", False)
    return_atomic_reprs = cfg.get("return_atomic_reprs", False)
    encoder = UniMolRepr(cfg=cfg)
    reprs = encoder.get_repr(
        data=data_path,
        return_atomic_reprs=return_atomic_reprs,
        return_tensor=return_tensor,
    )
    save_dir = cfg.get("save_path")

    if not return_tensor and not return_atomic_reprs:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "repr.npy"), reprs)
            np.savetxt(
                os.path.join(save_dir, "repr.csv"), np.asarray(reprs), delimiter="," 
            )
        else:
            print(reprs)
    else:
        print(reprs)


if __name__ == "__main__":
    main()
