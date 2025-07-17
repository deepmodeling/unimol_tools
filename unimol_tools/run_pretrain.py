import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unimol_tools.pretrain import (LMDBDataset, UniMolDataset, UniMolloss,
                                   UniMolModel, UniMolPretrainTrainer,
                                   build_dictionary)


class MolPretrain:
    def __init__(self, cfg: DictConfig):
        # Read configuration
        self.config = cfg
        self.local_rank = getattr(self.config.training, 'local_rank', 0)
        seed = getattr(self.config.training, 'seed', 42)
        self.set_seed(seed)
        # Build dictionary
        dict_path = self.config.dataset.get('dict_path', None)
        if dict_path:
            from unimol_tools.data.dictionary import Dictionary
            self.dictionary = Dictionary.load(dict_path)
        else:
            self.dictionary = build_dictionary(self.config.dataset.train_lmdb)

        # Build dataset
        lmdb_dataset = LMDBDataset(self.config.dataset.train_lmdb)
        self.dataset = UniMolDataset(
            lmdb_dataset,
            self.dictionary,
            remove_hs=self.config.dataset.remove_hydrogen,
            max_atoms=self.config.dataset.max_atoms,
            seed=seed,
            noise_type=self.config.dataset.noise_type,
            noise=self.config.dataset.noise,
            mask_prob=self.config.dataset.mask_prob,
            leave_unmasked_prob=self.config.dataset.leave_unmasked_prob,
            random_token_prob=self.config.dataset.random_token_prob,
        )

    def pretrain(self):
        # Build model
        model = UniMolModel(self.config.model, dictionary=self.dictionary)
        # Build loss function
        loss_fn = UniMolloss(
            self.dictionary,
            masked_token_loss=self.config.model.masked_token_loss,
            masked_coord_loss=self.config.model.masked_coord_loss,
            masked_dist_loss=self.config.model.masked_dist_loss,
            x_norm_loss=self.config.model.x_norm_loss,
            delta_pair_repr_norm_loss=self.config.model.delta_pair_repr_norm_loss,
        )
        # Build trainer
        trainer = UniMolPretrainTrainer(
            model,
            self.dataset,
            loss_fn,
            self.config.training,
            local_rank=self.local_rank,
            resume=self.config.training.get('resume', None)
        )
        trainer.train(epochs=self.config.training.epochs)
        # Save model
        trainer.save('unimol_pretrain.pth')

    def set_seed(self, seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # logger.info(f"Set random seed to {seed}")

@hydra.main(version_base=None, config_path=None, config_name="pretrain_config")
def main(cfg: DictConfig):
    task = MolPretrain(cfg)
    task.pretrain()

if __name__ == "__main__":
    main()