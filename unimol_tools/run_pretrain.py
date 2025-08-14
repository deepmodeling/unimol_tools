import os
import random
import logging
import shutil

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unimol_tools.pretrain import (
    LMDBDataset,
    UniMolDataset,
    UniMolLoss,
    UniMolModel,
    UniMolPretrainTrainer,
    build_dictionary,
    preprocess_dataset,
    compute_lmdb_dist_stats,
)

logger = logging.getLogger(__name__)


class MolPretrain:
    def __init__(self, cfg: DictConfig):
        # Read configuration
        self.config = cfg
        self.local_rank = getattr(self.config.training, 'local_rank', 0)
        seed = getattr(self.config.training, 'seed', 42)
        self.set_seed(seed)
        
        # Preprocess dataset if necessary
        ds_cfg = self.config.dataset
        train_lmdb = ds_cfg.train_path
        val_lmdb = ds_cfg.valid_path
        self.dist_mean = None
        self.dist_std = None
        if ds_cfg.data_type != 'lmdb' and not ds_cfg.train_path.endswith('.lmdb'):
            lmdb_path = os.path.splitext(ds_cfg.train_path)[0] + '.lmdb'
            logger.info(
                f"Preprocessing training data from {ds_cfg.train_path} to {lmdb_path}"
            )
            lmdb_path, self.dist_mean, self.dist_std = preprocess_dataset(
                ds_cfg.train_path,
                lmdb_path,
                data_type=ds_cfg.data_type,
                smiles_col=ds_cfg.smiles_column,
                num_conf=ds_cfg.num_conformers if ds_cfg.add_2d else 1,
                add_2d=ds_cfg.add_2d,
                remove_hs=ds_cfg.remove_hydrogen,
            )
            train_lmdb = lmdb_path
            logger.info(
                f"Dataset preprocessing finished, LMDB saved at {lmdb_path}"
            )

            if ds_cfg.valid_path:
                val_lmdb = os.path.splitext(ds_cfg.valid_path)[0] + '.lmdb'
                logger.info(
                    f"Preprocessing validation data from {ds_cfg.valid_path} to {val_lmdb}"
                )
                preprocess_dataset(
                    ds_cfg.valid_path,
                    val_lmdb,
                    data_type=ds_cfg.data_type,
                    smiles_col=ds_cfg.smiles_column,
                    num_conf=ds_cfg.num_conformers if ds_cfg.add_2d else 1,
                    add_2d=ds_cfg.add_2d,
                    remove_hs=ds_cfg.remove_hydrogen,
                )
                logger.info(
                    f"Validation dataset preprocessing finished, LMDB saved at {val_lmdb}"
                )
        else:
            if train_lmdb:
                self.dist_mean, self.dist_std = compute_lmdb_dist_stats(train_lmdb)

        # Build dictionary
        dict_path = ds_cfg.get('dict_path', None)
        if dict_path:
            from unimol_tools.data.dictionary import Dictionary
            self.dictionary = Dictionary.load(dict_path)
            self.dict_path = dict_path
            logger.info(f"Loaded dictionary from {dict_path}")
        else:
            self.dictionary = build_dictionary(train_lmdb)
            self.dict_path = os.path.join(os.path.dirname(train_lmdb), 'dictionary.txt')
            logger.info("Built dictionary from training LMDB")

        # Build dataset
        logger.info(f"Loading LMDB dataset from {train_lmdb}")
        lmdb_dataset = LMDBDataset(train_lmdb)
        self.dataset = UniMolDataset(
            lmdb_dataset,
            self.dictionary,
            remove_hs=ds_cfg.remove_hydrogen,
            max_atoms=ds_cfg.max_atoms,
            seed=seed,
            noise_type=ds_cfg.noise_type,
            noise=ds_cfg.noise,
            mask_prob=ds_cfg.mask_prob,
            leave_unmasked_prob=ds_cfg.leave_unmasked_prob,
            random_token_prob=ds_cfg.random_token_prob,
            sample_conformer=ds_cfg.add_2d,
        )

        if val_lmdb:
            logger.info(f"Loading validation LMDB dataset from {val_lmdb}")
            val_lmdb_dataset = LMDBDataset(val_lmdb)
            self.valid_dataset = UniMolDataset(
                val_lmdb_dataset,
                self.dictionary,
                remove_hs=ds_cfg.remove_hydrogen,
                max_atoms=ds_cfg.max_atoms,
                seed=seed,
                noise_type=ds_cfg.noise_type,
                noise=ds_cfg.noise,
                mask_prob=ds_cfg.mask_prob,
                leave_unmasked_prob=ds_cfg.leave_unmasked_prob,
                random_token_prob=ds_cfg.random_token_prob,
                sample_conformer=ds_cfg.add_2d,
            )
        else:
            self.valid_dataset = None

    def pretrain(self):
        # Build model
        model = UniMolModel(self.config.model, dictionary=self.dictionary)
        # Build loss function
        loss_fn = UniMolLoss(
            self.dictionary,
            masked_token_loss=self.config.model.masked_token_loss,
            masked_coord_loss=self.config.model.masked_coord_loss,
            masked_dist_loss=self.config.model.masked_dist_loss,
            x_norm_loss=self.config.model.x_norm_loss,
            delta_pair_repr_norm_loss=self.config.model.delta_pair_repr_norm_loss,
            dist_mean=self.dist_mean,
            dist_std=self.dist_std,
        )
        # Build trainer
        trainer = UniMolPretrainTrainer(
            model,
            self.dataset,
            loss_fn,
            self.config.training,
            local_rank=self.local_rank,
            resume=self.config.training.get('resume', None),
            valid_dataset=self.valid_dataset,
        )
        if self.local_rank == 0 and getattr(self, 'dict_path', None):
            try:
                dst_path = os.path.join(trainer.ckpt_dir, os.path.basename(self.dict_path))
                shutil.copy(self.dict_path, dst_path)
                logger.info(f"Copied dictionary file to {dst_path}")
            except Exception as e:
                logger.warning(f"Failed to copy dictionary file: {e}")
        logger.info("Starting pretraining")
        trainer.train(max_steps=self.config.training.total_steps)
        logger.info("Training finished. Checkpoints saved under the run directory.")

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