import os
import random
import logging
import shutil
import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from unimol_tools.pretrain import (
    LMDBDataset,
    UniMolDataset,
    UniMolLoss,
    UniMolModel,
    UniMolPretrainTrainer,
    build_dictionary,
    preprocess_dataset,
    compute_lmdb_dist_stats,
    count_input_data,
    count_lmdb_entries,
)
from unimol_tools.data.dictionary import Dictionary

logger = logging.getLogger(__name__)


def load_or_compute_dist_stats(lmdb_path, rank, num_workers=1):
    """Load cached distance statistics or compute them on rank 0."""
    stats_file = os.path.join(os.path.dirname(lmdb_path), "dist_stats.npy")
    if os.path.exists(stats_file):
        dist_mean, dist_std = np.load(stats_file)
        return float(dist_mean), float(dist_std)
    if rank == 0:
        dist_mean, dist_std = compute_lmdb_dist_stats(lmdb_path, num_workers=num_workers)
        np.save(stats_file, np.array([dist_mean, dist_std], dtype=np.float32))
    else:
        while not os.path.exists(stats_file):
            time.sleep(1)
        dist_mean, dist_std = np.load(stats_file)
    return float(dist_mean), float(dist_std)


def load_or_build_dictionary(lmdb_path, rank, dict_path=None, num_workers=1):
    """Load existing dictionary file or build it on rank 0."""
    if dict_path is None:
        dict_path = os.path.join(os.path.dirname(lmdb_path), "dictionary.txt")
    if os.path.exists(dict_path):
        return Dictionary.load(dict_path), dict_path
    if rank == 0:
        dictionary = build_dictionary(lmdb_path, save_path=dict_path, num_workers=num_workers)
    else:
        while not os.path.exists(dict_path):
            time.sleep(1)
        dictionary = Dictionary.load(dict_path)
    return dictionary, dict_path


class MolPretrain:
    def __init__(self, cfg: DictConfig):
        # Read configuration
        self.config = cfg
        # Ranks are provided by torchrun
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        seed = getattr(self.config.training, 'seed', 42)
        self.set_seed(seed)
        
        # Preprocess dataset if necessary
        ds_cfg = self.config.dataset
        train_lmdb = ds_cfg.train_path
        val_lmdb = ds_cfg.valid_path
        self.dist_mean = None
        self.dist_std = None
        stats_workers = ds_cfg.get("stats_workers", 1)
        if ds_cfg.data_type != 'lmdb' and not ds_cfg.train_path.endswith('.lmdb'):
            lmdb_path = os.path.splitext(ds_cfg.train_path)[0] + '.lmdb'
            expected_cnt = count_input_data(
                ds_cfg.train_path, ds_cfg.data_type, ds_cfg.smiles_column
            )
            stats_file = os.path.join(os.path.dirname(lmdb_path), "dist_stats.npy")
            if self.rank == 0:
                regenerate = True
                if os.path.exists(lmdb_path):
                    lmdb_cnt = count_lmdb_entries(lmdb_path)
                    if lmdb_cnt == expected_cnt:
                        logger.info(
                            f"Found existing training LMDB {lmdb_path} with {lmdb_cnt} molecules"
                        )
                        regenerate = False
                    else:
                        logger.warning(
                            f"Existing LMDB {lmdb_path} has {lmdb_cnt} molecules but {expected_cnt} are expected; regenerating"
                        )
                if regenerate:
                    lmdb_path, self.dist_mean, self.dist_std = preprocess_dataset(
                        ds_cfg.train_path,
                        lmdb_path,
                        data_type=ds_cfg.data_type,
                        smiles_col=ds_cfg.smiles_column,
                        num_conf=ds_cfg.num_conformers,
                        remove_hs=ds_cfg.remove_hydrogen,
                        num_workers=ds_cfg.preprocess_workers,
                    )
                    np.save(
                        stats_file,
                        np.array([self.dist_mean, self.dist_std], dtype=np.float32),
                    )
                train_lmdb = lmdb_path
            else:
                while True:
                    if os.path.exists(lmdb_path):
                        lmdb_cnt = count_lmdb_entries(lmdb_path)
                        if lmdb_cnt == expected_cnt and os.path.exists(stats_file):
                            break
                    time.sleep(1)
                train_lmdb = lmdb_path
            self.dist_mean, self.dist_std = load_or_compute_dist_stats(
                train_lmdb, self.rank, num_workers=stats_workers
            )

            if ds_cfg.valid_path:
                val_lmdb = os.path.splitext(ds_cfg.valid_path)[0] + '.lmdb'
                expected_val_cnt = count_input_data(
                    ds_cfg.valid_path, ds_cfg.data_type, ds_cfg.smiles_column
                )
                if self.rank == 0:
                    regenerate_val = True
                    if os.path.exists(val_lmdb):
                        val_cnt = count_lmdb_entries(val_lmdb)
                        if val_cnt == expected_val_cnt:
                            logger.info(
                                f"Found existing validation LMDB {val_lmdb} with {val_cnt} molecules"
                            )
                            regenerate_val = False
                        else:
                            logger.warning(
                                f"Existing validation LMDB {val_lmdb} has {val_cnt} molecules but {expected_val_cnt} are expected; regenerating"
                            )
                    if regenerate_val:
                        preprocess_dataset(
                            ds_cfg.valid_path,
                            val_lmdb,
                            data_type=ds_cfg.data_type,
                            smiles_col=ds_cfg.smiles_column,
                            num_conf=ds_cfg.num_conformers,
                            remove_hs=ds_cfg.remove_hydrogen,
                            num_workers=ds_cfg.preprocess_workers,
                        )
                else:
                    while True:
                        if os.path.exists(val_lmdb):
                            val_cnt = count_lmdb_entries(val_lmdb)
                            if val_cnt == expected_val_cnt:
                                break
                        time.sleep(1)
        else:
            if train_lmdb:
                self.dist_mean, self.dist_std = load_or_compute_dist_stats(
                    train_lmdb, self.rank, num_workers=stats_workers
                )

        # Build dictionary
        dict_path = ds_cfg.get('dict_path', None)
        if dict_path:
            self.dictionary = Dictionary.load(dict_path)
            self.dict_path = dict_path
            logger.info(f"Loaded dictionary from {dict_path}")
        else:
            self.dictionary, self.dict_path = load_or_build_dictionary(
                train_lmdb, self.rank, num_workers=stats_workers
            )
            if self.rank == 0:
                logger.info("Built dictionary from training LMDB")
            else:
                logger.info(f"Loaded dictionary from {self.dict_path}")

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
            sample_conformer=True,
            add_2d=ds_cfg.add_2d,
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
                sample_conformer=True,
                add_2d=ds_cfg.add_2d,
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
        if self.rank == 0 and getattr(self, 'dict_path', None):
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

@hydra.main(version_base=None, config_path=None, config_name="pretrain_config")
def main(cfg: DictConfig):
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        logging.disable(logging.WARNING)
    try:
        task = MolPretrain(cfg)
        task.pretrain()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()