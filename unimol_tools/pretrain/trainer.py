import os
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from hydra.core.hydra_config import HydraConfig

from .pretrain_config import PretrainConfig

logger = logging.getLogger(__name__)

class UniMolPretrainTrainer:
    def __init__(self, model, dataset, loss_fn, config, local_rank=0, resume: str=None):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.config = config
        self.local_rank = local_rank

        run_dir = HydraConfig.get().run.dir
        self.ckpt_dir = Path(os.path.join(run_dir, 'checkpoints'))
        self.writer = SummaryWriter(log_dir=run_dir) if local_rank == 0 else None
        if local_rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            logger.info(f"Checkpoints will be saved to {self.ckpt_dir}")

        # resume training from a checkpoint if provided
        self.start_epoch = 0
        self.global_step = 0
        if resume is not None and os.path.isfile(resume):
            self._load_checkpoint(resume)

        # DDP setup
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
            self.model = self.model.to(local_rank)
            self.model = DDP(self.model, device_ids=[local_rank])
        else:
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr if hasattr(self.config, 'lr') else 1e-4)
        self.criterion = nn.CrossEntropyLoss()


        # DDP DataLoader
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            self.sampler = None
        logger.info(f"Using sampler: {self.sampler}")

        g = torch.Generator()
        g.manual_seed(config.seed)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.model.batch_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def train(self, epochs=None):
        epochs = epochs or self.config.epochs
        save_every = getattr(self.config, "save_every_n_epoch", 5)

        logging_infos = []
        for epoch in range(self.start_epoch, epochs):
            if self.sampler:
                self.sampler.set_epoch(epoch)
            self.model.train()

            for i, batch in enumerate(self.dataloader):
                net_input, net_target = self.decorate_batch(batch)
                loss, logging_info = self.loss_fn(self.model, net_input, net_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                logging_infos.append(logging_info)

                if self.writer:
                    self.writer.add_scalar('Step/loss', loss.item(), epoch * len(self.dataloader) + i)
                    for key, value in logging_info.items():
                        if key == 'loss':
                            continue
                        self.writer.add_scalar(f'Step/{key}', value, epoch * len(self.dataloader) + i)
            if self.local_rank == 0:
                self.reduce_metrics(logging_infos, writer=self.writer, logger=logger, epoch=epoch, split="train")

            self._save_checkpoint(epoch)
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, epoch)

        if self.writer:
            self.writer.close()

    def _save_checkpoint(self, epoch, name=None):
        if self.local_rank != 0:
            return
        ckpt = {
            "model": self.model.module.state_dict()
                     if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                     else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": self.global_step,
        }
        if name is None:
            save_name = 'latest.ckpt'
        elif isinstance(name, int):
            save_name = f'epoch_{name}.ckpt'
        else:
            save_name = name
            if not save_name.endswith('.ckpt'):
                save_name += '.ckpt'
        save_path = os.path.join(self.ckpt_dir, save_name)
        torch.save(ckpt, save_path)
        if isinstance(name, int):
            self._cleanup_old_checkpoints(keep=3)

    def _load_checkpoint(self, ckpt_path: str):
        map_loc = {"cuda:%d" % 0: "cuda:%d" % self.local_rank} if self.local_rank >= 0 else None
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        model_sd = ckpt["model"]
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(model_sd)
        else:
            self.model.load_state_dict(model_sd)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["step"]
        logger.info(f"Resume from {ckpt_path} | start_epoch={self.start_epoch} step={self.global_step}")

    def _cleanup_old_checkpoints(self, keep: int = 3):
        epoch_files = sorted(self.ckpt_dir.glob("epoch_*.ckpt"), key=lambda p: int(p.stem.split("_")[1]))
        for f in epoch_files[:-keep]:
            f.unlink()

    def decorate_batch(self, batch):
        # batch为dict of tensors (batch_size, ...)
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        net_input = {
            'src_tokens': batch['net_input']['src_tokens'].to(device),
            'src_coord': batch['net_input']['src_coord'].to(device),
            'src_distance': batch['net_input']['src_distance'].to(device),
            'src_edge_type': batch['net_input']['src_edge_type'].to(device),
        }
        net_target = {
            'tgt_tokens': batch['net_target']['tgt_tokens'].to(device),
            'tgt_coordinates': batch['net_target']['tgt_coordinates'].to(device),
            'tgt_distance': batch['net_target']['tgt_distance'].to(device),
        }
        return net_input, net_target

    @staticmethod
    def reduce_metrics(logging_outputs, writer=None, logger=None, epoch=None, split="Epoch"):
        # 聚合
        agg = {}
        for log in logging_outputs:
            for k, v in log.items():
                if k not in agg:
                    agg[k] = []
                agg[k].append(v)
        # 计算均值
        metrics_mean = {k: (sum(v)/len(v) if len(v)>0 else 0) for k, v in agg.items()}
        # 记录到writer和logger
        if writer is not None and epoch is not None:
            if 'loss' in metrics_mean:
                writer.add_scalar(f"{split}/loss", metrics_mean['loss'], epoch)
            for k, v in metrics_mean.items():
                if k == 'loss':
                    continue
                writer.add_scalar(f"{split}/{k}", v, epoch)
        if logger is not None and epoch is not None:
            # loss放在第一位，其余按原顺序
            log_items = []
            if 'loss' in metrics_mean:
                v = metrics_mean['loss']
                log_items.append(f"loss={v:.4f}")
            for k, v in metrics_mean.items():
                if k == 'loss':
                    continue
                log_items.append(f"{k}={v:.4f}")
            logger.info(f"[{split}] Epoch {epoch}: " + ", ".join(log_items))
        return metrics_mean

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)