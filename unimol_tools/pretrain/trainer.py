import logging
import os
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unimol_tools.tasks.trainer import get_linear_schedule_with_warmup
from unimol_tools.utils.dynamic_loss_scaler import DynamicLossScaler

logger = logging.getLogger(__name__)


class UniMolPretrainTrainer:
    def __init__(self, model, dataset, loss_fn, config, local_rank=None, resume: str=None, valid_dataset=None):
        self.model = model
        self.dataset = dataset
        self.valid_dataset = valid_dataset
        self.loss_fn = loss_fn
        self.config = config
        # Use ranks provided by torchrun
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if local_rank is None else local_rank
        self.rank = int(os.environ.get("RANK", 0))
        self.fp16 = getattr(config, "fp16", True)

        run_dir = getattr(config, "output_dir", None)
        if run_dir:
            if not os.path.isabs(run_dir):
                run_dir = os.path.join(get_original_cwd(), run_dir)
        else:
            run_dir = HydraConfig.get().run.dir
        self.ckpt_dir = Path(os.path.join(run_dir, 'checkpoints'))

        # DDP setup
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.model = self.model.to(self.local_rank)
            if self.fp16:
                self.model = self.model.half()
                if isinstance(self.loss_fn, nn.Module):
                    self.loss_fn = self.loss_fn.to(self.local_rank).half()
            else:
                if isinstance(self.loss_fn, nn.Module):
                    self.loss_fn = self.loss_fn.to(self.local_rank)
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = self.model.cuda()
            if self.fp16:
                self.model = self.model.half()
                if isinstance(self.loss_fn, nn.Module):
                    self.loss_fn = self.loss_fn.cuda().half()
            else:
                if isinstance(self.loss_fn, nn.Module):
                    self.loss_fn = self.loss_fn.cuda()

        self.writer = SummaryWriter(log_dir=run_dir) if self.rank == 0 else None
        if self.rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            logger.info(f"Checkpoints will be saved to {self.ckpt_dir}")

        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        optim_groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        self.optimizer = optim.AdamW(
            optim_groups,
            lr=self.config.lr,
            betas=getattr(self.config, "adam_betas", (0.9, 0.99)),
            eps=getattr(self.config, "adam_eps", 1e-6),
        )
        self.scheduler = None
        warmup_steps = getattr(config, "warmup_steps", 0)
        total_steps = getattr(config, "total_steps", 0)
        if warmup_steps > 0 and total_steps > 0:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, warmup_steps, total_steps
            )
        self.criterion = nn.CrossEntropyLoss()

        self.best_loss = float("inf")
        self.patience = getattr(config, "patience", -1)
        self.no_improve_steps = 0

        # DDP DataLoader
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
            self.valid_sampler = (
                torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
                if self.valid_dataset is not None
                else None
            )
        else:
            self.sampler = None
            self.valid_sampler = None
        logger.info(f"Using sampler: {self.sampler}")

        g = torch.Generator()
        g.manual_seed(config.seed)

        num_workers = getattr(self.config, "num_workers", 8)
        collate_fn = (
            self.model.module.batch_collate_fn
            if isinstance(self.model, DDP)
            else self.model.batch_collate_fn
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if self.valid_dataset is not None:
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                sampler=self.valid_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            self.valid_dataloader = None

        self.world_size = (
            dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        )
        if self.rank == 0:
            effective_bs = (
                self.config.batch_size * self.config.update_freq * self.world_size
            )
            logger.info(
                f"GPUs: {self.world_size}, batch size per GPU: {self.config.batch_size}, update_freq: {self.config.update_freq}, total batch size: {effective_bs}"
            )
            logger.info(f"Learning rate: {self.config.lr:.4e}")

        if self.fp16 and not torch.cuda.is_available():
            logger.warning("FP16 requested but CUDA is not available; disabling fp16.")
            self.fp16 = False
        self.scaler = (
            DynamicLossScaler(
                init_scale=getattr(config, "fp16_init_scale", 4),
                scale_window=getattr(config, "fp16_scale_window", 256),
            )
            if self.fp16
            else None
        )

        # resume training from a checkpoint if provided or detect last
        self.start_epoch = 0
        self.global_step = 0
        resume_path = resume
        if resume_path is not None and not os.path.isabs(resume_path):
            resume_path = os.path.join(run_dir, resume_path)
        if resume_path is None:
            last_ckpt = self.ckpt_dir / 'checkpoint_last.ckpt'
            if last_ckpt.exists():
                resume_path = str(last_ckpt)
        if resume_path is not None and os.path.isfile(resume_path):
            self._load_checkpoint(resume_path)
            logger.info(f"Resumed from checkpoint: {resume_path}")
        else:
            logger.info("No checkpoint found, starting from scratch.")

    def train(self, epochs=None, max_steps=None):
        epochs = self.config.epochs if epochs is None else epochs
        max_steps = max_steps or getattr(self.config, "total_steps", 0)
        if epochs <= 0:
            epochs = math.inf
        log_every = getattr(self.config, "log_every_n_steps", 100)
        save_every = getattr(self.config, "save_every_n_steps", 1000)

        epoch = self.start_epoch
        stop_training = False
        while epoch < epochs and not stop_training:
            display_epoch = epoch + 1
            if self.sampler:
                self.sampler.set_epoch(epoch)
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(epoch)
            self.model.train()
            logger.info(f"Starting epoch {display_epoch}")
            logger.info("Start iterating over samples")
            step_logging_infos = []
            epoch_logging_infos = []
            num_batches = len(self.dataloader)
            update_freq = getattr(self.config, "update_freq", 1)
            self.optimizer.zero_grad()
            accum_logging = []

            for i, batch in enumerate(self.dataloader, start=1):
                net_input, net_target = self.decorate_batch(batch)
                loss, sample_size, logging_info = self.loss_fn(
                    self.model, net_input, net_target
                )
                if self.fp16:
                    self.scaler.scale(loss / sample_size / update_freq).backward()
                else:
                    (loss / sample_size / update_freq).backward()
                accum_logging.append(logging_info)

                if i % update_freq == 0 or i == num_batches:
                    grad_norm = 0.0
                    if self.fp16:
                        self.scaler.unscale_(self.model.parameters())

                    clip_val = getattr(self.config, "clip_grad_norm", 0)
                    if clip_val and clip_val > 0:
                        grad_norm = clip_grad_norm_(self.model.parameters(), clip_val)
                    else:
                        for p in self.model.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.data.float().norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5

                    overflow = False
                    if self.fp16:
                        try:
                            self.scaler.check_overflow(grad_norm)
                        except OverflowError:
                            overflow = True

                    if overflow:
                        self.optimizer.zero_grad()
                        if self.rank == 0:
                            logger.warning(
                                f"gradient overflow detected, ignoring gradient, setting loss scale to: {self.scaler.loss_scale:.1f}"
                            )
                        accum_logging = []
                        continue
                    self.optimizer.step()
                    if self.fp16:
                        self.scaler.update()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    merged_log = {}
                    for log in accum_logging:
                        for k, v in log.items():
                            merged_log[k] = merged_log.get(k, 0) + v
                    step_logging_infos.append(merged_log)
                    epoch_logging_infos.append(merged_log)
                    self.global_step += 1

                    if self.writer:
                        step_metrics = self.loss_fn.reduce_metrics([merged_log])
                        self.writer.add_scalar('Step/loss', step_metrics.get('loss', 0), self.global_step)
                        for key, value in step_metrics.items():
                            if key == 'loss':
                                continue
                            self.writer.add_scalar(f'Step/{key}', value, self.global_step)
                    accum_logging = []

                    if log_every > 0 and self.global_step % log_every == 0 and self.rank == 0:
                        self.reduce_metrics(
                            step_logging_infos,
                            writer=self.writer,
                            logger=logger,
                            step=self.global_step,
                            split="train_inner",
                            unit="Step",
                            epoch=display_epoch,
                            inner_step=i,
                            inner_total=num_batches,
                        )
                        step_logging_infos = []

                    if save_every > 0 and self.global_step % save_every == 0:
                        val_metrics = None
                        if self.valid_dataloader is not None:
                            val_metrics = self.evaluate(self.global_step, log=self.rank == 0)
                        if self.rank == 0:
                            self._save_checkpoint(epoch, self.global_step, 'checkpoint_last.ckpt')
                            if self.valid_dataloader is not None and val_metrics is not None:
                                curr_loss = val_metrics.get('loss', float('inf'))
                                if curr_loss < self.best_loss:
                                    self.best_loss = curr_loss
                                    self.no_improve_steps = 0
                                    logger.info(
                                        f"New best model at step {self.global_step} with loss {curr_loss:.4f}"
                                    )
                                    self._save_checkpoint(epoch, self.global_step, 'checkpoint_best.ckpt')
                                else:
                                    self.no_improve_steps += 1
                                    if (
                                        self.patience >= 0
                                        and self.no_improve_steps > self.patience
                                    ):
                                        logger.info(
                                            f"Early stopping triggered at step {self.global_step}"
                                        )
                                        stop_training = True
                            self._save_checkpoint(epoch, self.global_step)
                        if stop_training:
                            break

                    if max_steps and self.global_step >= max_steps:
                        if self.rank == 0:
                            logger.info(
                                f"Reached max steps {max_steps}, stopping training"
                            )
                        stop_training = True
                        break

            if self.rank == 0:
                logger.info(
                    f"End of epoch {display_epoch} (average epoch stats below)"
                )
                self.reduce_metrics(
                    epoch_logging_infos,
                    writer=self.writer,
                    logger=logger,
                    step=display_epoch,
                    split="train",
                    unit="Epoch",
                )
            epoch += 1

        final_epoch = epoch - 1
        if self.rank == 0:
            self._save_checkpoint(final_epoch, self.global_step, 'checkpoint_last.ckpt')

        if self.writer:
            self.writer.close()

    def _save_checkpoint(self, epoch, step, name=None):
        if self.rank != 0:
            return
        ckpt = {
            "model": self.model.module.state_dict()
                     if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                     else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        if name is None:
            save_name = f"checkpoint_{epoch+1}_{step}.ckpt"
        else:
            save_name = name
        save_path = os.path.join(self.ckpt_dir, save_name)
        torch.save(ckpt, save_path)
        logger.info(f"Saved checkpoint: {save_path}")
        if name is None:
            self._cleanup_old_checkpoints()

    def _load_checkpoint(self, ckpt_path: str):
        map_loc = {"cuda:%d" % 0: "cuda:%d" % self.local_rank} if self.local_rank >= 0 else None
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        model_sd = ckpt["model"]
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(model_sd)
        else:
            self.model.load_state_dict(model_sd)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["step"]
        logger.info(f"Resume from {ckpt_path} | start_epoch={self.start_epoch} step={self.global_step}")

    def _cleanup_old_checkpoints(self, keep: int = None):
        keep = keep if keep is not None else getattr(self.config, 'keep_last_n_checkpoints', 3)
        ckpt_files = sorted(
            self.ckpt_dir.glob("checkpoint_*_*.ckpt"), key=os.path.getmtime
        )
        for f in ckpt_files[:-keep]:
            f.unlink()
            
    def evaluate(self, step, log: bool = False):
        self.model.eval()
        logging_infos = []
        with torch.no_grad():
            for batch in self.valid_dataloader:
                net_input, net_target = self.decorate_batch(batch)
                loss, sample_size, logging_info = self.loss_fn(
                    self.model, net_input, net_target
                )
                logging_infos.append(logging_info)
        metrics = self.reduce_metrics(
            logging_infos,
            writer=self.writer,
            logger=logger if log else None,
            step=step,
            split="valid",
            unit="Step",
        )
        self.model.train()
        return metrics

    def decorate_batch(self, batch):
        # batch is a dict of tensors (batch_size, ...)
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
        if self.fp16:
            for k in ['src_coord', 'src_distance']:
                net_input[k] = net_input[k].half()
            for k in ['tgt_coordinates', 'tgt_distance']:
                net_target[k] = net_target[k].half()
        return net_input, net_target

    def reduce_metrics(
        self,
        logging_outputs,
        writer=None,
        logger=None,
        step=None,
        split="train",
        unit="Epoch",
        epoch=None,
        inner_step=None,
        inner_total=None,
    ):
        metrics_mean = self.loss_fn.reduce_metrics(logging_outputs, split=split)
        if split.startswith("train") and self.fp16 and self.scaler is not None:
            metrics_mean["loss_scale"] = self.scaler.loss_scale
        if "bsz" in metrics_mean:
            metrics_mean["bsz"] *= self.world_size
        if writer is not None and step is not None:
            if "loss" in metrics_mean:
                writer.add_scalar(f"{split}/loss", metrics_mean["loss"], step)
            for k, v in metrics_mean.items():
                if k == "loss":
                    continue
                writer.add_scalar(f"{split}/{k}", v, step)
        if logger is not None and step is not None:
            log_items = []
            if "loss" in metrics_mean:
                log_items.append(f"loss={metrics_mean['loss']:.4f}")
            for k, v in metrics_mean.items():
                if k in {"loss", "bsz", "loss_scale"}:
                    continue
                log_items.append(f"{k}={v:.4f}")
            bsz_val = metrics_mean.get("bsz")
            if bsz_val is not None:
                log_items.append(f"bsz={int(bsz_val)}")
            current_lr = self.optimizer.param_groups[0]["lr"]
            log_items.append(f"lr={current_lr:.4e}")
            if split.startswith("train") and self.fp16 and self.scaler is not None:
                log_items.append(f"loss_scale={self.scaler.loss_scale:.0f}")
            if (
                split == "train_inner"
                and epoch is not None
                and inner_step is not None
                and inner_total is not None
            ):
                logger.info(
                    f"[{split}] step {step}, epoch {epoch:03d}: {inner_step:6d} / {inner_total}: "
                    + ", ".join(log_items)
                )
            else:
                logger.info(f"[{split}] {unit} {step}: " + ", ".join(log_items))
        return metrics_mean

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)