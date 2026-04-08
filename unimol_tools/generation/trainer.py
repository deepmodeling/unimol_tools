import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

from .config import GenerationConfig
from .dataset import VAEDataset
from .loss import VAELoss
from ..tasks import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class GenerationTrainer:
    def __init__(self, model: nn.Module, dataset: VAEDataset, loss_fn: VAELoss, config: GenerationConfig, valid_dataset=None, local_rank=None):
        self.model = model
        self.dataset = dataset
        self.valid_dataset = valid_dataset
        self.loss_fn = loss_fn
        self.config = config
        self.model_name = config.model.model_name

        self.set_seed(getattr(config.training, "seed", 42))
        # DDP configuration
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if local_rank is None else local_rank
        self.rank = int(os.environ.get("RANK", 0))

        run_dir = getattr(config.training, "output_dir", None)
        if run_dir:
            if not os.path.isabs(run_dir):
                try:
                    run_dir = os.path.join(get_original_cwd(), run_dir)
                except ValueError:
                    run_dir = os.path.join(os.getcwd(), run_dir)
        else:
            try:
                run_dir = HydraConfig.get().run.dir
            except ValueError:
                run_dir = "./generation_runs/default_run"
        self.run_dir = run_dir
        self.ckpt_dir = os.path.join(self.run_dir, 'checkpoints')

        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.model = self.model.to(self.local_rank)
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)

        self.loss_fn = self.loss_fn.to(self.device)

        if self.rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            logger.info(f"Checkpoints will be saved to {self.ckpt_dir}")
        
        # Freeze Encoder
        real_model = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(real_model, "unimol_encoder"):
            for param in real_model.unimol_encoder.parameters():
                param.requires_grad = False
            if self.rank == 0:
                logger.info("Frozen UniMol Encoder parameters.")
                
        # Intercept model.train() to ensure encoder stays in eval mode
        original_train = self.model.train
        def custom_train(mode=True):
            original_train(mode)
            if mode:
                rmodel = self.model.module if isinstance(self.model, DDP) else self.model
                if hasattr(rmodel, "unimol_encoder"):
                    rmodel.unimol_encoder.eval()
            return self.model
        self.model.train = custom_train
        
        # Optimizer (only train requires_grad params)
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(model_params, lr=config.training.lr, weight_decay=getattr(config.training, "weight_decay", 1e-4))
        
        # Sampler
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
            self.valid_sampler = (
                torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
                if self.valid_dataset is not None else None
            )
        else:
            self.sampler = None
            self.valid_sampler = None

        # Dataloader
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=config.training.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            collate_fn=self.dataset.collater,
            num_workers=getattr(config.training, "num_workers", 4),
            pin_memory=True
        )
        if self.valid_dataset:
            self.valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                sampler=self.valid_sampler,
                collate_fn=self.valid_dataset.collater,
                num_workers=getattr(config.training, "num_workers", 4),
                pin_memory=True
            )
        else:
            self.valid_loader = None

        # Scheduler
        total_steps = len(self.train_loader) * config.training.max_epochs
        warmup_steps = getattr(config.training, "warmup_steps", 0)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        # Tensorboard Writer configuration
        self.writer = SummaryWriter(log_dir=self.run_dir) if self.rank == 0 else None
        self.global_step = 0

    def train_epoch(self, epoch):
        self.model.train()
        if self.sampler:
            self.sampler.set_epoch(epoch)

        total_loss = 0
        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_loader
        
        for batch in pbar:
            if not batch:
                continue
            
            self.optimizer.zero_grad()
            
            if self.model_name == "vae":
                # Encoder Inputs
                net_input = batch["net_input"]
                src_tokens = net_input["src_tokens"].to(self.device)
                src_coord = net_input["src_coord"].to(self.device)
                src_distance = net_input["src_distance"].to(self.device)
                src_edge_type = net_input["src_edge_type"].to(self.device)
                
                # Decoder Inputs/Targets
                target = batch["target"].to(self.device)
                
                # For auto-regressive generation, decoder input shouldn't see future tokens. 
                # Input should be: [BOS, t1, t2, ..., t_N-1]
                # Target should be: [t1, t2, ..., t_N-1, EOS]
                decoder_input = target[:, :-1]
                loss_target = target[:, 1:]
                
                output = self.model(
                    src_tokens=src_tokens,
                    src_distance=src_distance,
                    src_coord=src_coord,
                    src_edge_type=src_edge_type,
                    decoder_input_tokens=decoder_input
                )
                
                logits, mean, logv = output["logits"], output["mean"], output["logv"]
                
                loss, recon, kl = self.loss_fn(logits, loss_target, mean, logv, step=self.global_step)
                
            elif self.model_name == "edm":
                # Move inputs to device
                for k, v in batch["net_input"].items():
                    batch["net_input"][k] = v.to(self.device)
                
                loss, loss_x, loss_z, loss_dist = self.loss_fn(self.model, batch)
                recon = loss_x
                kl = loss_z
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            self.global_step += 1
            if self.rank == 0 and self.writer:
                if self.global_step % getattr(self.config.training, "log_every_n_steps", 10) == 0:
                    self.writer.add_scalar("Train/Loss", loss.item(), self.global_step)
                    self.writer.add_scalar("Train/Recon", recon.item(), self.global_step)
                    self.writer.add_scalar("Train/KL", kl.item(), self.global_step)
                    self.writer.add_scalar("Train/LR", self.optimizer.param_groups[0]["lr"], self.global_step)
                    
            if self.global_step % getattr(self.config.training, "save_every_n_steps", 5000) == 0:
                self.validate(epoch, step=self.global_step)
                if self.rank == 0:
                    self.save_checkpoint(epoch, step=self.global_step)
                    
            total_loss += loss.item()
            if self.rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": loss.item(), "recon": recon.item(), "kl": kl.item()})
            
        if dist.is_initialized():
            loss_tensor = torch.tensor(total_loss).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item() / dist.get_world_size()

        avg_loss = total_loss / len(self.train_loader)
        if self.rank == 0:
            logger.info(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")
            if self.writer:
                self.writer.add_scalar("Epoch/Train_Loss", avg_loss, epoch)
        return avg_loss

    def validate(self, epoch, step=None):
        if not self.valid_dataset:
            return
        
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                if not batch:
                    continue
                
                if self.model_name == "vae":
                    net_input = batch["net_input"]
                    src_tokens = net_input["src_tokens"].to(self.device)
                    src_coord = net_input["src_coord"].to(self.device)
                    src_distance = net_input["src_distance"].to(self.device)
                    src_edge_type = net_input["src_edge_type"].to(self.device)
                    
                    target = batch["target"].to(self.device)
                    decoder_input = target[:, :-1]
                    loss_target = target[:, 1:]
                    
                    output = self.model(
                        src_tokens=src_tokens,
                        src_distance=src_distance,
                        src_coord=src_coord,
                        src_edge_type=src_edge_type,
                        decoder_input_tokens=decoder_input
                    )
                    logits, mean, logv = output["logits"], output["mean"], output["logv"]
                    loss, _, _ = self.loss_fn(logits, loss_target, mean, logv)
                elif self.model_name == "edm":
                    for k, v in batch["net_input"].items():
                        batch["net_input"][k] = v.to(self.device)
                    loss, _, _, _ = self.loss_fn(self.model, batch)
                    
                total_loss += loss.item()
        
        if dist.is_initialized():
            loss_tensor = torch.tensor(total_loss).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item() / dist.get_world_size()

        avg_loss = total_loss / len(self.valid_loader)
        if self.rank == 0:
            if step is not None:
                logger.info(f"Epoch {epoch} Step {step} Valid Loss: {avg_loss:.4f}")
                if self.writer:
                    self.writer.add_scalar("Step/Valid_Loss", avg_loss, step)
            else:
                logger.info(f"Epoch {epoch} Valid Loss: {avg_loss:.4f}")
                if self.writer:
                    self.writer.add_scalar("Epoch/Valid_Loss", avg_loss, epoch)
        self.model.train()
        return avg_loss

    def train_loop(self):
        if self.rank == 0:
            logger.info(f"Starting training loop for {self.config.training.max_epochs} epochs.")
        for epoch in range(self.config.training.max_epochs):
            self.train_epoch(epoch)
            
        if self.rank == 0:
            self.save_checkpoint(self.config.training.max_epochs - 1, step=self.global_step, name='checkpoint_last.pt')
            
        if self.writer:
            self.writer.close()

    def _cleanup_old_checkpoints(self, keep: int = 5):
        if self.rank != 0:
            return
        from pathlib import Path
        ckpt_dir = Path(self.ckpt_dir)
        if not ckpt_dir.exists():
            return
        ckpt_files = sorted(ckpt_dir.glob("checkpoint_*_*.pt"), key=os.path.getmtime)
        keep = getattr(self.config.training, 'keep_last_n_checkpoints', keep)
        for f in ckpt_files[:-keep]:
            f.unlink()

    def save_checkpoint(self, epoch, step=None, name=None):
        if self.rank != 0:
            return
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        if name is not None:
            path = os.path.join(self.ckpt_dir, name)
        elif step is not None:
            path = os.path.join(self.ckpt_dir, f"checkpoint_{epoch+1}_{step}.pt")
        else:
            path = os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch}.pt")
        
        # Save only decoder and VAE heads
        # Filter state dict: exclude 'unimol_encoder'
        real_model = self.model.module if isinstance(self.model, DDP) else self.model
        model_state = real_model.state_dict()
        filtered_state = {k: v for k, v in model_state.items() if not k.startswith("unimol_encoder")}
        
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': filtered_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Saved checkpoint (decoder only) to {path}")
        if name is None:
            self._cleanup_old_checkpoints()

    def set_seed(self, seed):
        """
        Sets a random seed for torch and numpy to ensure reproducibility.
        :param seed: (int) The seed number to be set.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)