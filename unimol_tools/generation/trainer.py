import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import GenerationConfig
from .dataset import VAEDataset
from .loss import VAELoss

logger = logging.getLogger(__name__)

class GenerationTrainer:
    def __init__(self, model: nn.Module, dataset: VAEDataset, loss_fn: VAELoss, config: GenerationConfig, valid_dataset=None):
        self.model = model
        self.dataset = dataset
        self.valid_dataset = valid_dataset
        self.loss_fn = loss_fn
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Freeze Encoder
        if hasattr(self.model, "unimol_encoder"):
            for param in self.model.unimol_encoder.parameters():
                param.requires_grad = False
            logger.info("Frozen UniMol Encoder parameters.")
        
        # Optimizer (only train requires_grad params)
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        # Use AdamW if available or standard Adam
        self.optimizer = optim.Adam(model_params, lr=config.lr, weight_decay=config.weight_decay)
        
        # Dataloader
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collater,
            num_workers=4
        )
        if self.valid_dataset:
            self.valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=self.valid_dataset.collater,
                num_workers=4
            )

        # Scheduler
        total_steps = len(self.train_loader) * config.max_epochs
        warmup_steps = config.warmup_steps 
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    def train_epoch(self, epoch):
        self.model.train()
        # Ensure encoder stays in eval mode (for BatchNorm/Dropout behaviors if needed)
        if hasattr(self.model, "unimol_encoder"):
            self.model.unimol_encoder.eval()

        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            if not batch:
                continue
            
            # Encoder Inputs
            net_input = batch["net_input"]
            src_tokens = net_input["src_tokens"].to(self.device)
            src_coord = net_input["src_coord"].to(self.device)
            src_distance = net_input["src_distance"].to(self.device)
            src_edge_type = net_input["src_edge_type"].to(self.device)
            
            # Decoder Inputs/Targets
            target = batch["target"].to(self.device)
            # Decoder input: [BOS, t1, t2] (remove EOS at end)
            decoder_input = target[:, :-1]
            # Loss target: [t1, t2, EOS] (remove BOS at start)
            loss_target = target[:, 1:]
            
            self.optimizer.zero_grad()
            
            output = self.model(
                src_tokens=src_tokens,
                src_distance=src_distance,
                src_coord=src_coord,
                src_edge_type=src_edge_type,
                decoder_input_tokens=decoder_input
            )
            
            logits, mean, logv = output["logits"], output["mean"], output["logv"]
            
            loss, recon, kl = self.loss_fn(logits, loss_target, mean, logv)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "recon": recon.item(), "kl": kl.item()})
            
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch} Train Loss: {avg_loss}")
        return avg_loss

    def validate(self, epoch):
        if not self.valid_dataset:
            return
        
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                if not batch:
                    continue
                
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
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.valid_loader)
        logger.info(f"Epoch {epoch} Valid Loss: {avg_loss}")
        return avg_loss

    def train_loop(self):
        for epoch in range(self.config.max_epochs):
            self.train_epoch(epoch)
            self.validate(epoch)
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        path = os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.pt")
        
        # Save only decoder and VAE heads
        # Filter state dict: exclude 'unimol_encoder'
        model_state = self.model.state_dict()
        filtered_state = {k: v for k, v in model_state.items() if not k.startswith("unimol_encoder")}
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': filtered_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Saved checkpoint (decoder only) to {path}")
