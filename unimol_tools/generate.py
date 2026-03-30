import argparse
import os
import torch
from unimol_tools.generation.config import GenerationConfig
from unimol_tools.generation.trainer import GenerationTrainer
from unimol_tools.generation.dataset import VAEDataset
from unimol_tools.generation.loss import VAELoss
from unimol_tools.models.vae import UniMolVAE
from unimol_tools.data.dictionary import Dictionary
from unimol_tools.pretrain.preprocess import build_dictionary
from unimol_tools.generation.data_utils import build_vae_dictionary

class MolGeneration:
    def __init__(self, data_path, dict_path=None, vae_dict_path=None, **kwargs):
        # Filter kwargs to match GenerationConfig fields
        valid_keys = GenerationConfig.__annotations__.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        
        self.config = GenerationConfig(
            data_path=data_path, 
            dict_path=dict_path, 
            vae_dict_path=vae_dict_path,
            **filtered_kwargs)
        
        # Load dictionary
        if dict_path and os.path.exists(dict_path):
            self.dictionary = Dictionary.load(dict_path)
        else:
            # Build dictionary if not provided
            self.dictionary = build_dictionary(data_path, save_path=dict_path)

        if vae_dict_path and os.path.exists(vae_dict_path):
            self.vae_dict = Dictionary.load(vae_dict_path)
        else:
            self.vae_dict = build_vae_dictionary(data_path, save_path=vae_dict_path)

    def train(self):
        dataset = VAEDataset(self.config.data_path, self.dictionary, self.vae_dict)
        
        # Split dataset if valid_path provided, else just train on all or split here
        train_dataset = dataset
        valid_dataset = None # Implement split logic if needed
        
        model = UniMolVAE(self.config, self.dictionary, self.vae_dict)
        loss_fn = VAELoss(beta=self.config.beta, pad_idx=self.dictionary.pad())
        
        trainer = GenerationTrainer(model, train_dataset, loss_fn, self.config, valid_dataset)
        trainer.train_loop()
        
    def generate(self, num_samples=10, checkpoint_path=None):
        # Load model
        model = UniMolVAE(self.config, self.dictionary, self.vae_dict)
        if checkpoint_path:
             state = torch.load(checkpoint_path, map_location='cpu')
             model.load_state_dict(state['model_state_dict'])
        
        model.eval()
        # Sampling logic
        # z = torch.randn(num_samples, self.config.latent_dim)
        # Decode z -> smiles
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to SMILES data")
    parser.add_argument("--dict_path", type=str, default=None, help="Path to dictionary")
    parser.add_argument("--output_dir", type=str, default="checkpoints_gen", help="Output directory")
    args = parser.parse_args()
    
    generator = MolGeneration(data_path=args.data_path, dict_path=args.dict_path, output_dir=args.output_dir)
    generator.train()
