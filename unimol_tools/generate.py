import argparse
import os
import torch
from unimol_tools.generation.config import GenerationConfig, DatasetConfig, ModelConfig, TrainingConfig
from unimol_tools.generation.trainer import GenerationTrainer
from unimol_tools.generation.dataset import VAEDataset, EDMDataset
from unimol_tools.generation.loss import VAELoss, EDMLoss, get_token_weights
from unimol_tools.models.unimolvae import UniMolVAE
from unimol_tools.models.unimoledm import UniMolSE3EDM
from unimol_tools.data.dictionary import Dictionary
from unimol_tools.pretrain.preprocess import build_dictionary
from unimol_tools.generation.data_utils import build_vae_dictionary

MODEL_REGISTRY = {
    "vae": {
        "model": UniMolVAE,
        "dataset": VAEDataset,
        "loss": VAELoss
    },
    "edm": {
        "model": UniMolSE3EDM,
        "dataset": EDMDataset,
        "loss": EDMLoss
    }
}

class MolGeneration:
    def __init__(self, train_path, dict_path=None, vae_dict_path=None, model_name="vae", unimol_weight_path=None, **kwargs):
        # Filter kwargs to match config fields
        dataset_kwargs = {k: v for k, v in kwargs.items() if k in DatasetConfig.__annotations__}
        model_kwargs = {k: v for k, v in kwargs.items() if k in ModelConfig.__annotations__}
        training_kwargs = {k: v for k, v in kwargs.items() if k in TrainingConfig.__annotations__}
        
        self.unimol_weight_path = unimol_weight_path
        
        dataset_config = DatasetConfig(
            train_path=train_path, 
            dict_path=dict_path, 
            vae_dict_path=vae_dict_path,
            **dataset_kwargs
        )
        
        model_config = ModelConfig(
            model_name=model_name,
            unimol_weight_path=unimol_weight_path,
            **model_kwargs
        )
        training_config = TrainingConfig(**training_kwargs)
        
        self.config = GenerationConfig(dataset=dataset_config, model=model_config, training=training_config)
        self.model_name = self.config.model.model_name
        
        # Load dictionary
        if dict_path and os.path.exists(dict_path):
            self.dictionary = Dictionary.load(dict_path)
        else:
            # Build dictionary if not provided
            self.dictionary = build_dictionary(train_path, save_path=dict_path)

        if self.model_name == "vae":
            if vae_dict_path and os.path.exists(vae_dict_path):
                self.vae_dict = Dictionary.load(vae_dict_path)
            else:
                self.vae_dict = build_vae_dictionary(train_path, save_path=vae_dict_path)

            self.token_weights = get_token_weights(self.dictionary, self.vae_dict)
        elif self.model_name == "edm":
            self.vae_dict = None
            self.token_weights = None
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_name: {self.model_name}")

    def train(self):
        model_cls = MODEL_REGISTRY[self.model_name]["model"]
        dataset_cls = MODEL_REGISTRY[self.model_name]["dataset"]
        loss_cls = MODEL_REGISTRY[self.model_name]["loss"]

        if self.model_name == "vae":
            train_dataset = dataset_cls(self.config.dataset.train_path, self.dictionary, self.vae_dict)
            valid_dataset = dataset_cls(self.config.dataset.valid_path, self.dictionary, self.vae_dict) if self.config.dataset.valid_path else None

            model = model_cls(self.config, self.dictionary, self.vae_dict)
            loss_fn = loss_cls(beta=self.config.training.beta, pad_idx=self.vae_dict.pad(), token_weights=self.token_weights)
            
        elif self.model_name == "edm":
            train_dataset = dataset_cls(
                self.config.dataset.train_path, 
                self.dictionary, 
                max_len=self.config.model.max_seq_len, 
                remove_hs=self.config.model.remove_hs
            )
            valid_dataset = dataset_cls(
                self.config.dataset.valid_path, 
                self.dictionary, 
                max_len=self.config.model.max_seq_len, 
                remove_hs=self.config.model.remove_hs
            ) if self.config.dataset.valid_path else None

            model = model_cls(self.config, self.dictionary)
            loss_fn = loss_cls(pad_idx=self.dictionary.pad(), lambda_z=1.0, lambda_dist=1.0)
            
        if self.unimol_weight_path:
            print(f"Loading UniMol weights from {self.unimol_weight_path}...")
            model.load_unimol_weights(self.unimol_weight_path)
        trainer = GenerationTrainer(model, train_dataset, loss_fn, self.config, valid_dataset)
        trainer.train_loop()
        
    def generate(self, num_samples=10, checkpoint_path=None):
        # Load model
        model_cls = MODEL_REGISTRY[self.model_name]["model"]
        
        if self.model_name == "vae":
            model = model_cls(self.config, self.dictionary, self.vae_dict)
        elif self.model_name == "edm":
            model = model_cls(self.config, self.dictionary)
        
        if self.unimol_weight_path:
            print(f"Loading UniMol weights from {self.unimol_weight_path}...")
            model.load_unimol_weights(self.unimol_weight_path)
            
        if checkpoint_path:
             state = torch.load(checkpoint_path, map_location='cpu')
             model.load_state_dict(state['model_state_dict'], strict=False)
        
        model.eval()
        # Sampling logic
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Path to SMILES data")
    parser.add_argument("--dict_path", type=str, default=None, help="Path to dictionary")
    parser.add_argument("--vae_dict_path", type=str, default=None, help="Path to VAE dictionary")
    parser.add_argument("--model_name", type=str, default="vae", choices=["vae", "edm"], help="Model name (vae or edm)")
    parser.add_argument("--unimol_weight_path", type=str, default=None, help="Path to Unimol pretrained weight")
    parser.add_argument("--output_dir", type=str, default="checkpoints_gen", help="Output directory")
    args = parser.parse_args()
    
    generator = MolGeneration(train_path=args.train_path, dict_path=args.dict_path, vae_dict_path=args.vae_dict_path, model_name=args.model_name, unimol_weight_path=args.unimol_weight_path, output_dir=args.output_dir)
    generator.train()
