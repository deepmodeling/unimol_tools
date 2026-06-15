import json
import os
import shutil

import numpy as np
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from unimol_tools.data.conformer import coords2unimol, inner_smi2coords
from unimol_tools.data.dictionary import Dictionary

from .configuration_unimol import UnimolConfig


class UnimolTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "dist_mat", "edge_ids", "coords"]
    vocab_files_names = {"vocab_file": "mol.dict.txt"}

    def __init__(
        self,
        dict_path=None,
        conformer_seed=42,
        conformer_mode="fast",
        remove_hs=False,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_max_length=512,
        **kwargs,
    ):
        self.dict_path = dict_path
        self.conformer_seed = conformer_seed
        self.conformer_mode = conformer_mode
        self.remove_hs = remove_hs
        self.dictionary = Dictionary.load(dict_path) if dict_path else None

        super().__init__(
            cls_token=cls_token,
            sep_token=sep_token,
            pad_token=pad_token,
            unk_token=unk_token,
            model_max_length=model_max_length,
            **kwargs,
        )

        if self.dictionary is not None:
            # Keep parity with ConformerGen / UniMolModel, which append [MASK].
            if "[MASK]" not in self.dictionary:
                self.dictionary.add_symbol("[MASK]", is_special=True)
            self._sync_special_token_ids()

    def _sync_special_token_ids(self):
        self.pad_token_id = self.dictionary.pad()
        self.cls_token_id = self.dictionary.bos()
        self.sep_token_id = self.dictionary.eos()
        self.unk_token_id = self.dictionary.unk()

    @property
    def vocab_size(self):
        return len(self.dictionary) if self.dictionary is not None else 0

    def get_vocab(self):
        return dict(self.dictionary.indices) if self.dictionary is not None else {}

    def _tokenize(self, text):
        raise NotImplementedError("Use encode(smiles) for UniMol inputs.")

    def _convert_token_to_id(self, token):
        return self.dictionary.index(token)

    def _convert_id_to_token(self, index):
        return self.dictionary[index]

    def encode_smiles(self, smiles, add_special_tokens=True):
        if self.dictionary is None:
            raise ValueError("Tokenizer dictionary is not loaded.")

        atoms, coordinates, _ = inner_smi2coords(
            smiles,
            seed=self.conformer_seed,
            mode=self.conformer_mode,
            remove_hs=self.remove_hs,
        )
        sample = coords2unimol(
            atoms,
            coordinates,
            self.dictionary,
            remove_hs=self.remove_hs,
            data_type="molecule",
        )
        return {
            "input_ids": sample["src_tokens"].astype(np.int64),
            "dist_mat": sample["src_distance"].astype(np.float32),
            "edge_ids": sample["src_edge_type"].astype(np.int64),
            "coords": sample["src_coord"].astype(np.float32),
        }

    def encode(self, smiles, add_special_tokens=True, **kwargs):
        return self.encode_smiles(smiles, add_special_tokens=add_special_tokens)

    def batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        return BatchEncoding(
            {
                key: [encoded[key] for encoded in [self.encode_smiles(smi) for smi in batch_text_or_text_pairs]]
                for key in self.model_input_names
            }
        )

    def __call__(self, smiles, **kwargs):
        if isinstance(smiles, (list, tuple)):
            return self.batch_encode_plus(smiles, **kwargs)
        return BatchEncoding(self.encode_smiles(smiles))

    def save_vocabulary(self, save_directory, filename_prefix=""):
        if self.dict_path and os.path.isfile(self.dict_path):
            out = os.path.join(save_directory, "mol.dict.txt")
            shutil.copyfile(self.dict_path, out)
            return (out,)
        return ()

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        tokenizer_config = {
            "auto_map": {
                "AutoTokenizer": ["unimol_hf.tokenization_unimol.UnimolTokenizer", None],
            },
            "conformer_seed": self.conformer_seed,
            "conformer_mode": self.conformer_mode,
            "remove_hs": self.remove_hs,
            "model_max_length": self.model_max_length,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "tokenizer_class": self.__class__.__name__,
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2)
        self.save_vocabulary(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, **kwargs):
        if config is None:
            config = UnimolConfig.from_pretrained(pretrained_model_name_or_path)
        dict_path = config.resolve_dict_path(pretrained_model_name_or_path)
        local_dict = os.path.join(pretrained_model_name_or_path, "mol.dict.txt")
        if os.path.isfile(local_dict):
            dict_path = local_dict

        tokenizer_kwargs = {
            "dict_path": dict_path,
            "conformer_seed": config.conformer_seed,
            "conformer_mode": config.conformer_mode,
            "remove_hs": config.remove_hs,
        }
        tokenizer_kwargs.update(kwargs)
        return cls(**tokenizer_kwargs)
