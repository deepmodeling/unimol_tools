import torch
import torch.nn as nn
import numpy as np
import lmdb
import pickle
from torch.utils.data import Dataset
from scipy.spatial import distance_matrix
from unimol_tools.utils import pad_1d_tokens, pad_2d, pad_coords
from .data_utils import randomize_smiles, SmilesTokenizer

import logging
logger = logging.getLogger(__name__)

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None
        self.txn = None
        self.length = 0
        try:
            env = lmdb.open(
                lmdb_path, 
                subdir=False, 
                readonly=True, 
                lock=False, 
                readahead=False, 
                meminit=False, 
                max_readers=256,
            ) 
            with env.begin() as txn:
                self.length = txn.stat()['entries']
            env.close()
        except Exception as e:
            logger.error(f"Failed to open LMDB at {lmdb_path}: {e}")

    def _init_db(self):
        self.env = lmdb.open(
            self.lmdb_path, 
            subdir=False, 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False, 
            max_readers=256,
        )
        self.txn = self.env.begin()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()
        try:
            data = self.txn.get(str(idx).encode())
        except:
             return None
        if data is None:
            return None
        item = pickle.loads(data)
        return item

class VAEDataset(Dataset):
    def __init__(self, data_path, dictionary, vae_dict, max_len=256, add_bos=True, add_eos=True, randomize=True):
        self.lmdb_dataset = LMDBDataset(data_path)
        self.dictionary = dictionary
        self.vae_dict = vae_dict
        self.max_len = max_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.randomize = randomize
        self.tokenizer = SmilesTokenizer(self.vae_dict, encoder_dict=self.dictionary, max_len=max_len)

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, idx):
        item = self.lmdb_dataset[idx]
        if item is None:
            return None
            
        atoms = item['atoms']
        coordinates = item['coordinates']
        smiles = item['smi']

        # --- Encoder Input Preparation (UniMol Format) ---
        src_tokens = [self.dictionary.index(a) for a in atoms]

        src_tokens = torch.tensor(src_tokens, dtype=torch.long)
        
        # Coordinates
        idx_coord = np.random.randint(len(coordinates))
        coordinates = torch.from_numpy(np.array(coordinates[idx_coord])).float()
            
        # Distance Matrix
        dist = distance_matrix(coordinates.numpy(), coordinates.numpy()).astype(np.float32)
        src_distance = torch.from_numpy(dist)
        
        # Edge Type
        # src_edge_type = atom_i * vocab_size + atom_j
        src_edge_type = src_tokens.view(-1, 1) * len(self.dictionary) + src_tokens.view(1, -1)
        src_edge_type = src_edge_type.long()

        # --- Decoder Target Preparation (SMILES) ---
        if self.randomize and smiles:
            smiles = randomize_smiles(smiles)

        tgt_tokens = torch.tensor(self.tokenizer.encode(smiles))
        
        if self.add_bos:
            tgt_tokens = torch.cat([torch.tensor([self.vae_dict.bos()]), tgt_tokens])
        if self.add_eos:
            tgt_tokens = torch.cat([tgt_tokens, torch.tensor([self.vae_dict.eos()])])

        return {
            "net_input": {
                "src_tokens": src_tokens,
                "src_coord": coordinates, 
                "src_distance": src_distance,
                "src_edge_type": src_edge_type,
            },
            "target": tgt_tokens,
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}

        src_tokens = [s["net_input"]["src_tokens"] for s in samples]
        src_coord = [s["net_input"]["src_coord"] for s in samples]
        src_distance = [s["net_input"]["src_distance"] for s in samples]
        src_edge_type = [s["net_input"]["src_edge_type"] for s in samples]
        
        targets = [s["target"] for s in samples]

        pad_idx = self.dictionary.pad()
        target_pad_idx = self.vae_dict.pad()
        
        padded_src_tokens = pad_1d_tokens(src_tokens, pad_idx)
        padded_src_coord = pad_coords(src_coord, 0.0, dim=3) 
        padded_src_distance = pad_2d(src_distance, 0.0) 
        padded_src_edge_type = pad_2d(src_edge_type, 0) 

        padded_targets = pad_1d_tokens(targets, target_pad_idx)

        return {
            "net_input": {
                "src_tokens": padded_src_tokens,
                "src_coord": padded_src_coord,
                "src_distance": padded_src_distance,
                "src_edge_type": padded_src_edge_type,
            },
            "target": padded_targets,
        }

class EDMDataset(Dataset):
    def __init__(self, data_path, dictionary, max_len=256, remove_hs=False):
        self.lmdb_dataset = LMDBDataset(data_path)
        self.dictionary = dictionary
        self.max_len = max_len
        self.remove_hs = remove_hs

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, idx):
        item = self.lmdb_dataset[idx]
        if item is None:
            return None

        # --- Input Preparation (UniMol Format) ---   
        atoms = item['atoms']
        raw_coordinates = item['coordinates']
        idx_coord = np.random.randint(len(raw_coordinates))
        coordinates = torch.from_numpy(np.array(raw_coordinates[idx_coord])).float()

        atoms, coordinates = inner_coords(atoms, coordinates, remove_hs=self.remove_hs)

        atoms = atoms[:self.max_len - 2]  # Reserve space for BOS and EOS
        coordinates = coordinates[:self.max_len - 2]  # Reserve space for BOS and EOS

        # Atoms
        src_tokens = [self.dictionary.index(a) for a in atoms]
        src_tokens = torch.tensor(src_tokens, dtype=torch.long)
        src_tokens = torch.cat([
            torch.tensor([self.dictionary.bos()]), 
            src_tokens, 
            torch.tensor([self.dictionary.eos()])
        ])
        
        # Coordinates
        pad = torch.zeros((1, 3), dtype=torch.float32)
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        coordinates = torch.cat([pad, coordinates, pad], dim=0)  # Add
            
        # Distance Matrix
        diff = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
        src_distance = torch.norm(diff, dim=-1)

        # Edge Type
        src_edge_type = src_tokens.view(-1, 1) * len(self.dictionary) + src_tokens.view(1, -1)
        src_edge_type = src_edge_type.long()

        return {
            "net_input": {
                "src_tokens": src_tokens,
                "src_coord": coordinates, 
                "src_distance": src_distance,
                "src_edge_type": src_edge_type,
            }
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}

        src_tokens = [s["net_input"]["src_tokens"] for s in samples]
        src_coord = [s["net_input"]["src_coord"] for s in samples]
        src_distance = [s["net_input"]["src_distance"] for s in samples]
        src_edge_type = [s["net_input"]["src_edge_type"] for s in samples]
        
        pad_idx = self.dictionary.pad()
        
        padded_src_tokens = pad_1d_tokens(src_tokens, pad_idx)
        padded_src_coord = pad_coords(src_coord, 0.0, dim=3) 
        padded_src_distance = pad_2d(src_distance, 0.0) 
        padded_src_edge_type = pad_2d(src_edge_type, 0) 

        return {
            "net_input": {
                "src_tokens": padded_src_tokens,
                "src_coord": padded_src_coord,
                "src_distance": padded_src_distance,
                "src_edge_type": padded_src_edge_type,
            }
        }

def inner_coords(atoms, coordinates, remove_hs=True):
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(
            coordinates_no_h
        ), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates