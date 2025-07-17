from functools import lru_cache
import os
import random
import pickle
import lmdb
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class LMDBDataset(Dataset):
    """
    读取LMDB，输出 idx、元素种类、原子序数、3D坐标，支持缓存
    """
    def __init__(self, lmdb_path):
        env = lmdb.open(
            lmdb_path, 
            subdir=False, 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False, 
            max_readers=256,
        ) 
        self.txn = env.begin()
        self.length = self.txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.txn.get(str(idx).encode())
        if data is None:
            raise IndexError(f"Index {idx} not found in LMDB.")
        item = pickle.loads(data)
        atoms = item.get('atoms')
        coordinates = item.get('coordinates')

        result = {
            'idx': item.get('idx'),
            'atoms': atoms,
            'coordinates': coordinates,
            'smiles': item.get('smiles'),
        }
        return result

class UniMolDataset(Dataset):
    """
    Loads LMDBdataset for UniMol models.
    """
    def __init__(self, lmdb_dataset, dictionary, remove_hs=False, max_atoms=256, seed=1, **params):
        self.dataset = lmdb_dataset
        self.length = len(self.dataset)
        self.dictionary = dictionary
        self.remove_hs = remove_hs
        self.max_atoms = max_atoms
        self.seed = seed
        self.params = params
        self.mask_id = dictionary.add_symbol("[MASK]", is_special=True)
        self.set_epoch(0)  # Initialize epoch to 0

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
        np.random.seed(self.seed + epoch)
        self.sort_order = np.random.permutation(self.length)

    def ordered_indices(self):
        return self.sort_order


    def __getitem__(self, idx):
        # print("dataset worker random state", random.getstate()[1][:3])
        return self.__getitem__cached__(self.epoch, idx)
    
    @lru_cache(maxsize=16)
    def __getitem__cached__(self, epoch, idx):
        item = self.dataset[idx]
        atoms = item['atoms']
        coordinates = item['coordinates']
        
        if atoms is None or coordinates is None:
            raise ValueError(f"Invalid data at index {idx}: atoms or coordinates are None.")
        
        net_input, target = coords2unimol(
            atoms=atoms, 
            coordinates=coordinates, 
            dictionary=self.dictionary, 
            mask_id=self.mask_id,
            noise_type=self.params.get('noise_type', 'trunc_normal'),
            noise=self.params.get('noise', 1.0),
            seed=self.params.get('seed', 1),
            epoch=epoch,
            mask_prob=self.params.get('mask_prob', 0.15),
            leave_unmasked_prob=self.params.get('leave_unmasked_prob', 0.1),
            random_token_prob=self.params.get('random_token_prob', 0.1),
            max_atoms=self.max_atoms,
            remove_hs=self.remove_hs,
        )
        return net_input, target

def coords2unimol(
        atoms, 
        coordinates, 
        dictionary, 
        mask_id,
        noise_type,
        noise=1.0,
        seed=1,
        epoch=0,
        mask_prob=0.15,
        leave_unmasked_prob=0.1,
        random_token_prob=0.1,
        max_atoms=256, 
        remove_hs=True, 
        **params
    ):
    np.random.seed(seed + epoch)
    torch.manual_seed(seed + epoch)

    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        atoms, coordinates = atoms_no_h, coordinates_no_h

    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = torch.randperm(len(atoms))[:max_atoms]
        atoms = [atoms[i] for i in idx.tolist()]
        coordinates = coordinates[idx]

    # coordinates normalization
    coordinates = coordinates - coordinates.mean(dim=0)

    # add noise and mask
    src_tokens, src_coord, tgt_tokens = apply_noise_and_mask(
        src_tokens=torch.tensor([dictionary.index(atom) for atom in atoms], dtype=torch.long),
        coordinates=coordinates,
        dictionary=dictionary,
        mask_id=mask_id,
        noise_type=noise_type,
        noise=noise,
        mask_prob=mask_prob,
        leave_unmasked_prob=leave_unmasked_prob,
        random_token_prob=random_token_prob
    )

    # tokens padding
    src_tokens = torch.cat([torch.tensor([dictionary.bos()]), src_tokens, torch.tensor([dictionary.eos()])], dim=0)
    tgt_tokens = torch.cat([torch.tensor([dictionary.bos()]), tgt_tokens, torch.tensor([dictionary.eos()])], dim=0)

    # coordinates padding
    pad = torch.zeros((1, 3), dtype=torch.float32)
    src_coord = torch.cat([pad, src_coord, pad], dim=0)
    tgt_coordinates = torch.cat([pad, coordinates, pad], dim=0)

    # distance matrix
    diff = src_coord.unsqueeze(0) - src_coord.unsqueeze(1)
    src_distance = torch.norm(diff, dim=-1)
    tgt_distance = torch.norm(tgt_coordinates.unsqueeze(0) - tgt_coordinates.unsqueeze(1), dim=-1)

    # edge type
    src_edge_type = src_tokens.view(-1, 1) * len(dictionary) + src_tokens.view(1, -1)

    return {
        'src_tokens': src_tokens,
        'src_coord': src_coord,
        'src_distance': src_distance,
        'src_edge_type': src_edge_type,
    },{
        'tgt_tokens': tgt_tokens,
        'tgt_coordinates': tgt_coordinates,
        'tgt_distance': tgt_distance,
    }


def apply_noise_and_mask(
        src_tokens, 
        coordinates,
        dictionary,
        mask_id,
        noise_type, 
        noise=1.0, 
        mask_prob=0.15, 
        leave_unmasked_prob=0.1, 
        random_token_prob=0.1
    ):
    """
    Apply noise and masking to the source tokens.
    """
    if random_token_prob > 0:
        weights = np.ones(len(dictionary)) 
        weights[dictionary.special_index()] = 0
        weights /= weights.sum()
    
    sz = len(src_tokens)
    assert sz > 0, "Source tokens must not be empty."

    num_mask = int(sz * mask_prob + np.random.rand())
    mask_idc = np.random.choice(sz, num_mask, replace=False)
    mask = np.full(sz, fill_value=False)
    mask[mask_idc] = True

    tgt_tokens = np.full(sz, dictionary.pad())
    tgt_tokens[mask] = src_tokens[mask]
    tgt_tokens = torch.from_numpy(tgt_tokens).long()

    # determine unmasked and random tokens
    rand_or_unmask_prob = random_token_prob + leave_unmasked_prob
    if rand_or_unmask_prob > 0:
        rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
        if random_token_prob == 0:
            unmasked = rand_or_unmask
            rand_mask = None
        elif leave_unmasked_prob == 0:
            unmasked = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
            unmasked = rand_or_unmask & (np.random.rand(sz) < unmask_prob)
            rand_mask = rand_or_unmask & ~unmasked
    else:
        unmasked = None
        rand_mask = None
    
    if unmasked is not None:
        mask = mask ^ unmasked

    new_src_tokens = src_tokens.clone()
    new_src_tokens[mask] = mask_id

    num_mask = mask.sum().item()
    new_coordinates = coordinates.clone()
    # new_coordinates[mask, :] += noise_f
    if noise_type == "trunc_normal":
        noise_f = np.clip(np.random.randn(num_mask, 3) * noise, -noise*2, noise*2)
    elif noise_type == "normal":
        noise_f = np.random.randn(num_mask, 3) * noise
    elif noise_type == "uniform":
        noise_f = np.random.uniform(-noise, noise, size=(num_mask, 3))
    else:
        noise_f = np.zeros((num_mask, 3), dtype=np.float32)
    new_coordinates[mask, :] += torch.tensor(noise_f, dtype=torch.float32)

    if rand_mask is not None:
        num_rand = rand_mask.sum()
        if num_rand > 0:
            new_src_tokens[rand_mask] = torch.tensor(
                np.random.choice(len(dictionary), num_rand, p=weights), dtype=torch.long
            )
    return new_src_tokens, new_coordinates, tgt_tokens