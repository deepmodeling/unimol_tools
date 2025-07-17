import os
import pickle

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from unimol_tools.data import Dictionary
from unimol_tools.data.conformer import inner_smi2coords


def build_dictionary(lmdb_path, save_path=None):
    """
    Count all element types and return a dictionary list with special tokens.
    """
    env = lmdb.open(
        lmdb_path, 
        subdir=False, 
        readonly=True, 
        lock=False, 
        readahead=False, 
        meminit=False, 
        max_readers=256,
    )
    txn = env.begin()
    length = txn.stat()['entries']
    elements_set = set()
    for idx in range(length):
        data = txn.get(str(idx).encode())
        if data is None:
            continue
        item = pickle.loads(data)
        atoms = item.get('atoms')
        if atoms:
            elements_set.update(atoms)
    special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[UNK]']
    dictionary = special_tokens + sorted(list(elements_set))
    env.close()
    if save_path is None:
        save_path = os.path.join(os.path.dirname(lmdb_path), 'dictionary.txt')
    # Save dictionary to file
    with open(save_path, 'wb') as f:
        np.savetxt(f, dictionary, fmt='%s')
    return Dictionary.from_list(dictionary)

def write_to_lmdb(lmdb_path, smiles_list):

    env = lmdb.open(lmdb_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),  # 100GB
    )
    txn_write = env.begin(write=True)
    for i in tqdm(range(len(smiles_list)), total=len(smiles_list)):
        inner_output = process_smiles(smiles_list[i], i)
        if inner_output is not None:
            idx, data = inner_output
            txn_write.put(str(idx).encode(), data)
        if (i+1) % 1000 == 0:
            txn_write.commit()
            txn_write = env.begin(write=True)
    print(f"process {i+1} lines")
    txn_write.commit()
    env.close()
    print(f"已保存到LMDB: {lmdb_path}")
    return lmdb_path

def process_smiles(smiles, idx, remove_hs=False, **params):
    """
    Process a single SMILES string and return index and serialized data.
    """
    atoms, coordinates, mol = inner_smi2coords(
                smiles, seed=42, mode='fast', remove_hs=remove_hs
            )
    # atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    data = {
        'idx': idx,
        'atoms': atoms,
        'coordinates': coordinates,
        'smiles': smiles,
    }

    return idx, pickle.dumps(data)

def process_csv(csv_path, smiles_col='SMILES'):
    """
    Read a CSV file and return a list of SMILES strings.
    """
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in CSV file.")
    return df[smiles_col].tolist()

