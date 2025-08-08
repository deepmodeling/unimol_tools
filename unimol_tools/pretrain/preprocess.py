import os
import pickle
import logging

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from unimol_tools.data import Dictionary
from unimol_tools.data.conformer import inner_smi2coords

logger = logging.getLogger(__name__)


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

def write_to_lmdb(lmdb_path, smi_list, num_conf=10, add_2d=True, remove_hs=False):
    logger.info(f"Writing {len(smi_list)} SMILES to {lmdb_path}")

    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),  # 100GB
    )
    txn_write = env.begin(write=True)
    for i in tqdm(range(len(smi_list)), total=len(smi_list)):
        inner_output = process_smi(
            smi_list[i],
            i,
            remove_hs=remove_hs,
            num_conf=num_conf,
            add_2d=add_2d,
        )
        if inner_output is not None:
            idx, data = inner_output
            txn_write.put(str(idx).encode(), data)
        if (i + 1) % 1000 == 0:
            txn_write.commit()
            txn_write = env.begin(write=True)
    logger.info(f"Processed {i+1} molecules")
    txn_write.commit()
    env.close()
    logger.info(f"Saved to LMDB: {lmdb_path}")
    return lmdb_path

def write_dicts_to_lmdb(lmdb_path, mol_list):
    """Write a list of pre-generated molecules to LMDB."""
    logger.info(f"Writing {len(mol_list)} molecule dicts to {lmdb_path}")

    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env.begin(write=True)
    for i, item in enumerate(tqdm(mol_list, total=len(mol_list))):
        data = {
            "idx": i,
            "atoms": item["atoms"],
            "coordinates": item["coordinates"],
        }
        if "smi" in item:
            data["smi"] = item["smi"]
        txn_write.put(str(i).encode(), pickle.dumps(data))
        if (i + 1) % 1000 == 0:
            txn_write.commit()
            txn_write = env.begin(write=True)
    logger.info(f"Processed {i+1} molecules")
    txn_write.commit()
    env.close()
    logger.info(f"Saved to LMDB: {lmdb_path}")
    return lmdb_path

def smi2_2dcoords(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates)
    return coordinates

def process_smi(smi, idx, remove_hs=False, num_conf=10, add_2d=True, **params):
    """Process a single SMILES string and return index and serialized data."""
    if add_2d:
        conformers = []
        for i in range(num_conf):
            atoms, coordinates, _ = inner_smi2coords(
                smi, seed=42 + i, mode="fast", remove_hs=remove_hs
            )
            conformers.append(coordinates)
        conformers.append(smi2_2dcoords(smi))
        data = {
            "idx": idx,
            "atoms": atoms,
            "coordinates": conformers,
            "smi": smi,
        }
    else:
        atoms, coordinates, _ = inner_smi2coords(
            smi, seed=42, mode="fast", remove_hs=remove_hs
        )
        data = {
            "idx": idx,
            "atoms": atoms,
            "coordinates": coordinates,
            "smi": smi,
        }

    return idx, pickle.dumps(data)

def process_csv(csv_path, smiles_col='smi'):
    """
    Read a CSV file and return a list of SMILES strings.
    """
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in CSV file.")
    return df[smiles_col].tolist()

def process_smi_file(file_path):
    """Read a file containing one SMILES per line."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def process_sdf_file(file_path, remove_hs=False):
    """Read an SDF file and return a list of molecule dicts."""
    supplier = Chem.SDMolSupplier(file_path, removeHs=False)
    mols = []
    for mol in supplier:
        if mol is None:
            continue
        if remove_hs:
            mol = Chem.RemoveHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        conf = mol.GetConformer()
        coords = conf.GetPositions().astype(np.float32)
        smi = Chem.MolToSmiles(mol)
        mols.append({"atoms": atoms, "coordinates": coords, "smi": smi})
    return mols

def preprocess_dataset(data, lmdb_path, data_type='smi', smiles_col='smi',
                       num_conf=10, add_2d=True, remove_hs=False):
    """Preprocess various dataset formats into an LMDB file.

    Args:
        data: Input data; can be path or list depending on ``data_type``.
        lmdb_path: Path to the output LMDB file.
        data_type: Format of the input data. Supported values are
            ``'smi'``/``'txt'`` for a file with one SMILES per line,
            ``'csv'`` for CSV files, ``'sdf'`` for SDF molecule files,
            ``'list'`` for a Python list of SMILES strings.
        smiles_col: Column name used when ``data_type='csv'``.
    """
    logger.info(f"Preprocessing data of type '{data_type}' to {lmdb_path}")
    if data_type in ['smi', 'txt']:
        smi_list = process_smi_file(data)
        logger.info(f"Loaded {len(smi_list)} SMILES from file")
        return write_to_lmdb(
            lmdb_path,
            smi_list,
            num_conf=num_conf,
            add_2d=add_2d,
            remove_hs=remove_hs,
        )
    elif data_type == 'csv':
        smi_list = process_csv(data, smiles_col=smiles_col)
        logger.info(f"Loaded {len(smi_list)} SMILES from CSV")
        return write_to_lmdb(
            lmdb_path,
            smi_list,
            num_conf=num_conf,
            add_2d=add_2d,
            remove_hs=remove_hs,
        )
    elif data_type == 'sdf':
        mols = process_sdf_file(data, remove_hs=remove_hs)
        logger.info(f"Loaded {len(mols)} molecules from SDF")
        return write_dicts_to_lmdb(lmdb_path, mols)
    elif data_type == 'list':
        smi_list = list(data)
        logger.info(f"Loaded {len(smi_list)} SMILES from list")
        return write_to_lmdb(
            lmdb_path,
            smi_list,
            num_conf=num_conf,
            add_2d=add_2d,
            remove_hs=remove_hs,
        )
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")


