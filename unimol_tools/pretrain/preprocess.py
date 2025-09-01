import os
import pickle
import logging

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import multiprocessing as mp

from unimol_tools.data import Dictionary
from unimol_tools.data.conformer import inner_smi2coords

logger = logging.getLogger(__name__)


def _accum_dist_stats(coords):
    """Compute sum and squared sum of pairwise distances for given coordinates."""
    if isinstance(coords, list):
        coord_list = coords
    else:
        coord_list = [coords]
    dist_sum = 0.0
    dist_sq_sum = 0.0
    dist_cnt = 0
    for c in coord_list:
        if c is None:
            continue
        arr = np.asarray(c, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        diff = arr[:, None, :] - arr[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(len(arr), k=1)
        vals = dist[iu]
        dist_sum += vals.sum()
        dist_sq_sum += (vals ** 2).sum()
        dist_cnt += vals.size
    return dist_sum, dist_sq_sum, dist_cnt


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

def _worker_process_smi(args):
    smi, idx, remove_hs, num_conf = args
    return process_smi(smi, idx, remove_hs=remove_hs, num_conf=num_conf)

def write_to_lmdb(
    lmdb_path, smi_list, num_conf=10, remove_hs=False, num_workers=1
):
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
    dist_sum_total = 0.0
    dist_sq_sum_total = 0.0
    dist_cnt_total = 0
    processed = 0
    if num_workers > 1:
        args = [(smi_list[i], i, remove_hs, num_conf) for i in range(len(smi_list))]
        with mp.Pool(num_workers) as pool:
            for inner_output in tqdm(
                pool.imap_unordered(_worker_process_smi, args), total=len(smi_list)
            ):
                if inner_output is None:
                    continue
                idx, data, dsum, dsqsum, dcnt = inner_output
                txn_write.put(str(idx).encode(), data)
                dist_sum_total += dsum
                dist_sq_sum_total += dsqsum
                dist_cnt_total += dcnt
                processed += 1
                if processed % 1000 == 0:
                    txn_write.commit()
                    txn_write = env.begin(write=True)
    else:
        for i, smi in enumerate(tqdm(smi_list, total=len(smi_list))):
            inner_output = process_smi(smi, i, remove_hs=remove_hs, num_conf=num_conf)
            if inner_output is not None:
                idx, data, dsum, dsqsum, dcnt = inner_output
                txn_write.put(str(idx).encode(), data)
                dist_sum_total += dsum
                dist_sq_sum_total += dsqsum
                dist_cnt_total += dcnt
                processed += 1
                if processed % 1000 == 0:
                    txn_write.commit()
                    txn_write = env.begin(write=True)
    logger.info(f"Processed {processed} molecules")
    txn_write.commit()
    env.close()
    dist_mean = (
        dist_sum_total / dist_cnt_total if dist_cnt_total > 0 else 0.0
    )
    dist_std = (
        np.sqrt(dist_sq_sum_total / dist_cnt_total - dist_mean ** 2)
        if dist_cnt_total > 0
        else 1.0
    )
    logger.info(
        f"Saved to LMDB: {lmdb_path} (dist_mean={dist_mean:.6f}, dist_std={dist_std:.6f})"
    )
    return lmdb_path, dist_mean, dist_std

def _worker_process_dict(args):
    idx, item = args
    data = {
        "idx": idx,
        "atoms": item["atoms"],
        "coordinates": item["coordinates"],
    }
    if "smi" in item:
        data["smi"] = item["smi"]
    dsum, dsqsum, dcnt = _accum_dist_stats(item["coordinates"])
    return idx, pickle.dumps(data), dsum, dsqsum, dcnt

def write_dicts_to_lmdb(lmdb_path, mol_list, num_workers=1):
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
    dist_sum_total = 0.0
    dist_sq_sum_total = 0.0
    dist_cnt_total = 0
    processed = 0
    if num_workers > 1:
        args = [(i, mol_list[i]) for i in range(len(mol_list))]
        with mp.Pool(num_workers) as pool:
            for inner_output in tqdm(
                pool.imap_unordered(_worker_process_dict, args),
                total=len(mol_list),
            ):
                idx, data, dsum, dsqsum, dcnt = inner_output
                txn_write.put(str(idx).encode(), data)
                dist_sum_total += dsum
                dist_sq_sum_total += dsqsum
                dist_cnt_total += dcnt
                processed += 1
                if processed % 1000 == 0:
                    txn_write.commit()
                    txn_write = env.begin(write=True)
    else:
        for i, item in enumerate(tqdm(mol_list, total=len(mol_list))):
            data = {
                "idx": i,
                "atoms": item["atoms"],
                "coordinates": item["coordinates"],
            }
            if "smi" in item:
                data["smi"] = item["smi"]
            txn_write.put(str(i).encode(), pickle.dumps(data))
            dsum, dsqsum, dcnt = _accum_dist_stats(item["coordinates"])
            dist_sum_total += dsum
            dist_sq_sum_total += dsqsum
            dist_cnt_total += dcnt
            processed += 1
            if processed % 1000 == 0:
                txn_write.commit()
                txn_write = env.begin(write=True)
    logger.info(f"Processed {processed} molecules")
    txn_write.commit()
    env.close()
    dist_mean = (
        dist_sum_total / dist_cnt_total if dist_cnt_total > 0 else 0.0
    )
    dist_std = (
        np.sqrt(dist_sq_sum_total / dist_cnt_total - dist_mean ** 2)
        if dist_cnt_total > 0
        else 1.0
    )
    logger.info(
        f"Saved to LMDB: {lmdb_path} (dist_mean={dist_mean:.6f}, dist_std={dist_std:.6f})"
    )
    return lmdb_path, dist_mean, dist_std

def process_smi(smi, idx, remove_hs=False, num_conf=10, **params):
    """Process a single SMILES string and return index and serialized data."""
    conformers = []
    for i in range(num_conf):
        atoms, coordinates, _ = inner_smi2coords(
            smi, seed=42 + i, mode="fast", remove_hs=remove_hs
        )
        conformers.append(coordinates)
    data = {
        "idx": idx,
        "atoms": atoms,
        "coordinates": conformers if num_conf > 1 else conformers[0],
        "smi": smi,
    }

    dsum, dsqsum, dcnt = _accum_dist_stats(data["coordinates"])

    return idx, pickle.dumps(data), dsum, dsqsum, dcnt

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

def count_input_data(data, data_type="csv", smiles_col="smi"):
    """Return the number of molecules in a non-LMDB dataset."""
    if data_type in ["smi", "txt"]:
        with open(data, "r") as f:
            return sum(1 for line in f if line.strip())
    elif data_type == "csv":
        df = pd.read_csv(data)
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in CSV file.")
        return df.shape[0]
    elif data_type == "sdf":
        supplier = Chem.SDMolSupplier(data, removeHs=False)
        return sum(1 for mol in supplier if mol is not None)
    elif data_type == "list":
        return len(list(data))
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

def count_lmdb_entries(lmdb_path):
    """Return the number of entries stored in an LMDB file."""
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
    length = txn.stat()["entries"]
    env.close()
    return length

def preprocess_dataset(
    data,
    lmdb_path,
    data_type="smi",
    smiles_col="smi",
    num_conf=10,
    remove_hs=False,
    num_workers=1,
):
    """Preprocess various dataset formats into an LMDB file.

    Args:
        data: Input data; can be path or list depending on ``data_type``.
        lmdb_path: Path to the output LMDB file.
        data_type: Format of the input data. Supported values are
            ``'smi'``/``'txt'`` for a file with one SMILES per line,
            ``'csv'`` for CSV files, ``'sdf'`` for SDF molecule files,
            ``'list'`` for a Python list of SMILES strings.
        smiles_col: Column name used when ``data_type='csv'``.
        num_workers: Number of worker processes used for preprocessing.
    """
    logger.info(f"Preprocessing data of type '{data_type}' to {lmdb_path}")
    if data_type in ['smi', 'txt']:
        smi_list = process_smi_file(data)
        logger.info(f"Loaded {len(smi_list)} SMILES from file")
        return write_to_lmdb(
            lmdb_path,
            smi_list,
            num_conf=num_conf,
            remove_hs=remove_hs,
            num_workers=num_workers,
        )
    elif data_type == 'csv':
        smi_list = process_csv(data, smiles_col=smiles_col)
        logger.info(f"Loaded {len(smi_list)} SMILES from CSV")
        return write_to_lmdb(
            lmdb_path,
            smi_list,
            num_conf=num_conf,
            remove_hs=remove_hs,
            num_workers=num_workers,
        )
    elif data_type == 'sdf':
        mols = process_sdf_file(data, remove_hs=remove_hs)
        logger.info(f"Loaded {len(mols)} molecules from SDF")
        return write_dicts_to_lmdb(lmdb_path, mols, num_workers=num_workers)
    elif data_type == 'list':
        smi_list = list(data)
        logger.info(f"Loaded {len(smi_list)} SMILES from list")
        return write_to_lmdb(
            lmdb_path,
            smi_list,
            num_conf=num_conf,
            remove_hs=remove_hs,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

def compute_lmdb_dist_stats(lmdb_path):
    """Compute distance mean and std from an existing LMDB dataset."""
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
    dist_sum_total = 0.0
    dist_sq_sum_total = 0.0
    dist_cnt_total = 0
    for idx in range(length):
        data = txn.get(str(idx).encode())
        if data is None:
            continue
        item = pickle.loads(data)
        dsum, dsqsum, dcnt = _accum_dist_stats(item.get("coordinates"))
        dist_sum_total += dsum
        dist_sq_sum_total += dsqsum
        dist_cnt_total += dcnt
    env.close()
    dist_mean = (
        dist_sum_total / dist_cnt_total if dist_cnt_total > 0 else 0.0
    )
    dist_std = (
        np.sqrt(dist_sq_sum_total / dist_cnt_total - dist_mean ** 2)
        if dist_cnt_total > 0
        else 1.0
    )
    logger.info(
        f"dist_mean={dist_mean:.6f}, dist_std={dist_std:.6f}"
    )
    return dist_mean, dist_std
