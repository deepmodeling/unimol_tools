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


def _dict_worker(lmdb_path, start, end):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    txn = env.begin()
    elem = set()
    for idx in range(start, end):
        data = txn.get(str(idx).encode())
        if data is None:
            continue
        item = pickle.loads(data)
        atoms = item.get("atoms")
        if atoms:
            elem.update(atoms)
    env.close()
    return elem

def build_dictionary(lmdb_path, save_path=None, num_workers=1):
    """Count element types and return a Dictionary.

    Args:
        lmdb_path: Path to LMDB dataset.
        save_path: Optional output path for the text dictionary file.
        num_workers: Number of parallel workers to use.
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
    length = txn.stat()["entries"]
    env.close()

    elements_set = set()
    if num_workers > 1:
        chunk = (length + num_workers - 1) // num_workers
        args = [
            (lmdb_path, i * chunk, min((i + 1) * chunk, length))
            for i in range(num_workers)
        ]
        with mp.Pool(num_workers) as pool:
            for s in pool.starmap(_dict_worker, args):
                elements_set.update(s)
    else:
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        txn = env.begin()
        for idx in range(length):
            data = txn.get(str(idx).encode())
            if data is None:
                continue
            item = pickle.loads(data)
            atoms = item.get("atoms")
            if atoms:
                elements_set.update(atoms)
        env.close()

    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
    dictionary = special_tokens + sorted(list(elements_set))
    if save_path is None:
        save_path = os.path.join(os.path.dirname(lmdb_path), "dictionary.txt")
    with open(save_path, "wb") as f:
        np.savetxt(f, dictionary, fmt="%s")
    return Dictionary.from_list(dictionary)

def _worker_process_smi(args):
    smi, idx, remove_hs, num_conf = args
    return process_smi(smi, idx, remove_hs=remove_hs, num_conf=num_conf)

def write_to_lmdb(
    lmdb_path, smi_iter, num_conf=10, remove_hs=False, num_workers=1, total=None
):
    logger.info(f"Writing SMILES to {lmdb_path}")

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
        ctx = mp.get_context("spawn")
        def gen():
            for idx, smi in enumerate(smi_iter):
                yield (smi, idx, remove_hs, num_conf)
        with ctx.Pool(num_workers) as pool:
            for inner_output in tqdm(
                pool.imap_unordered(_worker_process_smi, gen()), total=total
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
        for i, smi in enumerate(tqdm(smi_iter, total=total)):
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

def write_dicts_to_lmdb(lmdb_path, mol_iter, num_workers=1, total=None):
    """Write an iterable of pre-generated molecules to LMDB."""
    logger.info(f"Writing molecule dicts to {lmdb_path}")

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
        ctx = mp.get_context("spawn")
        def gen():
            for idx, item in enumerate(mol_iter):
                yield (idx, item)
        with ctx.Pool(num_workers) as pool:
            for inner_output in tqdm(
                pool.imap_unordered(_worker_process_dict, gen()),
                total=total,
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
        for i, item in enumerate(tqdm(mol_iter, total=total)):
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

def iter_smi_file(file_path):
    """Yield SMILES strings from a text file one at a time."""
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def iter_csv_file(csv_path, smiles_col="smi", chunksize=10000):
    """Yield SMILES strings from a CSV file in chunks."""
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if smiles_col not in chunk.columns:
            raise ValueError(f"Column '{smiles_col}' not found in CSV file.")
        for smi in chunk[smiles_col].astype(str):
            smi = smi.strip()
            if smi:
                yield smi

def iter_sdf_file(file_path, remove_hs=False):
    """Yield molecule dicts from an SDF file without loading everything into memory."""
    supplier = Chem.SDMolSupplier(file_path, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        if remove_hs:
            mol = Chem.RemoveHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        conf = mol.GetConformer()
        coords = conf.GetPositions().astype(np.float32)
        smi = Chem.MolToSmiles(mol)
        yield {"atoms": atoms, "coordinates": coords, "smi": smi}

def count_input_data(data, data_type="csv", smiles_col="smi"):
    """Return the number of molecules in a non-LMDB dataset."""
    if data_type in ["smi", "txt"]:
        with open(data, "r") as f:
            return sum(1 for line in f if line.strip())
    elif data_type == "csv":
        count = 0
        for chunk in pd.read_csv(data, usecols=[smiles_col], chunksize=100000):
            count += chunk.shape[0]
        return count
    elif data_type == "sdf":
        supplier = Chem.SDMolSupplier(data, removeHs=False)
        return sum(1 for mol in supplier if mol is not None)
    elif data_type == "list":
        return len(data) if hasattr(data, "__len__") else sum(1 for _ in data)
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
    if data_type in ["smi", "txt"]:
        total = count_input_data(data, data_type)
        return write_to_lmdb(
            lmdb_path,
            iter_smi_file(data),
            num_conf=num_conf,
            remove_hs=remove_hs,
            num_workers=num_workers,
            total=total,
        )
    elif data_type == "csv":
        total = count_input_data(data, "csv", smiles_col)
        return write_to_lmdb(
            lmdb_path,
            iter_csv_file(data, smiles_col=smiles_col),
            num_conf=num_conf,
            remove_hs=remove_hs,
            num_workers=num_workers,
            total=total,
        )
    elif data_type == "sdf":
        total = count_input_data(data, "sdf")
        return write_dicts_to_lmdb(
            lmdb_path,
            iter_sdf_file(data, remove_hs=remove_hs),
            num_workers=num_workers,
            total=total,
        )
    elif data_type == "list":
        total = len(data) if hasattr(data, "__len__") else None
        return write_to_lmdb(
            lmdb_path,
            iter(data),
            num_conf=num_conf,
            remove_hs=remove_hs,
            num_workers=num_workers,
            total=total,
        )
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

def _dist_worker(lmdb_path, start, end):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    txn = env.begin()
    dsum = dsqsum = dcnt = 0.0
    for idx in range(start, end):
        data = txn.get(str(idx).encode())
        if data is None:
            continue
        item = pickle.loads(data)
        a, b, c = _accum_dist_stats(item.get("coordinates"))
        dsum += a
        dsqsum += b
        dcnt += c
    env.close()
    return dsum, dsqsum, dcnt

def compute_lmdb_dist_stats(lmdb_path, num_workers=1):
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
    length = txn.stat()["entries"]
    env.close()

    dist_sum_total = 0.0
    dist_sq_sum_total = 0.0
    dist_cnt_total = 0.0
    if num_workers > 1:
        chunk = (length + num_workers - 1) // num_workers
        args = [
            (lmdb_path, i * chunk, min((i + 1) * chunk, length))
            for i in range(num_workers)
        ]
        with mp.Pool(num_workers) as pool:
            for dsum, dsqsum, dcnt in pool.starmap(_dist_worker, args):
                dist_sum_total += dsum
                dist_sq_sum_total += dsqsum
                dist_cnt_total += dcnt
    else:
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        txn = env.begin()
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

    dist_mean = dist_sum_total / dist_cnt_total if dist_cnt_total > 0 else 0.0
    dist_std = (
        np.sqrt(dist_sq_sum_total / dist_cnt_total - dist_mean ** 2)
        if dist_cnt_total > 0
        else 1.0
    )
    logger.info(f"dist_mean={dist_mean:.6f}, dist_std={dist_std:.6f}")
    return dist_mean, dist_std
