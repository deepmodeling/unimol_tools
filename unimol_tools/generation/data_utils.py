
import os
import pickle
import lmdb
import multiprocessing as mp
from unimol_tools.data import Dictionary
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

import re

def randomize_smiles(smiles):
    """
    Randomize a SMILES string to augment data.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    except:
        return smiles

def collate_tokens(values, pad_idx, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == dst[0]
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

class SmilesTokenizer:
    def __init__(self, vae_dict, encoder_dict=None, max_len=256):
        self.vae_dict = vae_dict
        self.encoder_dict = encoder_dict
        self.max_len = max_len
        smi_regex = r"(\[[^\]]+\]|Br?|Cl?|[NnCcOoSsPpFfI]|\(|\)|\.|=|#|\\|\/|:|~|@|\?|>|\*|\+|-|%\d\d|\d)"
        self.specie_re = re.compile(smi_regex)

    # def tokenize(self, smiles):
    #     return self.specie_re.findall(smiles)
    def tokenize(self, smiles):
        tokens = []
        i = 0
        L = len(smiles)

        while i < L:
            ch = smiles[i]

            # ---------- bracket ----------
            if ch == '[':
                tokens.append('[')
                i += 1

                # 元素（1或2字符）
                if i + 1 < L and smiles[i:i+2] in self.encoder_dict.symbols:
                    tokens.append(smiles[i:i+2])
                    i += 2
                else:
                    tokens.append(smiles[i])
                    i += 1

                # 解析 bracket 内
                while i < L and smiles[i] != ']':
                    ch = smiles[i]

                    # 手性
                    if smiles[i:i+2] == '@@':
                        tokens.append('@@')
                        i += 2
                    elif ch == '@':
                        tokens.append('@')
                        i += 1

                    # H
                    elif ch == 'H':
                        j = i + 1
                        while j < L and smiles[j].isdigit():
                            j += 1
                        tokens.append(smiles[i:j])  # H, H2, H3
                        i = j

                    # 电荷
                    elif ch in '+-':
                        j = i + 1
                        while j < L and smiles[j].isdigit():
                            j += 1
                        tokens.append(smiles[i:j])  # +, -2
                        i = j

                    else:
                        tokens.append(ch)
                        i += 1

                tokens.append(']')
                i += 1
                continue

            # ---------- 两字符原子 ----------
            if i + 1 < L and smiles[i:i+2] in self.encoder_dict.symbols:
                tokens.append(smiles[i:i+2])
                i += 2
                continue

            # ---------- 普通字符 ----------
            tokens.append(ch)
            i += 1

        return tokens
    def encode(self, smiles):
        tokens = self.tokenize(smiles)
        tokens = tokens[:self.max_len-2]  # Reserve space for BOS/EOS
        return [self.vae_dict.index(t) for t in tokens]

# -------- Worker --------
def _vae_dict_worker(worker_id, num_workers, lmdb_path, encoder_dict, log_interval=50000):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        max_readers=1
    )

    txn = env.begin()
    cursor = txn.cursor()

    tokenizer = SmilesTokenizer(None, encoder_dict)

    smi_tokens = set()
    local_count = 0

    for i, (_, data) in enumerate(cursor):
        if i % num_workers != worker_id:
            continue

        if data is None:
            continue

        item = pickle.loads(data)

        smi = item.get("smi")
        if smi:
            smi_str = smi.decode() if isinstance(smi, bytes) else smi
            smi_tokens.update(tokenizer.tokenize(smi_str))

        local_count += 1

        # ⭐ 日志输出（关键）
        if local_count % log_interval == 0:
            print(f"[Worker {worker_id}] processed {local_count} samples", flush=True)

    print(f"[Worker {worker_id}] DONE. Total processed: {local_count}", flush=True)

    env.close()
    return smi_tokens


# -------- Main --------
def build_vae_dictionary(
    lmdb_path,
    encoder_dict,
    save_path=None,
    num_workers=8
):
    print("🚀 Building VAE Dictionary (log mode)...")

    # -------- 获取样本数 --------
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    total_samples = env.begin().stat()["entries"]
    env.close()

    print(f"Total samples: {total_samples}")
    print(f"Num workers: {num_workers}")

    # -------- 启动多进程 --------
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(
            _vae_dict_worker,
            [(wid, num_workers, lmdb_path, encoder_dict) for wid in range(num_workers)]
        )

    # -------- 汇总 token --------
    print("Merging tokens...")

    all_smi_tokens = set()
    for s in results:
        all_smi_tokens.update(s)

    # -------- 构建词表 --------
    base_symbols = list(encoder_dict.symbols)

    grammar_fixed = ['=', '#', '(', ')', '.', '/', '\\', '+', '-', ':']
    new_tokens = all_smi_tokens.union(set(grammar_fixed))

    unique_new_tokens = sorted(list(new_tokens - set(base_symbols)))
    final_dictionary = base_symbols + unique_new_tokens

    # -------- 保存 --------
    if save_path is None:
        save_path = os.path.join(os.path.dirname(lmdb_path), "vae_dictionary.txt")

    with open(save_path, "w") as f:
        for token in final_dictionary:
            f.write(token + "\n")

    print("\n✅ Done!")
    print(f"Total tokens: {len(final_dictionary)}")
    print(f"New tokens added: {len(final_dictionary) - len(base_symbols)}")

    return Dictionary.from_list(final_dictionary)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build VAE Dictionary from LMDB")
    parser.add_argument("--lmdb_path", type=str, help="Path to the LMDB dataset")
    parser.add_argument("--encoder_dict_path", type=str, help="Path to the encoder dictionary (e.g., unimol_dictionary.txt)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the vae_dictionary.txt")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for processing LMDB")
    
    args = parser.parse_args()
    
    args.lmdb_path = '/internfs/zsj/ligands/train.lmdb'
    args.encoder_dict_path = '/internfs/zsj/ligands/dictionary.txt'
    args.save_path = '/internfs/zsj/ligands/vae_dict.txt'
    args.num_workers = 16
    # args.lmdb_path = 'examples/pretrain/tdc_ames.lmdb'
    # args.encoder_dict_path = 'examples/pretrain/tdc_ames_dict.txt'
    # args.save_path = 'examples/pretrain/vae_dict.txt'
    # args.num_workers = 16


    # Load encoder dictionary
    encoder_dict = Dictionary.load(args.encoder_dict_path)
    
    # Build VAE dictionary
    vae_dict = build_vae_dictionary(args.lmdb_path, encoder_dict, args.save_path, args.num_workers)