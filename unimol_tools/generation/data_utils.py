
import os
import pickle
import lmdb
import multiprocessing as mp
from unimol_tools.data import Dictionary
import numpy as np
import torch
from rdkit import Chem

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
    def __init__(self, vae_dict, max_len=256):
        self.vae_dict = vae_dict
        self.max_len = max_len
        smi_regex = r"(\[[^\]]+\]|Br?|Cl?|[NnCcOoSsPpFfI]|\(|\)|\.|=|#|\\|\/|:|~|@|\?|>|\*|\+|-|%\d\d|\d)"
        self.specie_re = re.compile(smi_regex)

    def tokenize(self, smiles):
        return self.specie_re.findall(smiles)

    def encode(self, smiles):
        tokens = self.tokenize(smiles)
        tokens = tokens[:self.max_len-2]  # Reserve space for BOS/EOS
        return [self.vae_dict.index(t) for t in tokens]

def _vae_dict_worker(lmdb_path, start, end):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, max_readers=1)
    txn = env.begin()
    smi_tokens = set()
    tokenizer = SmilesTokenizer(None)
    
    for idx in range(start, end):
        data = txn.get(str(idx).encode())
        if data is None: continue
        item = pickle.loads(data)
        
        # 提取 SMILES 并切分
        smi = item.get("smiles")
        if smi:
            # 如果 smi 是字节串，先解码
            smi_str = smi.decode() if isinstance(smi, bytes) else smi
            tokens = tokenizer.tokenize(smi_str)
            smi_tokens.update(tokens)
            
    env.close()
    return smi_tokens

def build_vae_dictionary(lmdb_path, encoder_dict, save_path=None, num_workers=1):
    """
    Args:
        lmdb_path: 数据路径
        encoder_dict: 已有的 Dictionary 对象 (Uni-Mol Encoder 使用)
        save_path: 保存 vae_dict.txt 的路径
    """
    # 1. 获取 LMDB 总长度
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, max_readers=256)
    length = env.begin().stat()["entries"]
    env.close()

    # 2. 并行提取 LMDB 中的所有 SMILES Token
    all_smi_tokens = set()
    chunk = (length + num_workers - 1) // num_workers
    args = [(lmdb_path, i * chunk, min((i + 1) * chunk, length)) for i in range(num_workers)]
    
    with mp.Pool(num_workers) as pool:
        for s in pool.starmap(_vae_dict_worker, args):
            all_smi_tokens.update(s)

    # 3. 合并词表
    # 首先继承 encoder_dict 的所有符号 (保持顺序一致)
    # Dictionary 对象通常有 .symbols 属性，或者通过索引遍历
    base_symbols = list(encoder_dict.symbols)
    
    # for special in ["[BOS]", "[EOS]"]:
    #     if special not in base_symbols:
    #         base_symbols.append(special)
            
    print(base_symbols)
    # 强制补充一些即便当前数据集没出现，但未来生成可能需要的语法符号
    grammar_fixed = ['=', '#', '(', ')', '.', '/', '\\', '+', '-', ':']
    
    # 寻找所有不在 base_symbols 里的新 token
    new_tokens = all_smi_tokens.union(set(grammar_fixed))
    unique_new_tokens = sorted(list(new_tokens - set(base_symbols)))
    
    final_dictionary = base_symbols + unique_new_tokens

    if save_path is None:
        save_path = os.path.join(os.path.dirname(lmdb_path), "vae_dictionary.txt")
    
    with open(save_path, "wb") as f:
        np.savetxt(f, final_dictionary, fmt="%s")
    
    print(f"VAE Dictionary built: {len(final_dictionary)} tokens (Added {len(final_dictionary) - len(encoder_dict.symbols)} new tokens)")
    
    return Dictionary.from_list(final_dictionary)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build VAE Dictionary from LMDB")
    parser.add_argument("--lmdb_path", type=str, help="Path to the LMDB dataset")
    parser.add_argument("--encoder_dict_path", type=str, help="Path to the encoder dictionary (e.g., unimol_dictionary.txt)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the vae_dictionary.txt")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for processing LMDB")
    
    args = parser.parse_args()
    
    args.lmdb_path = 'examples/pretrain/tdc_ames.lmdb'
    args.encoder_dict_path = 'examples/pretrain/tdc_ames_dict.txt'
    args.save_path = 'examples/pretrain/vae_dict.txt'
    args.num_workers = 16

    # Load encoder dictionary
    encoder_dict = Dictionary.load(args.encoder_dict_path)
    
    # Build VAE dictionary
    vae_dict = build_vae_dictionary(args.lmdb_path, encoder_dict, args.save_path, args.num_workers)