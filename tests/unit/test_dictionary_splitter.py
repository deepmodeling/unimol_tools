import io

import numpy as np
import pytest

from unimol_tools.data.dictionary import Dictionary
from unimol_tools.data.split import Splitter


def test_dictionary_initializes_special_tokens():
    dictionary = Dictionary(extra_special_symbols=["[MASK]"])

    assert dictionary.bos() == dictionary.index("[CLS]")
    assert dictionary.pad() == dictionary.index("[PAD]")
    assert dictionary.eos() == dictionary.index("[SEP]")
    assert dictionary.unk() == dictionary.index("[UNK]")
    assert dictionary.index("missing") == dictionary.unk()
    assert dictionary.index("[MASK]") in dictionary.special_index()


def test_dictionary_add_symbol_counts_and_overwrite():
    dictionary = Dictionary()
    idx = dictionary.add_symbol("C", n=2)

    assert dictionary.add_symbol("C", n=3) == idx
    assert dictionary.count[idx] == 5

    new_idx = dictionary.add_symbol("C", n=7, overwrite=True)
    assert new_idx != idx
    assert dictionary.index("C") == new_idx
    assert dictionary.count[new_idx] == 7


def test_dictionary_loads_file_like_and_detects_bad_format():
    dictionary = Dictionary()

    dictionary.add_from_file(io.StringIO("C 4\nN 2\n"))

    assert dictionary.index("C") != dictionary.unk()
    assert dictionary.count[dictionary.index("N")] == 2

    with pytest.raises(ValueError, match="Incorrect dictionary format"):
        dictionary.add_from_file(io.StringIO("bad count\n"))


def test_dictionary_from_list_vectorizes_indices():
    dictionary = Dictionary.from_list(["C", "O"])
    values = np.array(["C", "missing", "O"])

    result = dictionary.vec_index(values)

    assert result.tolist() == [
        dictionary.index("C"),
        dictionary.unk(),
        dictionary.index("O"),
    ]


def test_splitter_random_is_reproducible_and_exhaustive():
    smiles = np.array(["C", "CC", "CCC", "O", "N", "CO"])

    split_a = Splitter(method="random", kfold=3, seed=7).split(smiles)
    split_b = Splitter(method="random", kfold=3, seed=7).split(smiles)

    assert [(tr.tolist(), te.tolist()) for tr, te in split_a] == [
        (tr.tolist(), te.tolist()) for tr, te in split_b
    ]
    assert sorted(np.concatenate([test_idx for _, test_idx in split_a]).tolist()) == list(
        range(len(smiles))
    )


def test_splitter_select_uses_each_group_as_test_fold():
    smiles = np.array(["C", "CC", "CCC", "O"])
    group = np.array(["a", "a", "b", "b"])

    folds = Splitter(method="select", kfold=2).split(smiles, group=group)

    assert len(folds) == 2
    assert [test_idx.tolist() for _, test_idx in folds] == [[0, 1], [2, 3]]
    for train_idx, test_idx in folds:
        assert not set(train_idx).intersection(set(test_idx))


def test_splitter_single_fold_returns_all_train_indices():
    smiles = np.array(["C", "CC"])

    folds = Splitter(method="random", kfold=1).split(smiles)

    assert len(folds) == 1
    train_idx, test_idx = folds[0]
    assert train_idx.tolist() == [0, 1]
    assert test_idx == ()


def test_splitter_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown splitter method"):
        Splitter(method="unknown", kfold=2)
