import ast
import inspect

import numpy as np
import pytest

from unimol_tools.data import conformer
from unimol_tools.data.conformer import (
    ConformerGen,
    UniMolV2Feature,
    _imap_with_optional_timeout,
)


def write_test_dictionary(tmp_path):
    dict_path = tmp_path / "test.dict.txt"
    dict_path.write_text("C 10\nH 10\nO 10\nN 10\n", encoding="utf-8")
    return str(dict_path)


class FakeIterator:
    def __init__(self, values):
        self.values = iter(values)
        self.timeouts = []

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def next(self, timeout=None):
        self.timeouts.append(timeout)
        return next(self.values)


class FakePool:
    def __init__(self, values):
        self.iterator = FakeIterator(values)
        self.imap_args = None
        self.entered = False
        self.exited = False
        self.exit_exc_type = None

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        self.exit_exc_type = exc_type
        return False

    def imap(self, func, items):
        self.imap_args = (func, list(items))
        return self.iterator


def unimol_feature(coord_value=1.0):
    return {
        "src_tokens": np.array([0, 1, 2]),
        "src_distance": np.ones((3, 3), dtype=np.float32),
        "src_coord": np.full((3, 3), coord_value, dtype=np.float32),
        "src_edge_type": np.ones((3, 3), dtype=int),
    }


def unimolv2_feature(coord_value=1.0):
    return {
        "src_tokens": [6],
        "src_coord": np.full((1, 3), coord_value, dtype=np.float32),
        "atom_feat": np.ones((1, 8), dtype=np.int32),
        "atom_mask": np.ones(1, dtype=np.int64),
        "edge_feat": np.ones((1, 1, 3), dtype=np.int32),
        "shortest_path": np.ones((1, 1), dtype=np.int32),
        "degree": np.ones(1, dtype=np.int32),
        "pair_type": np.ones((1, 1, 2), dtype=np.int32),
        "attn_bias": np.zeros((2, 2), dtype=np.float32),
    }


def install_fake_pool(monkeypatch, values):
    created = []

    def pool_factory(*args, **kwargs):
        pool = FakePool(values)
        pool.args = args
        pool.kwargs = kwargs
        created.append(pool)
        return pool

    monkeypatch.setattr(conformer, "Pool", pool_factory)
    return created


def test_imap_with_optional_timeout_passes_timeout_to_iterator():
    pool = FakePool(["feat-a", "feat-b"])
    func = object()
    items = ["C", "CC"]

    results = _imap_with_optional_timeout(pool, func, items, timeout=0.5)

    assert results == ["feat-a", "feat-b"]
    assert pool.imap_args == (func, items)
    assert pool.iterator.timeouts == [0.5, 0.5]


def test_imap_without_timeout_consumes_iterator_normally():
    pool = FakePool(["feat-a", "feat-b"])

    results = _imap_with_optional_timeout(pool, object(), ["C", "CC"])

    assert results == ["feat-a", "feat-b"]
    assert pool.iterator.timeouts == []


def test_imap_timeout_error_propagates():
    class TimeoutIterator:
        def next(self, timeout=None):
            raise TimeoutError("worker timed out")

    class TimeoutPool:
        def imap(self, func, items):
            return TimeoutIterator()

    with pytest.raises(TimeoutError, match="worker timed out"):
        _imap_with_optional_timeout(TimeoutPool(), object(), ["C"], timeout=0.01)


def test_conformer_transform_uses_pool_context_and_timeout(monkeypatch, tmp_path):
    smiles = ["C", "CC"]
    created = install_fake_pool(
        monkeypatch,
        [(unimol_feature(), None), (unimol_feature(), None)],
    )
    gen = ConformerGen(
        multi_process=True,
        conformer_timeout=0.25,
        pretrained_dict_path=write_test_dictionary(tmp_path),
    )

    inputs, mols = gen.transform(smiles)

    pool = created[0]
    assert pool.entered is True
    assert pool.exited is True
    assert pool.iterator.timeouts == [0.25, 0.25]
    assert pool.imap_args[1] == smiles
    assert len(inputs) == len(smiles)
    assert mols == [None, None]


def test_unimolv2_transform_uses_pool_context_and_timeout(monkeypatch):
    smiles = ["C", "CC"]
    created = install_fake_pool(
        monkeypatch,
        [(unimolv2_feature(), None), (unimolv2_feature(), None)],
    )
    gen = UniMolV2Feature(multi_process=True, conformer_timeout=0.5)

    inputs, mols = gen.transform(smiles)

    pool = created[0]
    assert pool.entered is True
    assert pool.exited is True
    assert pool.iterator.timeouts == [0.5, 0.5]
    assert pool.imap_args[1] == smiles
    assert len(inputs) == len(smiles)
    assert mols == [None, None]


def test_pool_context_manager_exits_when_timeout_raises(monkeypatch, tmp_path):
    created = []

    class TimeoutIterator:
        def next(self, timeout=None):
            raise TimeoutError("worker timed out")

    class TimeoutPool(FakePool):
        def __init__(self):
            super().__init__([])
            self.iterator = TimeoutIterator()

    def pool_factory(*args, **kwargs):
        pool = TimeoutPool()
        created.append(pool)
        return pool

    monkeypatch.setattr(conformer, "Pool", pool_factory)
    gen = ConformerGen(
        multi_process=True,
        conformer_timeout=0.01,
        pretrained_dict_path=write_test_dictionary(tmp_path),
    )

    with pytest.raises(TimeoutError, match="worker timed out"):
        gen.transform(["C"])

    assert created[0].entered is True
    assert created[0].exited is True
    assert created[0].exit_exc_type is TimeoutError


def test_conformer_source_has_no_bare_except_handlers():
    source = inspect.getsource(conformer)
    tree = ast.parse(source)

    bare_excepts = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    ]

    assert bare_excepts == []
