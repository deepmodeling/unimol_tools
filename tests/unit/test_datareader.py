import pandas as pd
import numpy as np
import pytest
from unimol_tools.data.datareader import MolDataReader


def test_read_data_from_smiles_list():
    smiles = ["CCO", "C"]
    reader = MolDataReader()
    result = reader.read_data(smiles)
    assert result["smiles"] == smiles
    assert len(result["scaffolds"]) == len(smiles)
    assert result["raw_data"].shape[0] == len(smiles)


def test_read_data_from_dict_converts_atoms_and_multitarget():
    data = {
        "atoms": [6, 8],
        "coordinates": [[0, 0, 0], [1, 0, 0]],
        "target": [[1.0, 2.0]],
    }
    reader = MolDataReader()

    result = reader.read_data(data, task="multilabel_regression")

    assert result["atoms"] == [["C", "O"]]
    assert result["coordinates"] == [[[0, 0, 0], [1, 0, 0]]]
    assert result["target_cols"] == ["TARGET0", "TARGET1"]
    assert result["raw_target"] == [[1.0, 2.0]]


def test_predict_mode_fills_missing_target_columns():
    df = pd.DataFrame({"SMILES": ["C"]})
    reader = MolDataReader()

    result = reader.read_data(
        df,
        is_train=False,
        task="regression",
        target_cols="A,B",
    )

    assert result["target_cols"] == ["A", "B"]
    assert result["raw_target"] == [[-1.0, -1.0]]


def test_check_smiles_behavior():
    reader = MolDataReader()
    # invalid SMILES should return False during training when not strict
    assert reader.check_smiles("invalid", is_train=True, smi_strict=False) is False
    # invalid SMILES should raise in strict mode
    with pytest.raises(ValueError):
        reader.check_smiles("invalid", is_train=True, smi_strict=True)


def test_convert_numeric_columns():
    from rdkit import Chem
    df = pd.DataFrame({
        "ROMol": [Chem.MolFromSmiles("CCO")],
        "num": ["1"],
        "alpha": ["a"],
    })
    reader = MolDataReader()
    out = reader._convert_numeric_columns(df.copy())
    assert pd.api.types.is_numeric_dtype(out["num"])
    assert not pd.api.types.is_numeric_dtype(out["alpha"])
    assert out["ROMol"].iloc[0] == df["ROMol"].iloc[0]


def test_anomaly_clean_regression():
    df = pd.DataFrame({
        "SMILES": ["C"] * 11,
        "TARGET": [1] * 10 + [100],
    })
    reader = MolDataReader()
    cleaned = reader.anomaly_clean_regression(df, ["TARGET"])
    assert 100 not in cleaned["TARGET"].values
    assert len(cleaned) == 10
