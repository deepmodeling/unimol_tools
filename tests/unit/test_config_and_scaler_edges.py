import numpy as np
import pytest
from addict import Dict

from unimol_tools.data.datascaler import TargetScaler
from unimol_tools.utils.config_handler import YamlHandler, addict2dict


def test_yaml_handler_roundtrip_and_addict_conversion(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("model:\n  name: unimol\n  layers: 2\n", encoding="utf-8")

    handler = YamlHandler(str(yaml_path))
    config = handler.read_yaml()

    assert config.model.name == "unimol"
    assert config.model.layers == 2

    out_path = tmp_path / "out.yaml"
    handler.write_yaml(Dict({"train": Dict({"epochs": 1})}), str(out_path))

    assert YamlHandler(str(out_path)).read_yaml().train.epochs == 1
    assert addict2dict(Dict({"a": Dict({"b": 1})})) == {"a": {"b": 1}}


def test_yaml_handler_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        YamlHandler(str(tmp_path / "missing.yaml"))


def test_classification_scaler_is_noop(tmp_path):
    target = np.array([[0], [1], [1]])
    scaler = TargetScaler("standard", "classification")

    scaler.fit(target, str(tmp_path))

    assert scaler.transform(target) is target
    assert scaler.inverse_transform(target) is target
    assert not (tmp_path / "target_scaler.ss").exists()


def test_none_scaler_is_noop_and_does_not_dump(tmp_path):
    target = np.array([[1.0], [2.0], [3.0]])
    scaler = TargetScaler("none", "regression")

    scaler.fit(target, str(tmp_path))

    assert scaler.transform(target) is target
    assert scaler.inverse_transform(target) is target
    assert not (tmp_path / "target_scaler.ss").exists()


def test_target_scaler_loads_existing_scaler(tmp_path):
    target = np.array([[1.0], [2.0], [3.0]])
    scaler = TargetScaler("standard", "regression")
    scaler.fit(target, str(tmp_path))

    loaded = TargetScaler("standard", "regression", load_dir=str(tmp_path))
    restored = loaded.inverse_transform(loaded.transform(target))

    assert np.allclose(restored, target)


def test_multilabel_regression_scaler_roundtrip(tmp_path):
    target = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = TargetScaler("standard", "multilabel_regression")

    scaler.fit(target, str(tmp_path))
    scaled = scaler.transform(target)
    restored = scaler.inverse_transform(scaled)

    assert np.allclose(restored, target)
    assert (tmp_path / "target_scaler.ss").exists()


def test_unknown_scaler_method_raises():
    scaler = TargetScaler("bad", "regression")

    with pytest.raises(ValueError, match="Unknown scaler method"):
        scaler.scaler_choose("bad", np.array([[1.0], [2.0]]))
