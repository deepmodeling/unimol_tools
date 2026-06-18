import numpy as np
import pytest

from unimol_tools import MolPredict, MolTrain
from unimol_tools.models import unimol as unimol_model_module
from unimol_tools.models import unimolv2 as unimolv2_model_module


pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings(
        "ignore:Precision is ill-defined and being set to 0.0 due to no predicted samples:sklearn.exceptions.UndefinedMetricWarning"
    ),
]


def write_tiny_dictionary(tmp_path):
    dict_path = tmp_path / "tiny.dict.txt"
    dict_path.write_text("C 10\nH 10\nO 10\nN 10\n", encoding="utf-8")
    return str(dict_path)


def fake_pipeline_data(task):
    target = (
        [0.1, 0.2, 0.0, 0.3, 0.15, 0.25]
        if task == "regression"
        else [0, 1, 0, 1, 0, 1]
    )
    return {
        "atoms": [
            ["C", "O"],
            ["C", "C"],
            ["N", "C"],
            ["O", "C"],
            ["C", "N"],
            ["C", "C", "O"],
        ],
        "coordinates": [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.1, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.2, 0.0]],
        ],
        "target": target,
    }


def use_tiny_unimol_architectures(monkeypatch):
    original_v1_architecture = unimol_model_module.molecule_architecture
    original_v2_architecture = unimolv2_model_module.molecule_architecture

    def tiny_v1_architecture():
        args = original_v1_architecture()
        args.encoder_layers = 1
        args.encoder_embed_dim = 16
        args.encoder_ffn_embed_dim = 32
        args.encoder_attention_heads = 4
        args.dropout = 0.0
        args.emb_dropout = 0.0
        args.attention_dropout = 0.0
        args.max_seq_len = 32
        return args

    def tiny_v2_architecture(model_size="84m"):
        args = original_v2_architecture(model_size)
        args.num_encoder_layers = 1
        args.encoder_embed_dim = 16
        args.num_attention_heads = 4
        args.encoder_attention_heads = 4
        args.ffn_embedding_dim = 32
        args.pair_embed_dim = 16
        args.pair_hidden_dim = 8
        args.dropout = 0.0
        args.attention_dropout = 0.0
        args.pair_dropout = 0.0
        return args

    monkeypatch.setattr(
        unimol_model_module,
        "molecule_architecture",
        tiny_v1_architecture,
    )
    monkeypatch.setattr(
        unimolv2_model_module,
        "molecule_architecture",
        tiny_v2_architecture,
    )


@pytest.mark.parametrize(
    ("model_name", "task", "metrics"),
    [
        ("unimolv1", "regression", "mae"),
        ("unimolv1", "classification", "acc"),
        ("unimolv2", "regression", "mae"),
        ("unimolv2", "classification", "acc"),
    ],
)
def test_random_init_train_predict_pipeline(tmp_path, monkeypatch, model_name, task, metrics):
    use_tiny_unimol_architectures(monkeypatch)
    exp_dir = tmp_path / f"exp_{model_name}_{task}"
    dict_path = write_tiny_dictionary(tmp_path)
    train_data = fake_pipeline_data(task)
    test_data = {
        "atoms": train_data["atoms"][:2],
        "coordinates": train_data["coordinates"][:2],
    }

    trainer = MolTrain(
        task=task,
        data_type="molecule",
        epochs=1,
        batch_size=2,
        early_stopping=1,
        kfold=2,
        split="random",
        metrics=metrics,
        target_normalize="none",
        use_cuda=False,
        use_amp=False,
        use_ddp=False,
        model_name=model_name,
        pretrained_dict_path=dict_path,
        load_pretrained=False,
        conf_cache_level=0,
        save_path=str(exp_dir),
    )

    trainer.fit(train_data)
    preds = MolPredict(load_model=str(exp_dir)).predict(test_data)

    assert preds.shape == (len(test_data["atoms"]), 1)
    assert np.isfinite(preds).all()
    assert (exp_dir / "model_0.pth").exists()
