"""Training / prediction wrappers aligned with unimol_tools MolTrain & MolPredict."""

from __future__ import annotations

import copy
import os

import joblib
import numpy as np
import pandas as pd

from unimol_tools.data import DataHub
from unimol_tools.models.nnmodel import NNModel
from unimol_tools.tasks import Trainer
from unimol_tools.utils import YamlHandler, logger

from .modeling_unimol import UnimolTrainModel


class HFNNModel(NNModel):
    """NNModel that swaps UniMolModel for UnimolTrainModel."""

    def _init_model(self, model_name, **params):
        if model_name != "unimolv1":
            return super()._init_model(model_name, **params)

        if self.task in ["regression", "multilabel_regression"]:
            params["pooler_dropout"] = 0
            logger.debug("set pooler_dropout to 0 for regression task")

        freeze_layers = params.get("freeze_layers", None)
        freeze_layers_reversed = params.get("freeze_layers_reversed", False)
        model = UnimolTrainModel(**params)
        if isinstance(freeze_layers, str):
            freeze_layers = freeze_layers.replace(" ", "").split(",")
        if isinstance(freeze_layers, list):
            for layer_name, layer_param in model.named_parameters():
                should_freeze = any(
                    layer_name.startswith(freeze_layer) for freeze_layer in freeze_layers
                )
                layer_param.requires_grad = not (freeze_layers_reversed ^ should_freeze)
        return model


class MolTrainHF:
    """Mirror of unimol_tools.MolTrain backed by unimol_hf.UnimolTrainModel."""

    def __init__(
        self,
        task="classification",
        data_type="molecule",
        epochs=10,
        learning_rate=1e-4,
        batch_size=16,
        early_stopping=5,
        metrics="none",
        split="random",
        kfold=1,
        save_path="./exp_hf",
        remove_hs=False,
        smiles_col="SMILES",
        target_cols=None,
        target_normalize="auto",
        max_norm=5.0,
        use_cuda=True,
        use_amp=True,
        use_ddp=False,
        use_gpu="all",
        model_name="unimolv1",
        pretrained_model_path=None,
        pretrained_dict_path=None,
        seed=42,
        **params,
    ):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "unimol_tools", "config", "default.yaml"
        )
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        config.task = task
        config.data_type = data_type
        config.epochs = epochs
        config.max_epochs = epochs
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.patience = early_stopping
        config.metrics = metrics
        config.split = split
        config.kfold = kfold
        config.remove_hs = remove_hs
        config.smiles_col = smiles_col
        config.target_cols = target_cols
        config.target_normalize = target_normalize
        config.max_norm = max_norm
        config.use_cuda = use_cuda
        config.use_amp = use_amp
        config.use_ddp = use_ddp
        config.use_gpu = use_gpu
        config.model_name = model_name
        config.pretrained_model_path = pretrained_model_path
        config.pretrained_dict_path = pretrained_dict_path
        config.seed = seed
        config.split_seed = seed
        config.split_method = f"{kfold}fold_{split}"
        self.save_path = save_path
        self.config = config

    def fit(self, data):
        self.datahub = DataHub(
            data=data, is_train=True, save_path=self.save_path, **self.config
        )
        self.data = self.datahub.data
        self._update_and_save_config()
        self.trainer = Trainer(save_path=self.save_path, **self.config)
        self.model = HFNNModel(self.data, self.trainer, **self.config)
        self.model.run()
        scalar = self.data["target_scaler"]
        y_pred = self.model.cv["pred"]
        y_true = np.array(self.data["target"])
        metrics = self.trainer.metrics
        if scalar is not None:
            y_pred = scalar.inverse_transform(y_pred)
            y_true = scalar.inverse_transform(y_true)
        if self.config["task"] in ["classification", "multilabel_classification"]:
            threshold = metrics.calculate_classification_threshold(y_true, y_pred)
            joblib.dump(threshold, os.path.join(self.save_path, "threshold.dat"))
        self.cv_pred = y_pred
        return self

    def _update_and_save_config(self):
        self.config["num_classes"] = self.data["num_classes"]
        self.config["target_cols"] = ",".join(self.data["target_cols"])
        self.config["split_method"] = f"{self.config['kfold']}fold_{self.config['split']}"
        os.makedirs(self.save_path, exist_ok=True)
        out_path = os.path.join(self.save_path, "config.yaml")
        self.yamlhandler.write_yaml(data=self.config, out_file_path=out_path)


class MolPredictHF:
    """Mirror of unimol_tools.MolPredict for models trained with MolTrainHF."""

    def __init__(self, load_model):
        if not load_model:
            raise ValueError("load_model is empty")
        self.load_model = load_model
        config_path = os.path.join(load_model, "config.yaml")
        self.config = YamlHandler(config_path).read_yaml()
        self.config.target_cols = self.config.target_cols.split(",")
        self.task = self.config.task
        self.target_cols = self.config.target_cols

    def predict(self, data, save_path=None, metrics="none"):
        if metrics and metrics != "none":
            self.config.metrics = metrics
        self.datahub = DataHub(
            data=data, is_train=False, save_path=self.load_model, **self.config
        )
        self.config.use_ddp = False
        self.trainer = Trainer(save_path=self.load_model, **self.config)
        self.model = HFNNModel(self.datahub.data, self.trainer, **self.config)
        self.model.evaluate(self.trainer, self.load_model)

        y_pred = self.model.cv["test_pred"]
        scalar = self.datahub.data["target_scaler"]
        if scalar is not None:
            y_pred = scalar.inverse_transform(y_pred)

        df = self.datahub.data["raw_data"].copy()
        predict_cols = ["predict_" + col for col in self.target_cols]
        if self.task in ["classification", "multilabel_classification"]:
            threshold = joblib.load(os.path.join(self.load_model, "threshold.dat"))
            prob_cols = ["prob_" + col for col in self.target_cols]
            df[prob_cols] = y_pred
            df[predict_cols] = (y_pred > threshold).astype(int)
        else:
            prob_cols = predict_cols
            df[predict_cols] = y_pred

        result_metrics = None
        if not (df[self.target_cols] == -1.0).all().all():
            result_metrics = self.trainer.metrics.cal_metric(
                df[self.target_cols].values, df[prob_cols].values
            )
            logger.info("final predict metrics score: \n{}".format(result_metrics))
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                joblib.dump(result_metrics, os.path.join(save_path, "test_metric.result"))
        return y_pred, result_metrics
