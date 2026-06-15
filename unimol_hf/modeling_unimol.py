import os
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

from unimol_tools.models.unimol import UniMolModel as UniMolBackend

from .configuration_unimol import UnimolConfig


def _as_tensor(value, dtype=None, device=None):
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if dtype is not None:
        value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device)
    return value


def _ensure_batch_dim(tensor, ndim):
    while tensor.dim() < ndim:
        tensor = tensor.unsqueeze(0)
    return tensor


def _has_transformers_weights(pretrained_model_name_or_path):
    if not isinstance(pretrained_model_name_or_path, (str, os.PathLike)):
        return False
    path = os.fspath(pretrained_model_name_or_path)
    if not os.path.isdir(path):
        return False
    filenames = {
        WEIGHTS_NAME,
        SAFE_WEIGHTS_NAME,
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }
    return any(os.path.isfile(os.path.join(path, name)) for name in filenames)


class UnimolTrainModel(nn.Module):
    """nn.Module compatible with unimol_tools Trainer / NNModel."""

    def __init__(
        self,
        output_dim=2,
        data_type="molecule",
        remove_hs=False,
        pretrained_model_path=None,
        pretrained_dict_path=None,
        pooler_dropout=0.2,
        **params,
    ):
        super().__init__()
        problem_type = (
            "single_label_classification" if output_dim > 1 else "regression"
        )
        config = UnimolConfig(
            num_labels=output_dim,
            data_type=data_type,
            remove_hs=remove_hs,
            pretrained_model_path=pretrained_model_path,
            pretrained_dict_path=pretrained_dict_path,
            pooler_dropout=pooler_dropout,
            problem_type=problem_type,
        )
        self.unimol = UniMolBackend(
            output_dim=output_dim,
            data_type=data_type,
            remove_hs=remove_hs,
            pretrained_model_path=config.resolve_weight_path(),
            pretrained_dict_path=config.resolve_dict_path(),
            pooler_dropout=pooler_dropout,
        )

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        **kwargs,
    ):
        return self.unimol(
            src_tokens,
            src_distance,
            src_coord,
            src_edge_type,
            return_repr=False,
        )

    def batch_collate_fn(self, samples):
        return self.unimol.batch_collate_fn(samples)

    def load_pretrained_weights(self, path, strict=False):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=strict)


class UnimolPreTrainedModel(PreTrainedModel):
    config_class = UnimolConfig
    base_model_prefix = "unimol"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def _init_weights(self, module):
        return

    def _prepare_model_inputs(self, input_ids, dist_mat, edge_ids, coords):
        device = self.device
        input_ids = _ensure_batch_dim(_as_tensor(input_ids, torch.long, device), 2)
        dist_mat = _ensure_batch_dim(_as_tensor(dist_mat, torch.float, device), 3)
        edge_ids = _ensure_batch_dim(_as_tensor(edge_ids, torch.long, device), 3)
        coords = _ensure_batch_dim(_as_tensor(coords, torch.float, device), 3)
        return input_ids, dist_mat, edge_ids, coords

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config, kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                **kwargs,
            )
        config.pretrained_model_path = config.resolve_weight_path(pretrained_model_name_or_path)
        config.pretrained_dict_path = config.resolve_dict_path(pretrained_model_name_or_path)
        if _has_transformers_weights(pretrained_model_name_or_path):
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                **kwargs,
            )
        return cls(config, *model_args)


class UnimolModel(UnimolPreTrainedModel):
    def __init__(self, config: UnimolConfig):
        super().__init__(config)
        weight_path = config.resolve_weight_path()
        dict_path = config.resolve_dict_path()
        self.unimol = UniMolBackend(
            output_dim=config.num_labels,
            data_type=config.data_type,
            remove_hs=config.remove_hs,
            pretrained_model_path=weight_path,
            pretrained_dict_path=dict_path,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.unimol.embed_tokens

    def _hidden_states(self, input_ids, dist_mat, edge_ids):
        padding_mask = input_ids.eq(self.unimol.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        x = self.unimol.embed_tokens(input_ids)
        n_node = dist_mat.size(-1)
        gbf_feature = self.unimol.gbf(dist_mat, edge_ids)
        gbf_result = self.unimol.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)

        encoder_rep, _, _, _, _ = self.unimol.encoder(
            x,
            padding_mask=padding_mask,
            attn_mask=graph_attn_bias,
        )
        return encoder_rep

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        dist_mat: Optional[torch.FloatTensor] = None,
        edge_ids: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if coords is None:
            raise ValueError("coords are required for UniMol forward pass.")

        input_ids, dist_mat, edge_ids, coords = self._prepare_model_inputs(
            input_ids, dist_mat, edge_ids, coords
        )
        encoder_rep = self._hidden_states(input_ids, dist_mat, edge_ids)

        if not return_dict:
            return (encoder_rep,)

        return BaseModelOutput(last_hidden_state=encoder_rep)

    def get_cls_repr(self, input_ids, dist_mat, edge_ids, coords):
        if coords is None:
            raise ValueError("coords are required for UniMol forward pass.")
        input_ids, dist_mat, edge_ids, coords = self._prepare_model_inputs(
            input_ids, dist_mat, edge_ids, coords
        )
        return self.unimol(
            input_ids,
            dist_mat,
            coords,
            edge_ids,
            return_repr=True,
            return_atomic_reprs=False,
        )

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)


class UnimolForMaskedLM(UnimolPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: UnimolConfig):
        super().__init__(config)
        weight_path = config.resolve_weight_path()
        dict_path = config.resolve_dict_path()
        self.unimol = UniMolBackend(
            output_dim=config.num_labels,
            data_type=config.data_type,
            remove_hs=config.remove_hs,
            pretrained_model_path=weight_path,
            pretrained_dict_path=dict_path,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.unimol.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.unimol.embed_tokens

    def set_input_embeddings(self, value):
        self.unimol.embed_tokens = value
        self.lm_head.weight = value.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        dist_mat: Optional[torch.FloatTensor] = None,
        edge_ids: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if coords is None:
            raise ValueError("coords are required for UniMol forward pass.")

        input_ids, dist_mat, edge_ids, coords = self._prepare_model_inputs(
            input_ids, dist_mat, edge_ids, coords
        )
        encoder_rep = UnimolModel._hidden_states(self, input_ids, dist_mat, edge_ids)
        prediction_scores = self.lm_head(encoder_rep)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.to(prediction_scores.device).view(-1),
            )

        if not return_dict:
            output = (prediction_scores,)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=None,
            attentions=None,
        )


class UnimolForSequenceClassification(UnimolPreTrainedModel):
    def __init__(self, config: UnimolConfig):
        super().__init__(config)
        weight_path = config.resolve_weight_path()
        dict_path = config.resolve_dict_path()
        self.unimol = UniMolBackend(
            output_dim=config.num_labels,
            data_type=config.data_type,
            remove_hs=config.remove_hs,
            pretrained_model_path=weight_path,
            pretrained_dict_path=dict_path,
        )
        self.num_labels = config.num_labels
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        dist_mat: Optional[torch.FloatTensor] = None,
        edge_ids: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if coords is None:
            raise ValueError("coords are required for UniMol forward pass.")

        input_ids, dist_mat, edge_ids, coords = self._prepare_model_inputs(
            input_ids, dist_mat, edge_ids, coords
        )
        logits = self.unimol(
            input_ids,
            dist_mat,
            coords,
            edge_ids,
            return_repr=False,
        )
        if logits.dim() == 3:
            cls_logits = logits[:, 0, :]
        else:
            cls_logits = logits

        loss = None
        if labels is not None:
            labels = labels.to(cls_logits.device)
            if self.config.problem_type == "regression":
                loss_fn = nn.MSELoss()
                loss = loss_fn(cls_logits.reshape(-1), labels.reshape(-1))
            elif self.config.problem_type == "single_label_classification":
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(cls_logits, labels.long())
            else:
                raise ValueError(f"Unsupported problem_type: {self.config.problem_type}")

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=cls_logits,
            hidden_states=None,
            attentions=None,
        )

    def get_repr(
        self,
        input_ids,
        dist_mat,
        edge_ids,
        coords,
        return_atomic_reprs=False,
    ):
        input_ids, dist_mat, edge_ids, coords = self._prepare_model_inputs(
            input_ids, dist_mat, edge_ids, coords
        )
        return self.unimol(
            input_ids,
            dist_mat,
            coords,
            edge_ids,
            return_repr=True,
            return_atomic_reprs=return_atomic_reprs,
        )
