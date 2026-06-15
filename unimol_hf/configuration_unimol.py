import os

from transformers import PretrainedConfig

from unimol_tools.weights import get_weight_dir


class UnimolConfig(PretrainedConfig):
    model_type = "unimol"

    def __init__(
        self,
        vocab_size=31,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=15,
        num_attention_heads=64,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        activation_fn="gelu",
        pooler_activation_fn="tanh",
        pooler_dropout=0.2,
        data_type="molecule",
        remove_hs=False,
        weight_name=None,
        dict_name=None,
        pretrained_model_path=None,
        pretrained_dict_path=None,
        num_labels=1,
        problem_type="regression",
        # transformers may overwrite problem_type to None when num_labels is set;
        # keep regression as the default downstream setting.
        conformer_seed=42,
        conformer_mode="fast",
        pad_token_id=2,
        bos_token_id=0,
        eos_token_id=3,
        unk_token_id=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.activation_fn = activation_fn
        self.pooler_activation_fn = pooler_activation_fn
        self.pooler_dropout = pooler_dropout
        self.data_type = data_type
        self.remove_hs = remove_hs
        self.weight_name = weight_name or self._default_weight_name(data_type, remove_hs)
        self.dict_name = dict_name or self._default_dict_name(data_type)
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_dict_path = pretrained_dict_path
        self.conformer_seed = conformer_seed
        self.conformer_mode = conformer_mode
        auto_map = kwargs.pop(
            "auto_map",
            {
                "AutoConfig": "unimol_hf.configuration_unimol.UnimolConfig",
                "AutoTokenizer": ["unimol_hf.tokenization_unimol.UnimolTokenizer", None],
                "AutoModel": "unimol_hf.modeling_unimol.UnimolModel",
                "AutoModelForMaskedLM": "unimol_hf.modeling_unimol.UnimolForMaskedLM",
                "AutoModelForSequenceClassification": "unimol_hf.modeling_unimol.UnimolForSequenceClassification",
            },
        )
        architectures = kwargs.pop("architectures", None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            num_labels=num_labels,
            **kwargs,
        )
        self.auto_map = auto_map
        self.architectures = architectures
        if problem_type is not None:
            self.problem_type = problem_type
        elif self.problem_type is None and self.num_labels == 1:
            self.problem_type = "regression"

    def resolve_weight_path(self, pretrained_model_name_or_path=None):
        if self.pretrained_model_path:
            return self._resolve_path(self.pretrained_model_path, pretrained_model_name_or_path)
        base = pretrained_model_name_or_path or get_weight_dir()
        if os.path.isfile(os.path.join(base, self.weight_name)):
            return os.path.join(base, self.weight_name)
        return os.path.join(get_weight_dir(), self.weight_name)

    def resolve_dict_path(self, pretrained_model_name_or_path=None):
        if self.pretrained_dict_path:
            return self._resolve_path(self.pretrained_dict_path, pretrained_model_name_or_path)
        base = pretrained_model_name_or_path or get_weight_dir()
        if os.path.isfile(os.path.join(base, self.dict_name)):
            return os.path.join(base, self.dict_name)
        return os.path.join(get_weight_dir(), self.dict_name)

    def _default_weight_name(self, data_type, remove_hs):
        if data_type == "molecule":
            return "mol_pre_no_h_220816.pt" if remove_hs else "mol_pre_all_h_220816.pt"
        defaults = {
            "protein": "poc_pre_220816.pt",
            "crystal": "mp_all_h_230313.pt",
            "oled": "oled_pre_no_h_230101.pt",
            "pocket": "pocket_pre_220816.pt",
        }
        return defaults.get(data_type, "mol_pre_all_h_220816.pt")

    def _default_dict_name(self, data_type):
        defaults = {
            "protein": "poc.dict.txt",
            "crystal": "mp.dict.txt",
            "oled": "oled.dict.txt",
            "pocket": "dict_coarse.txt",
        }
        return defaults.get(data_type, "mol.dict.txt")

    def _resolve_path(self, path, pretrained_model_name_or_path=None):
        if os.path.isabs(path):
            return path
        candidates = []
        if pretrained_model_name_or_path:
            candidates.append(os.path.join(pretrained_model_name_or_path, path))
        candidates.extend(
            [
                path,
                os.path.join(get_weight_dir(), os.path.basename(path)),
            ]
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return path
