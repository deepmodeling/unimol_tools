from .configuration_unimol import UnimolConfig
from .modeling_unimol import (
    UnimolForMaskedLM,
    UnimolForSequenceClassification,
    UnimolModel,
    UnimolTrainModel,
)
from .tokenization_unimol import UnimolTokenizer
from .data_collator import UnimolDataCollator
from .dataset import UnimolSmilesDataset
from .trainer import MolPredictHF, MolTrainHF


def register_unimol_auto_classes():
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    def safe_register(register_fn, *args, **kwargs):
        try:
            register_fn(*args, **kwargs, exist_ok=True)
        except TypeError:
            try:
                register_fn(*args, **kwargs)
            except ValueError:
                pass
        except ValueError:
            pass

    safe_register(AutoConfig.register, UnimolConfig.model_type, UnimolConfig)
    safe_register(
        AutoTokenizer.register,
        UnimolConfig,
        slow_tokenizer_class=UnimolTokenizer,
    )
    safe_register(AutoModel.register, UnimolConfig, UnimolModel)
    safe_register(AutoModelForMaskedLM.register, UnimolConfig, UnimolForMaskedLM)
    safe_register(
        AutoModelForSequenceClassification.register,
        UnimolConfig,
        UnimolForSequenceClassification,
    )


register_unimol_auto_classes()

__all__ = [
    "UnimolConfig",
    "UnimolModel",
    "UnimolForMaskedLM",
    "UnimolForSequenceClassification",
    "UnimolTrainModel",
    "UnimolTokenizer",
    "UnimolDataCollator",
    "UnimolSmilesDataset",
    "MolTrainHF",
    "MolPredictHF",
    "register_unimol_auto_classes",
]
