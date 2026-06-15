from typing import Optional

from torch.utils.data import Dataset


class UnimolSmilesDataset(Dataset):
    """Map SMILES rows to UniMol tokenizer features for Transformers Trainer."""

    def __init__(
        self,
        data,
        tokenizer,
        smiles_col: str,
        target_col: Optional[str] = None,
        problem_type: str = "regression",
    ):
        self.data = data.reset_index(drop=True) if hasattr(data, "reset_index") else data
        self.tokenizer = tokenizer
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.problem_type = problem_type

    def __len__(self):
        return len(self.data)

    def _row_value(self, row, key):
        if hasattr(row, "__getitem__"):
            return row[key]
        return getattr(row, key)

    def __getitem__(self, idx):
        row = self.data.iloc[idx] if hasattr(self.data, "iloc") else self.data[idx]
        item = dict(self.tokenizer.encode(self._row_value(row, self.smiles_col)))
        if self.target_col is not None:
            label = self._row_value(row, self.target_col)
            if self.problem_type == "single_label_classification":
                label = int(label)
            else:
                label = float(label)
            item["labels"] = label
        return item
