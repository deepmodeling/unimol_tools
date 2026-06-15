from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from unimol_tools.utils import pad_1d_tokens, pad_2d, pad_coords


class UnimolDataCollator:
    """Batch UniMol features using the same padding rules as unimol_tools."""

    def __init__(
        self,
        pad_token_id: int = 2,
        device: Optional[Union[str, torch.device]] = None,
        problem_type: Optional[str] = None,
    ):
        self.pad_token_id = pad_token_id
        self.device = torch.device(device) if device is not None else None
        self.problem_type = problem_type

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_1d_tokens(
            [torch.tensor(f["input_ids"]).long() for f in features],
            pad_idx=self.pad_token_id,
        )
        dist_mat = pad_2d(
            [torch.tensor(f["dist_mat"]).float() for f in features],
            pad_idx=0.0,
        )
        edge_ids = pad_2d(
            [torch.tensor(f["edge_ids"]).long() for f in features],
            pad_idx=self.pad_token_id,
        )
        coords = pad_coords(
            [torch.tensor(f["coords"]).float() for f in features],
            pad_idx=0.0,
        )

        batch = {
            "input_ids": input_ids,
            "dist_mat": dist_mat,
            "edge_ids": edge_ids,
            "coords": coords,
        }
        label_key = "labels" if "labels" in features[0] else "label" if "label" in features[0] else None
        if label_key is not None:
            label_values = [f[label_key] for f in features]
            dtype = (
                torch.long
                if self.problem_type == "single_label_classification"
                else torch.float32
            )
            batch["labels"] = torch.tensor(
                label_values,
                dtype=dtype,
            )

        if self.device is not None:
            return {k: v.to(self.device) for k, v in batch.items()}
        return batch
