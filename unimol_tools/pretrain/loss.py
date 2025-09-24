import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import metrics


class UniMolLoss(nn.Module):
    """Pretraining loss for Uni-Mol model.

    This implementation mirrors the original Uni-Mol loss which combines
    token, coordinate and distance objectives together with several
    regularization terms.
    """

    def __init__(
        self,
        dictionary,
        masked_token_loss=1,
        masked_coord_loss=5,
        masked_dist_loss=10,
        x_norm_loss=0.01,
        delta_pair_repr_norm_loss=0.01,
        # Statistics computed from the original Uni-Mol dataset:
        # https://github.com/dptech-corp/Uni-Mol/blob/main/unimol/unimol/losses/unimol.py
        dist_mean=6.312581655060595,
        dist_std=3.3899264663911888,
    ):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.masked_token_loss = masked_token_loss
        self.masked_coord_loss = masked_coord_loss
        self.masked_dist_loss = masked_dist_loss
        self.x_norm_loss = x_norm_loss
        self.delta_pair_repr_norm_loss = delta_pair_repr_norm_loss
        # statistics used for distance normalization
        self.dist_mean = dist_mean
        self.dist_std = dist_std

    def forward(self, model, net_input, net_target):
        tgt_tokens = net_target["tgt_tokens"]
        tgt_coordinates = net_target["tgt_coordinates"]
        tgt_distance = net_target["tgt_distance"]
        masked_tokens = tgt_tokens.ne(self.padding_idx)

        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = model(**net_input, encoder_masked_tokens=masked_tokens)

        target = tgt_tokens
        if masked_tokens is not None:
            target = target[masked_tokens]

        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred = logits_encoder.argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = masked_tokens.long().sum()

        loss = masked_token_loss * self.masked_token_loss

        logging_output = {
            "sample_size": 1,
            "bsz": tgt_tokens.size(0),
            "seq_len": tgt_tokens.size(0) * tgt_tokens.size(1),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
        }

        if encoder_coord is not None:
            coord_target = tgt_coordinates
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.masked_coord_loss
            logging_output["masked_coord_loss"] = masked_coord_loss.data

        if encoder_distance is not None:
            masked_dist_loss = self.cal_dist_loss(
                encoder_distance,
                tgt_distance,
                net_input["src_tokens"],
                masked_tokens,
                normalize=True,
            )
            loss = loss + masked_dist_loss * self.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

        if self.x_norm_loss > 0 and x_norm is not None:
            loss = loss + self.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        if (
            self.delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    def cal_dist_loss(
        self,
        encoder_distance,
        tgt_distance,
        src_tokens,
        masked_tokens,
        normalize=False,
    ):
        dist_masked_tokens = masked_tokens
        masked_distance = encoder_distance[dist_masked_tokens, :]
        masked_distance_target = tgt_distance[dist_masked_tokens]

        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)
        masked_src_tokens = src_tokens.ne(self.padding_idx)
        masked_src_tokens_expanded = torch.repeat_interleave(
            masked_src_tokens, nb_masked_tokens, dim=0
        )

        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std

        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss
    
    @staticmethod
    def logging_outputs_can_be_summed(is_train: bool) -> bool:
        """Indicate logging outputs are safe to sum across workers."""
        return True

    @staticmethod
    def reduce_metrics(logging_outputs, split="train"):
        """Aggregate logging outputs using unicore.metrics"""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)

        result = {}
        if sample_size > 0:
            result["loss"] = loss_sum / sample_size
            metrics.log_scalar("loss", result["loss"], sample_size, round=3)
        if bsz > 0:
            result["bsz"] = bsz
            metrics.log_scalar("bsz", bsz, round=1)
            metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(
            log.get("masked_token_loss", 0) for log in logging_outputs
        )
        if sample_size > 0:
            result["masked_token_loss"] = masked_loss / sample_size
            metrics.log_scalar(
                "masked_token_loss",
                result["masked_token_loss"],
                sample_size,
                round=3,
            )

        masked_hit = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        )
        masked_cnt = sum(
            log.get("masked_token_cnt", 0) for log in logging_outputs
        )
        if masked_cnt > 0:
            masked_acc = masked_hit / masked_cnt
            result["masked_acc"] = masked_acc
            metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0 and sample_size > 0:
            result["masked_coord_loss"] = masked_coord_loss / sample_size
            metrics.log_scalar(
                "masked_coord_loss",
                result["masked_coord_loss"],
                sample_size,
                round=3,
            )

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0 and sample_size > 0:
            result["masked_dist_loss"] = masked_dist_loss / sample_size
            metrics.log_scalar(
                "masked_dist_loss",
                result["masked_dist_loss"],
                sample_size,
                round=3,
            )

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0 and sample_size > 0:
            result["x_norm_loss"] = x_norm_loss / sample_size
            metrics.log_scalar(
                "x_norm_loss", result["x_norm_loss"], sample_size, round=3
            )

        delta_pair_repr_norm_loss = sum(
            log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if delta_pair_repr_norm_loss > 0 and sample_size > 0:
            result["delta_pair_repr_norm_loss"] = (
                delta_pair_repr_norm_loss / sample_size
            )
            metrics.log_scalar(
                "delta_pair_repr_norm_loss",
                result["delta_pair_repr_norm_loss"],
                sample_size,
                round=3,
            )

        return result
