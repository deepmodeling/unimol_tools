import torch
import torch.nn as nn
import torch.nn.functional as F


class UniMolloss(nn.Module):
    def __init__(
            self, 
            dictionary,
            masked_token_loss=1,
            masked_coord_loss=5,
            masked_dist_loss=10,
            x_norm_loss=0.01,
            delta_pair_repr_norm_loss=0.01,
        ):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.masked_token_loss = masked_token_loss
        self.masked_coord_loss = masked_coord_loss
        self.masked_dist_loss = masked_dist_loss
        self.x_norm_loss = x_norm_loss
        self.delta_pair_repr_norm_loss = delta_pair_repr_norm_loss

    def forward(self, model, net_input, net_target):
        tgt_tokens = net_target['tgt_tokens']
        tgt_coordinates = net_target['tgt_coordinates']
        tgt_distance = net_target['tgt_distance']
        masked_tokens = tgt_tokens.ne(self.padding_idx)
        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = model(**net_input, encoder_masked_tokens=masked_tokens)
        
        if masked_tokens is not None:
            tgt_tokens = tgt_tokens[masked_tokens]
            # tgt_coordinates = tgt_coordinates[masked_tokens]
            # tgt_distance = tgt_distance[masked_tokens]

        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            tgt_tokens,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred = logits_encoder.argmax(dim=-1)
        masked_hit = (masked_pred == tgt_tokens).long().sum()
        masked_cnt = masked_tokens.long().sum()

        loss = masked_token_loss * self.masked_token_loss

        logging_info = {
            'masked_token_loss': masked_token_loss.data,
            'masked_token_hit': masked_hit.data,
            'masked_tokeb_cnt': masked_cnt,
        }


        if encoder_coord is not None:
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                tgt_coordinates[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss += masked_coord_loss * self.masked_coord_loss
            logging_info['masked_coord_loss'] = masked_coord_loss.data

        if encoder_distance is not None:
            masked_dist_loss = self.cal_dist_loss(
                encoder_distance, tgt_distance, masked_tokens, normalize=False
            )
            loss += masked_dist_loss * self.masked_dist_loss
            logging_info['masked_dist_loss'] = masked_dist_loss.data

        if x_norm is not None and self.x_norm_loss > 0:
            loss += self.x_norm_loss * x_norm
            logging_info['x_norm_loss'] = x_norm.data

        if delta_encoder_pair_rep_norm is not None and self.delta_pair_repr_norm_loss > 0:
            loss += self.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            logging_info['delta_pair_repr_norm_loss'] = delta_encoder_pair_rep_norm.data
            
        logging_info['loss'] = loss.data
        return loss, logging_info
    

    def cal_dist_loss(self, encoder_distance, tgt_distance, masked_tokens, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = encoder_distance[dist_masked_tokens, :]
        masked_distance_target = tgt_distance[dist_masked_tokens]
        # padding distance
        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)
        masked_src_tokens = masked_tokens # ?
        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)
        #
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