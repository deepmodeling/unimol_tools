import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.transformers import (
    TransformerEncoderWithPair,
    get_activation_fn,
    init_bert_params,
)
from ..utils import pad_1d_tokens, pad_2d, pad_coords


class UniMolModel(nn.Module):
    def __init__(self, config, dictionary):
        super(UniMolModel, self).__init__()
        self.config = config
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), config.encoder_embed_dim, padding_idx=self.padding_idx)
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=config.encoder_layers,
            embed_dim=config.encoder_embed_dim,
            ffn_embed_dim=config.encoder_ffn_embed_dim,
            attention_heads=config.encoder_attention_heads,
            emb_dropout=config.emb_dropout,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            max_seq_len=config.max_seq_len,
            activation_fn=config.activation_fn,
            no_final_head_layer_norm=config.delta_pair_repr_norm_loss < 0,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, config.encoder_attention_heads, config.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        if config.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=config.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=config.activation_fn,
                weight=None,
            )
        if config.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                config.encoder_attention_heads, 1, config.activation_fn
            )
        if config.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                config.encoder_attention_heads, config.activation_fn
            )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        **params
    ):

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if self.config.masked_token_loss > 0:
            logits = self.lm_head(encoder_rep, encoder_masked_tokens)
        if self.config.masked_coord_loss > 0:
            coords_emb = src_coord
            if padding_mask is not None:
                atom_num = torch.sum(1 - padding_mask.type_as(x), dim=1).view(
                    -1, 1, 1, 1
                )  # consider BOS and EOS as part of the object
            else:
                atom_num = src_coord.shape[1]
            delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
            attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
            coord_update = delta_pos / atom_num * attn_probs
            # Mask padding
            pair_coords_mask = (1 - padding_mask.float()).unsqueeze(-1) * (1 - padding_mask.float()).unsqueeze(1)
            coord_update = coord_update * pair_coords_mask.unsqueeze(-1)
            #
            coord_update = torch.sum(coord_update, dim=2)
            encoder_coord = coords_emb + coord_update
        if self.config.masked_dist_loss > 0:
            encoder_distance = self.dist_head(encoder_pair_rep)

        return (
            logits,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
        )   
    
    def batch_collate_fn(self, batch):
        net_input = {
            'src_tokens': pad_1d_tokens([item[0]['src_tokens'] for item in batch], self.padding_idx),
            'src_coord': pad_coords([item[0]['src_coord'] for item in batch], pad_idx=0.0),
            'src_distance': pad_2d([item[0]['src_distance'] for item in batch], pad_idx=0.0),
            'src_edge_type': pad_2d([item[0]['src_edge_type'] for item in batch], pad_idx=0),
        }
        net_target = {
            'tgt_tokens': pad_1d_tokens([item[1]['tgt_tokens'] for item in batch], self.padding_idx),
            'tgt_coordinates': pad_coords([item[1]['tgt_coordinates'] for item in batch], pad_idx=0.0),
            'tgt_distance': pad_2d([item[1]['tgt_distance'] for item in batch], pad_idx=0.0),
        }
        return {'net_input': net_input, 'net_target': net_target}

class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwconfig):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwconfig):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)
