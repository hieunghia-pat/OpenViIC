from torch.nn import functional as F
from .utils import PositionWiseFeedForward
import torch
from torch import nn
from .attention import MultiHeadAttention
from .utils import clones, box_relational_embedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff

class AugmentedGeometryEncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super().__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None):

        att = self.mhatt(queries=queries, keys=keys, values=values, 
                            relative_geometry_weights=relative_geometry_weights,
                            attention_mask=attention_mask, 
                            attention_weights=attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class TransformerEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout

        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
        
        return out, attention_mask


class GeometricEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 trignometric_embedding=True, identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(GeometricEncoder, self).__init__()
        
        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.padding_idx = padding_idx

        self.d_model = d_model
        self.trignometric_embedding = trignometric_embedding
        if self.trignometric_embedding:
            self.d_g = d_model // h
        else:
            self.d_g = 4

        self.fc_gs = clones(nn.Linear(self.d_g, 1), h)
        
        self.layers = nn.ModuleList([AugmentedGeometryEncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                                    identity_map_reordering=identity_map_reordering,
                                                                    attention_module=attention_module,
                                                                    attention_module_kwargs=attention_module_kwargs)
                                                        for _ in range(N)])

        self.self_att = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=1, dropout=dropout,
                                            identity_map_reordering=identity_map_reordering,
                                            attention_module=attention_module,
                                            attention_module_kwargs=attention_module_kwargs)
        self.mlp1 = nn.Linear(3*d_model, 3*d_model)
        self.mlp2 = nn.Linear(3*d_model, d_model)

        self.init_weights()

    def init_weights(self):
        for fc_g in self.fc_gs:
            nn.init.xavier_uniform_(fc_g.weight)

        for fc_g in self.fc_gs:
            nn.init.constant_(fc_g.bias, 0)

    def forward(self, input: torch.Tensor, boxes: torch.Tensor):
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        # embedding geometric information from boxes' coordinates
        relative_geometry_embeddings = box_relational_embedding(boxes, dim_g=self.d_g, trignometric_embedding=self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.d_g)
        bs, nk, _, _ = relative_geometry_embeddings.shape
        box_size_per_head = [bs, 1, nk, nk]
        relative_geometry_weights_per_head = [fc_g(flatten_relative_geometry_embeddings).view(box_size_per_head) for fc_g in self.fc_gs]
        relative_geometry_weights = torch.cat(relative_geometry_weights_per_head, dim=1) # (bs, h, nk, nk)
        relative_geometry_weights = F.relu(relative_geometry_weights)
        
        outs = []
        for layer in self.layers:
            out = layer(queries=out, 
                        keys=out, 
                        values=out, 
                        relative_geometry_weights=relative_geometry_weights,
                        attention_mask=attention_mask)
            outs.append(out)

        out1, out2, out3 = outs
        out2 = 0.1*self.self_att(out2, out1, out1) + out2
        out3 = 0.1*self.self_att(out3, out2, out2) + out3

        out = self.mlp1(torch.cat(outs, dim=-1))
        out = F.leaky_relu(out)
        out = self.mlp2(out)
        out = F.leaky_relu(out)

        out = out3 + 0.2*out

        return out, attention_mask