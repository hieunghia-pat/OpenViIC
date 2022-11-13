from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention


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

        att, att_score = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff, att_score


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

        self.self_att = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=1, dropout=dropout,
                                            identity_map_reordering=identity_map_reordering,
                                            attention_module=attention_module,
                                            attention_module_kwargs=attention_module_kwargs)
        self.mlp1 = nn.Linear(3*d_model, 3*d_model)
        self.mlp2 = nn.Linear(3*d_model, d_model)

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        outs = []
        att_scores = []
        for l in self.layers:
            out, att_score = l(out, out, out, attention_mask, attention_weights)
            outs.append(out)
            att_scores.append(att_score)

        out1, out2, out3 = outs
        out2 = 0.1*self.self_att(out2, out1, out1)[0] + out2
        out3 = 0.1*self.self_att(out3, out2, out2)[0] + out3

        out = self.mlp1(torch.cat(outs, dim=-1))
        out = F.leaky_relu(out)
        out = self.mlp2(out)
        out = F.leaky_relu(out)

        out = out3 + 0.2*out
        
        return out, attention_mask, att_scores
