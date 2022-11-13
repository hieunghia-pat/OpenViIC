import torch
from torch import nn
import numpy as np
from builders.attention_builder import build_attention, META_ATTENTION

from models.modules.containers import Module

@META_ATTENTION.register()
class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()

        d_model = config.D_MODEL
        h = config.HEAD
        d_k = config.D_KEY
        d_v = config.D_VALUE

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out, att.view(-1, nq, nk)

@META_ATTENTION.register()
class AugmentedGeometryScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention with box relation
    '''

    def __init__(self, config):
        super(AugmentedGeometryScaledDotProductAttention, self).__init__()

        d_model = config.D_MODEL
        h = config.HEAD
        d_k = config.D_KEY
        d_v = config.D_VALUE

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        a = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            a = a.masked_fill(attention_mask, -np.inf)

        g = relative_geometry_weights
        mn = torch.log(torch.clamp(g, min = 1e-6)) + a
        mn = torch.softmax(mn, dim=-1)
        out = torch.matmul(mn, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out, mn.view(-1, nq, nk)

@META_ATTENTION.register()
class AugmentedMemoryScaledDotProductAttention(nn.Module):
    '''
        Scaled dot-product attention with memory
    '''

    def __init__(self, config):
        super(AugmentedMemoryScaledDotProductAttention, self).__init__()

        d_model = config.D_MODEL
        h = config.HEAD
        d_k = config.D_KEY
        d_v = config.D_VALUE
        m = config.MEMORY

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.m_k = nn.Parameter(torch.FloatTensor(1, m, h * d_k))
        self.m_v = nn.Parameter(torch.FloatTensor(1, m, h * d_v))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.m = m

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.normal_(self.m_k, 0, 1 / self.d_k)
        nn.init.normal_(self.m_v, 0, 1 / self.m)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None):
        '''
        Computes
            :param queries: Queries (b_s, nq, d_model)
            :param keys: Keys (b_s, nk, d_model)
            :param values: Values (b_s, nk, d_model)
            :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
            :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        m_k = np.sqrt(self.d_k) * self.m_k.expand(b_s, self.m, self.h * self.d_k)
        m_v = np.sqrt(self.m) * self.m_v.expand(b_s, self.m, self.h * self.d_v)

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = torch.cat([self.fc_k(keys), m_k], 1).view(b_s, nk + self.m, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = torch.cat([self.fc_v(values), m_v], 1).view(b_s, nk + self.m, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out, att.view(-1, nq, nk)

@META_ATTENTION.register()
class AdaptiveScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product with adaptive attention
    '''

    def __init__(self, config):
        super(AdaptiveScaledDotProductAttention, self).__init__()

        d_model = config.D_MODEL
        h = config.HEAD
        d_k = config.D_KEY
        d_v = config.D_VALUE
        dropout = config.DROPOUT

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_s = nn.Linear(d_model, h * d_k) # for language signals
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.xavier_uniform_(self.fc_s.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        nn.init.constant_(self.fc_s.bias, 0)

    def forward(self, queries, keys, values, language_signals, attention_mask=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param language_signals: Language signals (b_s, ns, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        s = self.fc_s(language_signals).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        attn = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask, -np.inf)

        language_attn = torch.matmul(q, s.permute(0, 1, 3, 2)) / np.sqrt(self.d_k)  # (b_s, h, nq, nq)
        language_attn = torch.cat([language_attn[:, :, i, i].unsqueeze(-1) for i in range(nq)], -1) # (b_s, h, nq)

        combined_attn = torch.cat([attn, language_attn.unsqueeze(-1)], dim=-1)     # (b_s, h, nq, nk + 1)
        combined_attn = [torch.softmax(combined_attn[:, :, i, :].unsqueeze(2), dim=-1) for i in range(nq)] # [ (b_s, h, 1, nk + 1) ]

        combined_v = [torch.cat([v, s[:, :, i, :].unsqueeze(2)], 2) for i in range(nq)] # [ (b_s, h, nk + 1, d_v) ]

        assert len(combined_attn) == len(combined_v) == nq
        out = torch.cat([torch.matmul(combined_attn[i], combined_v[i]) for i in range(nq)], dim=2)

        out = out.permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out, None

class MultiHeadAttention(Module):
    '''
        Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        
        d_model = config.D_MODEL

        self.use_aoa = config.USE_AOA # whether to use Attention on Attention (AoA) mechanism or not
        
        if self.use_aoa:    # define additionally AoA layers
            self.informative_attention = nn.Linear(2*d_model, d_model)
            self.gated_attention = nn.Linear(2*d_model, d_model)

        self.attention = build_attention(config)

        self.dropout = nn.Dropout(p=config.DROPOUT)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = config.CAN_BE_STATEFUL
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, padding_mask, attention_mask, **kwargs):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        out, att_score = self.attention(queries, keys, values, attention_mask=attention_mask, **kwargs)
        # out = out.masked_fill(padding_mask.squeeze(1).squeeze(1).unsqueeze(-1), value=0)
        
        # normalization after residual connection
        out = self.dropout(out)
        out = self.layer_norm(queries + out)

        if self.use_aoa:
            aoa_input = torch.cat([queries, out], dim=-1)
            i = self.informative_attention(aoa_input)
            g = torch.sigmoid(self.gated_attention(aoa_input))
            out = i * g
            # out = out.masked_fill(padding_mask.squeeze(1).squeeze(1).unsqueeze(-1), value=0)
            
        return out, att_score.mean(dim=0)