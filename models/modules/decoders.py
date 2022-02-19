import torch
from torch import nn
from torch.nn import functional as F

from models.modules.attentions import MultiHeadAttention
from models.utils import generate_sequential_mask, sinusoid_encoding_table, generate_padding_mask
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.containers import Module, ModuleList

class DecoderLayer(Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 use_aoa=False, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                            use_aoa=use_aoa,
                                            attention_module=self_att_module,
                                            attention_module_kwargs=self_att_module_kwargs)
        self.enc_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                            use_aoa=use_aoa,
                                            attention_module=enc_att_module,
                                            attention_module_kwargs=enc_att_module_kwargs)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, attention_mask=mask_self_att)
        self_att = self_att * mask_pad

        enc_att = self.enc_att(self_att, enc_output, enc_output, attention_mask=mask_enc_att) * mask_pad
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        
        return ff

class MeshedDecoderLayer(Module):
    def __init__(self, N_enc, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 use_aoa=False, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                            use_aoa=use_aoa,
                                            attention_module=self_att_module,
                                            attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                            use_aoa=use_aoa,
                                            attention_module=enc_att_module,
                                            attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.N_enc = N_enc
        self.fc_alphas = nn.ModuleList([nn.Linear(d_model + d_model, d_model) for _ in range(N_enc)])

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        assert enc_output.size(1) == self.N_enc, "total layers of the encoder must equal to total number of the encoder outputs"
        
        self_att = self.self_att(input, input, input, attention_mask=mask_self_att)
        self_att = self_att * mask_pad

        enc_atts = []
        for ith in range(self.N_enc):
            enc_atts.append(self.enc_att(self_att, enc_output[:, ith], enc_output[:, ith], attention_mask=mask_enc_att) * mask_pad)

        alphas = []
        for fc_alpha, enc_att in zip(self.fc_alphas, enc_atts):
            alphas.append(torch.sigmoid(fc_alpha(torch.cat([self_att, enc_att], -1))))

        attn = 0
        for alpha, enc_att in zip(alphas, enc_atts):
            attn += enc_att * alpha
        attn = attn / torch.sqrt(self.N_enc)
        enc_att = enc_att * mask_pad

        ff = self.pwff(attn)
        ff = ff * mask_pad

        return ff

class Decoder(Module):
    "Generic N layer decoder with masking."
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 use_aoa=False, self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, use_aoa=use_aoa,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = generate_padding_mask(input, self.padding_idx)  # (b_s, seq_len)
        mask_self_attention = generate_sequential_mask(seq_len)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention or mask_queries.unsqueeze(1).unsqueeze(1)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
        for layer in self.layers:
            out = layer(out, encoder_output, mask_queries.unsqueeze(-1), mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

class MeshedDecoder(Module):
    def __init__(self, vocab_size, max_len, N_enc, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 use_aoa=False, self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoder, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [MeshedDecoderLayer(N_enc, d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, use_aoa=use_aoa,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = generate_padding_mask(input, self.padding_idx)  # (b_s, seq_len)
        mask_self_attention = generate_sequential_mask(seq_len)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention or mask_queries.unsqueeze(1).unsqueeze(1)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
        for layer in self.layers:
            out = layer(out, encoder_output, mask_queries.unsqueeze(-1), mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
