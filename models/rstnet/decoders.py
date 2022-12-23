import torch
from torch import nn
from torch.nn import functional as F

from models.transformer.attention import MultiHeadAttention
from models.rstnet.attention import MultiHeadAdaptiveAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList
from models.rstnet.language_model import BertAdaptiveLanguageModel, AlbertAdaptiveLanguageModel

class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att, pos):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad
        # MHA+AddNorm
        key = enc_output + pos
        enc_att = self.enc_att(self_att, key, enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class DecoderAdaptiveLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderAdaptiveLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)

        self.enc_att = MultiHeadAdaptiveAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False, attention_module=enc_att_module, attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att, language_feature=None, pos=None):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad
        # MHA+AddNorm
        key = enc_output + pos
        enc_att = self.enc_att(self_att, key, enc_output, mask_enc_att, language_feature=language_feature)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff

class TransformerBertModelDecoderLayer(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, bert_hidden_size=768, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None, language_model_path=None):
        super(TransformerBertModelDecoderLayer, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) if i < N_dec else DecoderAdaptiveLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) for i in range(N_dec + 1)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        self.language_model = BertAdaptiveLanguageModel(padding_idx=padding_idx, bert_hidden_size=bert_hidden_size, vocab_size=vocab_size, max_len=max_len)
        assert language_model_path is not None
        language_model_file = torch.load(language_model_path)
        self.language_model.load_state_dict(language_model_file['state_dict'], strict=False)
        for p in self.language_model.parameters():
            p.requires_grad = False

        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder, pos):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.float() * -10e4 # for HuggingFace compatition
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
        _, language_feature = self.language_model(input)

        if encoder_output.shape[0] != pos.shape[0]:
            assert encoder_output.shape[0] % pos.shape[0] == 0
            beam_size = int(encoder_output.shape[0] / pos.shape[0])
            shape = (pos.shape[0], beam_size, pos.shape[1], pos.shape[2])
            pos = pos.unsqueeze(1)  # bs * 1 * 50 * 512
            pos = pos.expand(shape)  # bs * 5 * 50 * 512
            pos = pos.contiguous().flatten(0, 1)  # (bs*5) * 50 * 512

        for i, l in enumerate(self.layers):
            if i < self.N:
                out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder, pos=pos)
            else:
                out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder, language_feature, pos=pos)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

class TransformerAlbertModelDecoderLayer(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, bert_hidden_size=768, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None, language_model_path=None):
        super(TransformerAlbertModelDecoderLayer, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) if i < N_dec else DecoderAdaptiveLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) for i in range(N_dec + 1)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        self.language_model = AlbertAdaptiveLanguageModel(padding_idx=padding_idx, bert_hidden_size=bert_hidden_size, vocab_size=vocab_size, max_len=max_len)
        assert language_model_path is not None
        language_model_file = torch.load(language_model_path)
        self.language_model.load_state_dict(language_model_file['state_dict'], strict=False)
        for p in self.language_model.parameters():
            p.requires_grad = False

        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder, pos):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.float() * -10e4 # for HuggingFace compatition
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
        _, language_feature = self.language_model(input)

        if encoder_output.shape[0] != pos.shape[0]:
            assert encoder_output.shape[0] % pos.shape[0] == 0
            beam_size = int(encoder_output.shape[0] / pos.shape[0])
            shape = (pos.shape[0], beam_size, pos.shape[1], pos.shape[2])
            pos = pos.unsqueeze(1)  # bs * 1 * 50 * 512
            pos = pos.expand(shape)  # bs * 5 * 50 * 512
            pos = pos.contiguous().flatten(0, 1)  # (bs*5) * 50 * 512

        for i, l in enumerate(self.layers):
            if i < self.N:
                out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder, pos=pos)
            else:
                out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder, language_feature, pos=pos)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

class BertLinguisticDecoderLayer(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, bert_hidden_size=768, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None, pretrained_name="bert-base-multilingual-cased"):
        super().__init__()

        self.d_model = d_model

        self.word_emb = nn.Embedding(vocab_size, bert_hidden_size, padding_idx=padding_idx)
        self.language_model = AlbertAdaptiveLanguageModel(padding_idx=padding_idx, bert_hidden_size=bert_hidden_size, 
                                                    pretrained_name=pretrained_name, vocab_size=vocab_size)
        self.proj_embedding_to_model = nn.Linear(bert_hidden_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        
        self.layers = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, 
                                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, 
                                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])

        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder, pos):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.float() * -10e4 # for HuggingFace compatition
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input)
        out = self.language_model(
            input_features=out,
            padding_mask=mask_queries.unsqueeze(-1),
            attention_mask=mask_self_attention
        )
        out = self.proj_embedding_to_model(out)

        out += self.pos_emb(out)
        if encoder_output.shape[0] != pos.shape[0]:
            assert encoder_output.shape[0] % pos.shape[0] == 0
            beam_size = int(encoder_output.shape[0] / pos.shape[0])
            shape = (pos.shape[0], beam_size, pos.shape[1], pos.shape[2])
            pos = pos.unsqueeze(1)  # bs * 1 * 50 * 512
            pos = pos.expand(shape)  # bs * 5 * 50 * 512
            pos = pos.contiguous().flatten(0, 1)  # (bs*5) * 50 * 512

        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder, pos=pos)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)