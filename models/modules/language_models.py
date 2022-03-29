import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel

from models.modules.encoders import EncoderLayer
from models.utils import generate_sequential_mask, sinusoid_encoding_table, generate_padding_mask
from models.modules.containers import Module

import config

class BERTModel(Module):
    def __init__(self, padding_idx=0, bert_hidden_size=768, vocab_size=10201, pretrained_language_model=config.pretrained_language_model,
                    d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, max_len=54, dropout=.1):
        super(BERTModel, self).__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model

        self.language_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.language_model.config.vocab_size = vocab_size
        self.proj_to_caption_model = nn.Linear(bert_hidden_size, d_model)

        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.encoder_layer = EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout)
        self.proj_to_vocab = nn.Linear(d_model, vocab_size)

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
        output_attentions=False, output_hidden_states=False, return_dict=False, encoder_hidden_states=None, encoder_attention_mask=None):
        
        # input (b_s, seq_len)
        b_s, seq_len = input_ids.shape[:2]
        mask_queries = generate_padding_mask(input, self.padding_idx).to(input.device)  # (b_s, seq_len)
        mask_self_attention = generate_sequential_mask(seq_len).to(input.device)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = torch.logical_or(mask_self_attention, mask_queries.unsqueeze(1).unsqueeze(1))
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention
        
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids).long()

        bert_output = self.language_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        language_feature = self.proj_to_caption_model(bert_output.last_hidden_state)
        language_feature = language_feature + self.pos_emb(seq)

        language_feature = self.encoder_layer(language_feature, mask_queries, mask_self_attention)

        logits = self.proj_to_vocab(language_feature)
        out = F.log_softmax(logits, dim=-1)
        return out, language_feature
