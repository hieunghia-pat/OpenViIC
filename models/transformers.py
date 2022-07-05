import torch
from torch import nn

from models.captioning_model import CaptioningModel

class EncoderDecoderTransformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(EncoderDecoderTransformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder

        self.register_state('enc_output', None)
        self.register_state('enc_attention_mask', None)
        self.register_state("enc_pos_embedding", None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens, **visual_inputs):
        enc_output, enc_attention_mask, enc_pos_embedding = self.encoder(**visual_inputs)

        decoder_inputs = {
            "tokens": tokens,
            "enc_outputs": enc_output,
            "enc_attention_mask": enc_attention_mask,
            "enc_pos_embedding": enc_pos_embedding
        }
        dec_output = self.decoder(**decoder_inputs)
        
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, ith, prev_output, **visual_inputs):
        if ith == 0:
            self.enc_output, self.enc_attention_mask, self.enc_pos_embedding = self.encoder(**visual_inputs)
            bs = self.enc_output.shape[0]
            it = torch.zeros((bs, 1)).long().fill_(self.bos_idx).to(self.enc_output.device)
        else:
            it = prev_output

        decoder_inputs = {
            "tokens": it,
            "enc_outputs": self.enc_output,
            "enc_attention_mask": self.enc_attention_mask,
            "enc_pos_embedding": self.enc_pos_embedding
        }
        dec_output = self.decoder(**decoder_inputs)

        return dec_output