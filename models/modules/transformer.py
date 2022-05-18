import torch
from torch import nn
from models.captioning_model import CaptioningModel
from models.modules.embeddings import SinusoidPositionalEmbedding, PositionEmbeddingSine

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, use_img_pos=False, use_box_embedd=False, **kwargs):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.use_img_pos = use_img_pos
        if self.use_img_pos:
            # self.sinusoid_pos_embedding = SinusoidPositionalEmbedding(decoder.d_model // 2, normalize=True)
            self.pos_embedding = PositionEmbeddingSine(decoder.d_model // 2, normalize=True)
        self.use_box_embedd = use_box_embedd
        if self.use_box_embedd:
            self.box_embedding = nn.Linear(4, 512)

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_pos_embedding(self, boxes, grids, split=False):
        bs = boxes.shape[0]
        region_embed = self.box_embedding(boxes)
        grid_embed = self.grid_embedding(grids.view(bs, 7, 7, -1))
        if not split:
            pos = torch.cat([region_embed, grid_embed], dim=1)
            return pos
        else:
            return region_embed, grid_embed

    def forward(self, input, tokens, boxes=None, grid_sizes=None, **kwargs):
        # Get batch size
        bs = input.shape[0]

        # Init positional embeddings
        region_embed, grid_embed, pos_emb = None, None, None

        if self.use_box_embedd:
            # Dual-Collborative mode.
            region_features = input
            grid_features = kwargs['grid_features']
            masks = kwargs['masks']
            # Positional embedding for region features.
            region_embed = self.box_embedding(boxes) if self.use_box_embedd else None
            # Positional embedding for grid features.
            grid_embed = self.pos_embedding(grid_features.view(bs, 7, 7, -1)) if self.use_img_pos else None
            # Fit into the Encoder.
            enc_output, mask_enc = self.encoder(region_features=region_features, grid_features=grid_features, boxes=boxes, aligns=masks, \
                                            region_embed=region_embed, grid_embed=grid_embed)
        
        else:
          # Positional Embeddings.
          pos_emb = self.sinusoid_pos_embedding(input) if self.use_img_pos else None
          enc_output, mask_enc = self.encoder(input, boxes, grid_sizes, positional_emb=pos_emb)
        
        dec_output = self.decoder(tokens, enc_output, mask_encoder=mask_enc, positional_emb=pos_emb)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, boxes, grid_sizes=None, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            pos_emb = self.sinusoid_pos_embedding(visual) if self.use_img_pos else None
            if t == 0:
                if self.use_box_embedd:
                    # Dual-Collborative Encoder.
                    bs = visual.shape[0]
                    region_features = visual
                    grid_features = kwargs['grid_features']
                    masks = kwargs['masks']
                    region_embed = self.box_embedding(boxes) if self.use_box_embedd else None
                    grid_embed = self.pos_embedding(grid_features.view(bs, 7, 7, -1)) if self.use_img_pos else None
                    self.enc_output, self.mask_enc = self.encoder(region_features=region_features, grid_features=grid_features, aligns=masks, \
                    boxes=boxes, region_embed=region_embed, grid_embed=grid_embed)
                else:
                    # Using one type of feature.
                    self.enc_output, self.mask_enc = self.encoder(visual, boxes, grid_sizes, positional_emb=pos_emb)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, mask_encoder=self.mask_enc, positional_emb=pos_emb)