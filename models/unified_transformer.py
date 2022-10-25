import torch

from builders.encoder_builder import build_encoder
from builders.decoder_builder import build_decoder
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE
from utils.instance import Instance
from .base_transformer import BaseTransformer

@META_ARCHITECTURE.register()
class UnifiedTransformer(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(vocab)

        self.device = torch.device(config.DEVICE)

        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)
        self.encoder = build_encoder(config.ENCODER)
        self.decoder = build_decoder(config.DECODER, vocab)

    def forward(self, input_features: Instance):
        region_features = input_features.region_features
        region_boxes = input_features.region_boxes
        grid_features = input_features.grid_features
        grid_boxes = input_features.grid_boxes
        vision_features = torch.cat([region_features, region_boxes, grid_features, grid_boxes], dim=1)
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        encoder_features = self.encoder(Instance(
            features=vision_features,
            features_padding_mask=vision_padding_mask
        ))

        caption_tokens = input_features.caption_tokens
        output = self.decoder(Instance(
            caption_tokens=caption_tokens,
            encoder_features=encoder_features,
            encoder_attention_mask=vision_padding_mask
        ))

        return output

    def encoder_forward(self, input_features: Instance):
        region_features = input_features.region_features
        region_boxes = input_features.region_boxes
        grid_features = input_features.grid_features
        grid_boxes = input_features.grid_boxes
        vision_features = torch.cat([region_features, region_boxes, grid_features, grid_boxes], dim=1)
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        encoder_features = self.encoder(Instance(
            features=vision_features,
            features_padding_mask=vision_padding_mask
        ))

        return encoder_features, vision_padding_mask