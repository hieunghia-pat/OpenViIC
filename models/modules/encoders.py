from torch import nn

from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention
from builders.encoder_builder import META_ENCODER
from utils.instances import Instances

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAttention(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask, **kwargs)
        ff = self.pwff(att)

        return ff

@META_ENCODER.register()
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, input_features: Instances):
        features = input_features.features
        padding_mask = input_features.features_padding_mask
        
        out = self.layer_norm(features)
        for layer in self.layers:
            out = layer(queries=out, keys=out, values=out, attention_mask=padding_mask)

        return out

@META_ENCODER.register()
class GeometricEncoder(nn.Module):
    def __init__(self, config):
        super(GeometricEncoder, self).__init__()
        
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, input_features: Instances):
        features = input_features.features
        boxes = input_features.boxes
        padding_mask = input_features.features_padding_mask
        
        out = self.layer_norm(features)
        for layer in self.layers:
            out = layer(queries=out, keys=out, values=out, boxes=boxes, attention_mask=padding_mask)

        return out
