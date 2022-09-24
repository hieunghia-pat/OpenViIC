import torch
from torch import nn
from torch import functional as F

from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.pos_embeddings import SinusoidPositionalEmbedding
from models.modules.attentions import MultiHeadAttention
from models.utils import clones, box_relational_embedding
from builders.encoder_builder import META_ENCODER
from utils.instances import Instances

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAttention(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, padding_mask, attention_mask, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, padding_mask=padding_mask, attention_mask=attention_mask, **kwargs)
        ff = self.pwff(att)
        ff = ff.masked_fill(padding_mask.squeeze(1).squeeze(1).unsqueeze(-1), value=0)

        return ff

@META_ENCODER.register()
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, input_features: Instances):
        features = input_features.features
        padding_mask = input_features.features_padding_mask
        
        out = self.layer_norm(features) + self.pos_embedding(features)
        for layer in self.layers:
            out = layer(queries=out, keys=out, values=out, padding_mask=padding_mask, attention_mask=padding_mask)

        return 

@META_ENCODER.register()
class GeometricEncoder(nn.Module):
    def __init__(self, config):
        super(GeometricEncoder, self).__init__()
        
        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.trignometric_embedding = config.TRIGNOMETRIC_EMBEDDING
        if self.trignometric_embedding:
            self.d_g = config.D_MODEL // config.HEAD
        else:
            self.d_g = 4
        
        self.layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

        self.init_weights()

    def init_weights(self):
        for fc_g in self.fc_gs:
            nn.init.xavier_uniform_(fc_g.weight)

        for fc_g in self.fc_gs:
            nn.init.constant_(fc_g.bias, 0)

    def forward(self, input_features: Instances):
        features = input_features.features
        boxes = input_features.boxes
        padding_mask = input_features.features_padding_mask

        # embedding geometric information from boxes' coordinates
        relative_geometry_embeddings = box_relational_embedding(boxes, dim_g=self.d_g, trignometric_embedding=self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.d_g)
        bs, nk, _, _ = relative_geometry_embeddings.shape
        box_size_per_head = [bs, 1, nk, nk]
        relative_geometry_weights_per_head = [fc_g(flatten_relative_geometry_embeddings).view(box_size_per_head) for fc_g in self.fc_gs]
        relative_geometry_weights = torch.cat(relative_geometry_weights_per_head, dim=1) # (bs, h, nk, nk)
        relative_geometry_weights = F.relu(relative_geometry_weights)
        
        out = self.layer_norm(features)
        for layer in self.layers:
            out = layer(queries=out, 
                        keys=out, 
                        values=out, 
                        relative_geometry_weights=relative_geometry_weights,
                        padding_mask=padding_mask,
                        attention_mask=padding_mask)

        return out

@META_ENCODER.register()
class DualCollaborativeLevelEncoder(nn.Module):
    def __init__(self, config):
        super(DualCollaborativeLevelEncoder, self).__init__()

        self.d_model = config.D_MODEL

        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.trignometric_embedding = config.TRIGNOMETRIC_EMBEDDING
        if self.trignometric_embedding:
            self.d_g = config.D_MODEL // config.HEAD
        else:
            self.d_g = 4

        self.layer_norm_region = nn.LayerNorm(self.d_model)
        self.layer_norm_grid = nn.LayerNorm(self.d_model)

        self.fc_gs = clones(nn.Linear(self.d_g, 1), config.HEAD)

        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL, normalize=True)

        # Attention on regions
        self.layers_region = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

        # Attention on grids
        self.layers_grid = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

        # Cross Attention between regions and grids
        self.region2grid = nn.ModuleList([EncoderLayer(config.CROSS_ATTENTION) for _ in range(config.LAYERS)])

        # Cross Attention between grids and regions
        self.grid2region = nn.ModuleList([EncoderLayer(config.CROSS_ATTENTION) for _ in range(config.LAYERS)])

        self.init_weights()

    def init_weights(self):
        for fc_g in self.fc_gs:
            nn.init.xavier_uniform_(fc_g.weight)

        for fc_g in self.fc_gs:
            nn.init.constant_(fc_g.bias, 0)

    def forward(self, input_features: Instances):
        region_features = input_features.region_features
        region_boxes = input_features.region_boxes
        region_padding_mask = input_features.region_padding_mask
        region2all_mask = input_features.region2all_mask
        grid_features = input_features.grid_features
        grid_boxes = input_features.grid_boxes
        grid_padding_mask = input_features.grid_padding_mask
        grid2all_mask = input_features.grid2all_mask

        n_regions = region_features.shape[1]

        boxes = torch.cat([region_boxes, grid_boxes], dim=1) # (bs, n_regions + n_grids, 4)
        relative_geometry_embeddings = box_relational_embedding(boxes, dim_g=self.d_g, trignometric_embedding=self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.d_g)
        bs, nk, _, _ = relative_geometry_embeddings.shape
        box_size_per_head = [bs, 1, nk, nk]
        relative_geometry_weights_per_head = [fc_g(flatten_relative_geometry_embeddings).view(box_size_per_head) for fc_g in self.fc_gs]
        relative_geometry_weights = torch.cat(relative_geometry_weights_per_head, dim=1) # (bs, h, nk, nk)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        region_features = self.layer_norm_region(region_features) + self.pos_embedding(region_features)
        grid_features = self.layer_norm_grid(grid_features) + self.pos_embedding(grid_features)
        for l_region, l_grid, l_r2g, l_g2r in zip(self.layers_region, self.layers_grid, self.region2grid, self.grid2region):
            # self-attention on region feature
            region_features = l_region(queries=region_features,
                                        values=region_features, 
                                        keys=region_features, 
                                        relative_geometry_weights=relative_geometry_weights[:, :, :n_regions, :n_regions],
                                        padding_mask=region_padding_mask,
                                        attention_mask=region_padding_mask)

            #self-attention on grid feature
            grid_features = l_grid(queries=grid_features, 
                                    values=grid_features, 
                                    keys=grid_features,
                                    relative_geometry_weights=relative_geometry_weights[:, :, n_regions:, n_regions:],
                                    padding_mask=grid_padding_mask,
                                    attention_mask=grid_padding_mask)

            # prepare the combined output
            combined_features = torch.cat([region_features, grid_features], dim=1)
            combined_features = combined_features + self.pos_embedding(combined_features)

            # locally contrained cross-attention for region
            region_features = l_r2g(queries=region_features, 
                                keys=combined_features, 
                                values=combined_features, 
                                relative_geometry_weights=relative_geometry_weights[:, :, :n_regions, :],
                                padding_mask=region2all_mask,
                                attention_mask=region2all_mask)

            # locally contrained cross-attention for grid
            grid_features = l_g2r(queries=grid_features, 
                                keys=combined_features, 
                                values=combined_features, 
                                relative_geometry_weights=relative_geometry_weights[:, :, n_regions:, :],
                                padding_mask=grid2all_mask,
                                attention_mask=grid2all_mask)

        out = torch.cat([region_features, grid_features], dim=1)
        padding_mask = torch.cat([region_padding_mask, grid_padding_mask], dim=-1)
        
        return out, padding_mask