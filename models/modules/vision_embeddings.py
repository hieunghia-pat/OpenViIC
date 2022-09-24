import torch
from torch import nn

from builders.vision_embedding_builder import META_VISION_EMBEDDING
from models.utils import generate_padding_mask, get_combine_masks

@META_VISION_EMBEDDING.register()
class FeatureEmbedding(nn.Module):
    def __init__(self, config):
        super(FeatureEmbedding, self).__init__()

        self.proj = nn.Linear(config.D_FEATURE, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, features):
        masks = generate_padding_mask(features, padding_idx=0).to(features.device)
        features = self.proj(features)
        features = self.dropout(features)

        return features, masks

@META_VISION_EMBEDDING.register()
class DualFeatureEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.region_proj = nn.Linear(config.D_REGION_FEATURE, config.D_MODEL)
        self.region_dropout = nn.Dropout(config.DROPOUT)

        self.grid_proj = nn.Linear(config.D_GRID_FEATURE, config.D_MODEL)
        self.grid_dropout = nn.Dropout(config.DROPOUT)

    def forward(self, region_features, grid_features):
        region_masks = generate_padding_mask(region_features, padding_idx=0).to(region_features.device)
        grid_masks = generate_padding_mask(grid_features, padding_idx=0).to(grid_features.device)

        region_features = self.region_proj(region_features)
        region_features = self.grid_dropout(region_features)

        grid_features = self.grid_proj(grid_features)
        grid_features = self.grid_dropout(grid_features)

        return (region_features, region_masks), (grid_features, grid_masks)

@META_VISION_EMBEDDING.register()
class GeometricDualFeatureEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.region_proj = nn.Linear(config.D_REGION_FEATURE, config.D_MODEL)
        self.region_dropout = nn.Dropout(config.DROPOUT)

        self.grid_proj = nn.Linear(config.D_GRID_FEATURE, config.D_MODEL)
        self.grid_dropout = nn.Dropout(config.DROPOUT)

    def forward(self, region_features, region_boxes, grid_features, grid_boxes):
        region_masks = generate_padding_mask(region_features, padding_idx=0).to(region_features.device)
        grid_masks = generate_padding_mask(grid_features, padding_idx=0).to(grid_features.device)
        grid_size = int(grid_boxes.shape[1]**0.5)
        region2grid_masks = get_combine_masks(region_boxes, grid_size)
        grid2region_masks = region2grid_masks.permute(0, 1, 3, 2) # (bs, 1, n_grids, n_regions)
        region2all_masks = torch.cat([region_masks, region2grid_masks], dim=-1) # (bs, 1, n_regions, n_regions + n_grids)
        grid2all_masks = torch.cat([grid2region_masks, grid_masks], dim=-1) # (bs, 1, n_grids, n_regions + n_grids)

        region_features = self.region_proj(region_features)
        region_features = self.grid_dropout(region_features)

        grid_features = self.grid_proj(grid_features)
        grid_features = self.grid_dropout(grid_features)

        return (region_features, region_masks), (grid_features, grid_masks), (region2all_masks, grid2all_masks)