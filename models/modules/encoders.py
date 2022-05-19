import torch
from torch import nn
from torch.nn import functional as F
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention
from models.modules.geometry_features import AllRelationalEmbedding

class EncoderLayer(nn.Module):
    '''
    Encoder layer
    '''
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        use_aoa=use_aoa,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, boxes=None, grid_sizes=None, positional_emb=None, attention_mask=None, attention_weights=None):

        if positional_emb is not None:
            queries += positional_emb
            keys += positional_emb
        
        # Init att
        att = None

        if torch.is_tensor(grid_sizes):
          # Fit directly geometry features into MHA, not grid_sizes.
          box_relation_embed_matrix = grid_sizes
          att = self.mhatt(queries, keys, values, boxes=boxes, grid_sizes=box_relation_embed_matrix, attention_mask=attention_mask, attention_weights=attention_weights)

        else:
          # Fit grid sizes into MHA.
          att = self.mhatt(queries, keys, values, boxes=boxes, grid_sizes=grid_sizes, attention_mask=attention_mask, attention_weights=attention_weights)
        
        ff = self.pwff(att)
        return ff

class LCCA(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(LCCA, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None,
                pos_source=None, pos_cross=None):

        q = queries + pos_source
        k = keys + pos_cross
        att = self.mhatt(q, k, values, grid_sizes=relative_geometry_weights, attention_mask=attention_mask, attention_weights=attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff

class Encoder(nn.Module):
    '''
    Baseline Encoder
    '''
    def __init__(self, N, padding_idx, d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(Encoder, self).__init__()
        
        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, boxes=None, grid_sizes=None, positional_emb=None, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out = layer(out, out, out, boxes, grid_sizes, positional_emb, attention_mask, attention_weights)

        return out, attention_mask

class MultiLevelEncoder(nn.Module):
    '''
    Multi-level Encoder
    '''
    def __init__(self, N, padding_idx, d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()

        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, boxes=None, grid_sizes=None, positional_emb=None, attention_weights=None):
        # input (b_s, seq_len, d_in)
        # blank features are added by zero tensors
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        outs = []
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out = layer(out, out, out, boxes, grid_sizes, positional_emb, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, dim=1)
        return outs, attention_mask

class DualCollaborativeLevelEncoder(nn.Module):
    '''
    DLCT Encoder
    '''
    def __init__(self, N, padding_idx, d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, use_aoa=False, attention_module=None, attention_module_kwargs=None, multi_level_output=None):
        super(DualCollaborativeLevelEncoder, self).__init__()

        self.d_model = d_model
        self.dropout = dropout

        self.fc_region = nn.Linear(d_in, self.d_model)
        self.dropout_region = nn.Dropout(p=self.dropout)
        self.layer_norm_region = nn.LayerNorm(self.d_model)

        self.fc_grid = nn.Linear(d_in, self.d_model)
        self.dropout_grid = nn.Dropout(p=self.dropout)
        self.layer_nrom_grid = nn.LayerNorm(self.d_model)

        # Attention on regions
        self.layers_region = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                                identity_map_reordering=identity_map_reordering,
                                                                attention_module=attention_module,
                                                                attention_module_kwargs=attention_module_kwargs)
                                                    for _ in range(N)])

        # Attention on grids
        self.layers_grid = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                       identity_map_reordering=identity_map_reordering,
                                                       attention_module=attention_module,
                                                       attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        
        # Cross Attention between regions and grids
        self.region2grid = nn.ModuleList([LCCA(d_model, d_k, d_v, h, d_ff, dropout,
                                                            identity_map_reordering=identity_map_reordering,
                                                            attention_module=attention_module,
                                                            attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        
        # Cross Attention between grids and regions
        self.grid2region = nn.ModuleList([LCCA(d_model, d_k, d_v, h, d_ff, dropout,
                                                            identity_map_reordering=identity_map_reordering,
                                                            attention_module=attention_module,
                                                            attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])

        self.padding_idx = padding_idx
        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])
        
        # Whether using multi level output in encoder head.
        self.multi_level_output = multi_level_output

    def forward(self, region_features, grid_features, boxes, aligns, attention_weights=None, region_embed=None, grid_embed=None):
        # input (b_s, seq_len, d_in)
        # blank features are added by zero tensors

        mask_regions = (torch.sum(region_features, dim=-1) == 0).unsqueeze(-1)
        mask_grids = (torch.sum(grid_features, dim=-1) == 0).unsqueeze(-1)

        out_region = F.relu(self.fc_region(region_features))
        out_region = self.dropout_region(out_region)
        out_region = self.layer_norm_region(out_region)
        out_region = out_region.masked_fill(mask_regions, 0)

        out_grid = F.relu(self.fc_grid(grid_features))
        out_grid = self.dropout_grid(out_grid)
        out_grid = self.layer_nrom_grid(out_grid)
        out_grid = out_grid.masked_fill(mask_grids, 0)

        region_features = out_region
        grid_features = out_grid

        attention_mask_region = (torch.sum(region_features == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        attention_mask_grid = (torch.sum(grid_features == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        relative_geometry_embeddings = AllRelationalEmbedding(boxes)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)

        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        n_regions = region_features.shape[1]  # 50
        n_grids = grid_features.shape[1]  # 49

        region2region = relative_geometry_weights[:, :, :n_regions, :n_regions]
        grid2grid = relative_geometry_weights[:, :, n_regions:, n_regions:]
        region2all = relative_geometry_weights[:, :, :n_regions,:]
        grid2all = relative_geometry_weights[:, :, n_regions:, :]

        bs = region_features.shape[0]

        out_region = region_features
        out_grid = grid_features
        aligns = aligns.unsqueeze(1)  # bs * 1 * n_regions * n_grids

        tmp_mask = torch.eye(n_regions, device=out_region.device).unsqueeze(0).unsqueeze(0)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_regions * n_regions
        region_aligns = (torch.cat([tmp_mask, aligns], dim=-1) == 0) # bs * 1 * n_regions *(n_regions+n_grids)

        tmp_mask = torch.eye(n_grids, device=out_region.device).unsqueeze(0).unsqueeze(0)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_grids * n_grids
        grid_aligns = (torch.cat([aligns.permute(0, 1, 3, 2), tmp_mask], dim=-1)==0) # bs * 1 * n_grids *(n_grids+n_regions)

        pos_cross = torch.cat([region_embed, grid_embed],dim=-2)
        
        outs = None
        if self.multi_level_output:
          outs = []
        out = None
        for l_region, l_grid, l_r2g, l_g2r in zip(self.layers_region, self.layers_grid, self.region2grid,
                                                  self.grid2region):

            # Dual-way Self-Attention
            out_region = l_region(queries=out_region, values=out_region, keys=out_region, grid_sizes=region2region, positional_emb=region_embed, \
            attention_mask=attention_mask_region, attention_weights=attention_weights)

            out_grid = l_grid(queries=out_grid, values=out_grid, keys=out_grid, grid_sizes=grid2grid, positional_emb=grid_embed, \
            attention_mask=attention_mask_grid, attention_weights=attention_weights)

            # Concat
            out_all = torch.cat([out_region, out_grid], dim=1)

            # Cross Self-Attention between regions and grids
            out_region = l_r2g(queries=out_region, keys=out_all, values=out_all, relative_geometry_weights=region2all, \
                              attention_mask=region_aligns, attention_weights=attention_weights, \
                              pos_source=region_embed, pos_cross=pos_cross)

            out_grid = l_g2r(queries=out_grid, keys=out_all, values=out_all, relative_geometry_weights=grid2all, \
                            attention_mask=grid_aligns, attention_weights=attention_weights, \
                            pos_source=grid_embed, pos_cross=pos_cross)

            # Concat
            out = torch.cat([out_region, out_grid], dim=1)
            
            # If 'multi_level_output' is applied.
            if self.multi_level_output:
              outs.append(out.unsqueeze(1))

        # If 'multi_level_output' is applied.
        if self.multi_level_output:
          outs = torch.cat(outs, dim=1)
        
        attention_mask = torch.cat([attention_mask_region, attention_mask_grid], dim=-1)
        
        # If 'multi_level_output' is applied.
        if self.multi_level_output:
          return outs, attention_mask
        else:
          return out, attention_mask