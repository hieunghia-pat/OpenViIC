import torch
from data_utils.dataset import GridFeatureDataset, RegionFeatureDataset
from typing import Union, Sequence

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]
GridOrRegionFeatureDataset = Union[GridFeatureDataset, RegionFeatureDataset]