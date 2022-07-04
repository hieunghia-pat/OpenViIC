import torch
from data_utils.feature import Feature
import numpy as np

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

def get_region_features(sample):
    region_features = sample.region_features
    region_boxes = sample.region_boxes

    assert region_features is not None, "region features is required"

    if isinstance(region_features, np.ndarray):
        region_features = torch.tensor(region_features).to(device)
    if isinstance(region_boxes, np.ndarray):
        region_boxes = torch.tensor(region_boxes).to(device)

    if len(region_features.shape) < 3:
        region_features.unsqueeze_(0)
    if len(region_boxes.shape) < 3:
        region_boxes.unsqueeze_(0)

    return Feature({
        "batch_size": region_features.shape[0],
        "device": device,
        "features": region_features,
        "boxes": region_boxes
    })

def get_grid_features(sample):
    grid_features = sample.grid_features
    grid_boxes = sample.grid_boxes

    assert grid_features is not None, "grid features is required"

    if isinstance(grid_features, np.ndarray):
        grid_features = torch.tensor(grid_features).to(device)
    if isinstance(grid_boxes, np.ndarray):
        grid_boxes = torch.tensor(grid_boxes).to(device)

    if len(grid_features.shape) < 3:
        grid_features.unsqueeze_(0)
    if len(grid_boxes.shape) < 3:
        grid_boxes.unsqueeze_(0)

    return Feature({
        "batch_size": grid_features.shape[0],
        "device": device,
        "features": grid_features,
        "boxes": grid_boxes
    })

def get_hybrid_features(sample):
    region_features = sample.region_features
    region_boxes = sample.region_boxes
    grid_features = sample.grid_features
    grid_boxes = sample.grid_boxes

    assert region_features is not None and grid_features is not None, "region features and grid features are required"

    if isinstance(region_features, np.ndarray):
        region_features = torch.tensor(region_features).to(device)
    if isinstance(region_boxes, np.ndarray):
        region_boxes = torch.tensor(region_boxes).to(device)

    if len(region_features.shape) < 3:
        region_features.unsqueeze_(0)
    if len(region_boxes.shape) < 3:
        region_boxes.unsqueeze_(0)

    if isinstance(grid_features, np.ndarray):
        grid_features = torch.tensor(grid_features).to(device)
    if isinstance(grid_boxes, np.ndarray):
        grid_boxes = torch.tensor(grid_boxes).to(device)

    if len(grid_features.shape) < 3:
        grid_features.unsqueeze_(0)
    if len(grid_boxes.shape) < 3:
        grid_boxes.unsqueeze_(0)

    return Feature({
        "batch_size": region_features.shape[0],
        "device": device,
        "region_features": region_features,
        "region_boxes": region_boxes,
        "grid_features": grid_features,
        "grid_boxes": grid_boxes
    })

def get_visual_getter(using_features):
    if using_features == "region":
        return get_region_features

    if using_features == "grid":
        return get_grid_features

    if using_features == "region+grid":
        return get_hybrid_features