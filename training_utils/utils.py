import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_region_features(sample):
    region_features = sample["region_features"]
    region_boxes = sample["region_boxes"]

    assert region_features is not None, "region features is required"

    if isinstance(region_features, np.ndarray):
        region_features = torch.tensor(region_features)
    if isinstance(region_boxes, np.ndarray):
        region_boxes = torch.tensor(region_boxes)

    if len(region_features.shape) < 3:
        region_features.unsqueeze_(0)
    if len(region_boxes.shape) < 3:
        region_boxes.unsqueeze_(0)

    return {
        "device": device,
        "batch_size": region_features.shape[0],
        "features": region_features.to(device),
        "boxes": region_boxes.to(device)
    }

def get_grid_features(sample):
    grid_features = sample["grid_features"]
    grid_boxes = sample["grid_boxes"]

    assert grid_features is not None, "grid features is required"

    if isinstance(grid_features, np.ndarray):
        grid_features = torch.tensor(grid_features)
    if isinstance(grid_boxes, np.ndarray):
        grid_boxes = torch.tensor(grid_boxes)

    if len(grid_features.shape) < 3:
        grid_features.unsqueeze_(0)
    if len(grid_boxes.shape) < 3:
        grid_boxes.unsqueeze_(0)

    return {
        "device": device,
        "batch_size": grid_features.shape[0],
        "features": grid_features.to(device),
        "boxes": grid_boxes.to(device)
    }

def get_hybrid_features(sample):
    region_features = sample["region_features"]
    region_boxes = sample["region_boxes"]
    grid_features = sample["grid_features"]
    grid_boxes = sample["grid_boxes"]

    assert region_features is not None and grid_features is not None, "region features and grid features are required"

    if isinstance(region_features, np.ndarray):
        region_features = torch.tensor(region_features)
    if isinstance(region_boxes, np.ndarray):
        region_boxes = torch.tensor(region_boxes)

    if len(region_features.shape) < 3:
        region_features.unsqueeze_(0)
    if len(region_boxes.shape) < 3:
        region_boxes.unsqueeze_(0)

    if isinstance(grid_features, np.ndarray):
        grid_features = torch.tensor(grid_features)
    if isinstance(grid_boxes, np.ndarray):
        grid_boxes = torch.tensor(grid_boxes)

    if len(grid_features.shape) < 3:
        grid_features.unsqueeze_(0)
    if len(grid_boxes.shape) < 3:
        grid_boxes.unsqueeze_(0)

    return {
        "device": device,
        "batch_size": region_features.shape[0],
        "region_features": region_features.to(device),
        "region_boxes": region_boxes.to(device),
        "grid_features": grid_features.to(device),
        "grid_boxes": grid_boxes.to(device)
    }

def get_visual_getter(using_features):
    if using_features == "region":
        return get_region_features

    if using_features == "grid":
        return get_grid_features

    if using_features == "region+grid":
        return get_hybrid_features