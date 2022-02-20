import torch
from torch.utils import data
from torch.utils.data.dataset import random_split
from data_utils.typing import GridOrRegionFeatureDataset
from data_utils.utils import preprocess_sentence
from data_utils.vocab import Vocab
import json
import config
import os
import numpy as np
import cv2 as cv
from typing import Dict, List, Tuple, Union

class GridFeatureDataset(data.Dataset):
    def __init__(self, json_path: str, image_features_path: str, vocab: Vocab = None) -> None:
        super(GridFeatureDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path]) if vocab is None else vocab

        # captions
        self.annotations = self.load_json(json_data)

        # images
        self.image_features_path = image_features_path

    @property
    def max_caption_length(self) -> int:
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.captions)) + 2
        
        return self._max_length

    def load_json(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    annotation = {
                        "caption": preprocess_sentence(ann["caption"]),
                        "image_id": ann["image_id"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    def load_feature(self, image_id: int) -> np.ndarray:
        feature_file = os.path.join("features", f"image_{image_id}.npy")
        feature = np.load(feature_file, "r", allow_pickle=True)["feature"]

        return feature

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        caption = self.vocab.encode_caption(self.annotations[idx]["caption"])
        visual = self.load_feature(self.annotations[idx]["image_id"])

        return visual, caption[:-1], caption[1:] # shifted-right output

    def __len__(self) -> int:
        return len(self.annotations)

class RegionFeatureDataset(data.Dataset):
    def __init__(self, json_path: str, image_features_path: str, vocab: Vocab = None) -> None:
        super(GridFeatureDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path]) if vocab is None else vocab

        # captions
        self.annotations = self.load_json(json_data)

        # images
        self.image_features_path = image_features_path

    @property
    def max_caption_length(self) -> int:
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.captions)) + 2
        
        return self._max_length

    def load_json(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    annotation = {
                        "caption": preprocess_sentence(ann["caption"]),
                        "image_id": ann["image_id"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    def load_feature(self, image_id: int) -> np.ndarray:
        feature_file = os.path.join("features", f"image_{image_id}.npy")
        feature = np.load(feature_file, "r", allow_pickle=True)["feature"]

        return feature

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        caption = self.vocab.encode_sentence(self.annotations[idx]["caption"])
        visual = self.load_image(self.annotations[idx]["image_id"])

        return visual, caption[:-1], caption[1:] # shifted-right output

    def __len__(self) -> int:
        return len(self.annotations)

def get_loader(train_dataset: GridOrRegionFeatureDataset, 
                test_dataset: GridOrRegionFeatureDataset = None) -> Union[List[data.DataLoader], Tuple[List[data.DataLoader], data.DataLoader]]:
    """ Returns a data loader for the desired split """

    fold_size = int(len(train_dataset) * 0.2)

    subdatasets = random_split(train_dataset, [fold_size, fold_size, fold_size, fold_size, len(train_dataset) - fold_size*4], generator=torch.Generator().manual_seed(13))
    
    folds = []
    for subdataset in subdatasets:
        folds.append(
            torch.utils.data.DataLoader(
                subdataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=config.data_workers))

    if test_dataset:
        test_fold = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=config.batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=config.data_workers)

        return folds, test_fold

    return folds