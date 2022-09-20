import torch
from torch.utils import data

from data_utils.utils import preprocess_caption
from utils.instances import Instances

import json
import os
import numpy as np
import cv2 as cv
from typing import Dict, List, Any

class BaseDataset(data.Dataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(BaseDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = vocab

        # quesion-answer pairs
        self.annotations = self.load_json(json_data)

        # image features
        self.image_features_path = config.FEATURE_PATH.FEATURES

        # scene text features
        self.scene_text_feature_path = config.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.SCENE_TEXT_THRESHOLD

    def load_json(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    annotation = {
                        "caption": preprocess_caption(ann["caption"], self.vocab.tokenizer),
                        "image_id": ann["image_id"],
                        "filename": image["file_name"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    @property
    def captions(self):
        return [ann["caption"] for ann in self.annotations]

    def load_image_feature(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        
        return features

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_feature_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]

        return features

    def load_features(self, image_id: int) -> Dict[str, Any]:
        image_features = self.load_image_feature(image_id)
        if self.scene_text_feature_path is not None:
            scene_text_features = self.load_scene_text_features(image_id)
            features = {
                **image_features,
                **scene_text_features
            }
        else:
            features = image_features

        return features

    def __getitem__(self, idx: int):
        raise NotImplementedError("Please inherit the BaseDataset class and implement the __getitem__ method")

    def __len__(self) -> int:
        return len(self.annotations)

class DictionaryDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(DictionaryDataset, self).__init__(json_path, vocab, config)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        caption = [item["caption"]]

        return Instances(
            image_id=image_id,
            filename=filename,
            caption=caption,
            **features
        )

class ImageDataset(BaseDataset):
    # This class is designed especially for visualizing purposes
    def __init__(self, json_path: str, vocab, config) -> None:
        super(ImageDataset, self).__init__(json_path, vocab, config)

        self.image_path = config.FEATURE_PATH.IMAGE

    def __getitem__(self, idx: int):
        item = self.annotations[idx]

        image_file = os.path.join(self.image_path, f"{item['filename']}")
        image = cv.imread(image_file)
        image = cv.resize(image, (512, 512), interpolation=cv.INTER_AREA)

        caption = [item["caption"]]
        features = self.load_features(item["image_id"])

        return Instances(
            **features,
            caption=caption
        )

class FeatureDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(FeatureDataset, self).__init__(json_path, vocab, config)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        caption = self.vocab.encode_caption(item["caption"])

        shifted_right_caption = torch.zeros_like(caption).fill_(self.vocab.padding_idx)
        shifted_right_caption[:-1] = caption[1:]
        caption = torch.where(caption == self.vocab.eos_idx, self.vocab.padding_idx, caption) # remove eos_token in caption
        
        features = self.load_features(self.annotations[idx]["image_id"])

        return Instances(
            caption_tokens=caption,
            shifted_right_caption_tokens=shifted_right_caption,
            **features,
        )

    def __len__(self) -> int:
        return len(self.annotations)
