import torch
from torch.utils import data

from data_utils.utils import preprocess_caption
from data_utils.vocab import Vocab
from data_utils.feature import Feature

import json
import os
import numpy as np
from typing import Dict, List, Union

class DictionaryDataset(data.Dataset):
    def __init__(self, json_path: str, image_features_path: str, vocab: Vocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(DictionaryDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path], tokenizer_name=tokenizer_name) if vocab is None else vocab

        # captions
        self.image_ids, self.filenames, self.captions_with_image = self.load_json(json_data)

        # images
        self.image_features_path = image_features_path

    @property
    def max_caption_length(self) -> int:
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.captions)) + 2
        
        return self._max_length

    def load_json(self, json_data: Dict) -> List[Dict]:
        examples = {}
        filenames = {}
        for image in json_data["images"]:
            examples[image["id"]] = []
            filenames[image["id"]] = image["file_name"]

        for ann in json_data["annotations"]:
            caption = preprocess_caption(ann["caption"], self.vocab.tokenizer)
            caption = " ".join(caption)
            examples[ann["image_id"]].append(caption)

        image_ids = []
        captions_with_image = []
        for image_id, captions in examples.items():
            image_ids.append(image_id)
            captions_with_image.append(captions)

        return image_ids, list(filenames.values()), captions_with_image
    
    @property
    def captions(self) -> List[str]:
        return [caption for caption in self.captions]

    def load_feature(self, image_id: int) -> np.ndarray:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        feature = np.load(feature_file, allow_pickle=True)[()]

        region_features = feature["region_features"]
        region_boxes = feature["region_boxes"]
        grid_features = feature["grid_features"]
        grid_boxes = feature["grid_boxes"]

        return region_features, region_boxes, grid_features, grid_boxes

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        filename = self.filenames[idx]
        region_features, region_boxes, grid_features, grid_boxes = self.load_feature(image_id)
        captions = self.captions_with_image[idx]

        return Feature({
            "image_id": image_id,
            "filename": filename, 
            "region_features": region_features,
            "region_boxex": region_boxes,
            "grid_features": grid_features,
            "grid_boxes": grid_boxes,
            "captions": captions
        })

    def __len__(self) -> int:
        return len(self.image_ids)

class FeatureDataset(data.Dataset):
    def __init__(self, json_path: str, image_features_path: str, vocab: Vocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(FeatureDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path], tokenizer_name=tokenizer_name) if vocab is None else vocab

        # captions
        self.annotations = self.load_json(json_data)

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
                        "caption": preprocess_caption(ann["caption"], self.vocab.tokenizer),
                        "image_id": ann["image_id"],
                        "file_name": image["file_name"].split('.')[0]
                    }
                    break

            annotations.append(annotation)

        return annotations

    @property
    def captions(self):
        return [ann["caption"] for ann in self.annotations]

    def load_feature(self, image_id: int) -> np.ndarray:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        feature = np.load(feature_file, allow_pickle=True)[()]

        region_features = feature["region_features"]
        region_boxes = feature["region_boxes"]
        grid_features = feature["grid_features"]
        grid_boxes = feature["grid_boxes"]

        return region_features, region_boxes, grid_features, grid_boxes

    def __getitem__(self, idx: int):
        caption = self.vocab.encode_caption(self.annotations[idx]["caption"])
        shifted_right_caption = torch.zeros_like(caption).fill_(self.vocab.padding_idx)
        shifted_right_caption[:-1] = caption[1:]
        caption = torch.where(caption == self.vocab.eos_idx, self.vocab.padding_idx, caption) # remove eos_token in caption
        region_features, region_boxes, grid_features, grid_boxes = self.load_feature(self.annotations[idx]["image_id"])

        return Feature({
            "region_features": region_features, 
            "region_boxes": region_boxes,
            "grid_features": grid_features,
            "grid_boxes": grid_boxes,
            "caption": caption, 
            "shifted_right_caption": shifted_right_caption
        })

    def __len__(self) -> int:
        return len(self.annotations)