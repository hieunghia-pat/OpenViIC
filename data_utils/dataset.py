import torch
from torch.utils import data

from data_utils.utils import preprocess_caption
from data_utils.vocab import Vocab

import json
import os
import numpy as np
from typing import Dict, List, Union
from collections import defaultdict
import h5py

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

        if "boxes" in feature:
            boxes = feature["boxes"]
        else:
            boxes = None

        if "grid_size" in feature:
            grid_size = feature["grid_size"]
        else:
            grid_size = None

        return feature["features"], boxes, grid_size

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        filename = self.filenames[idx]
        features, boxes, grid_size = self.load_feature(image_id)
        captions = self.captions_with_image[idx]

        returning_dict = defaultdict(lambda: None)
        result_dict = {
            "image_id": image_id, 
            "filename": filename, 
            "features": features, 
            "boxes": boxes,
            "grid_size": grid_size, 
            "captions": captions
        }

        for key, value in result_dict.items():
            returning_dict[key] = value

        return returning_dict

    def __len__(self) -> int:
        return len(self.image_ids)

class FeatureDataset(data.Dataset):
    def __init__(self, json_path: str, image_features_path: str, hdf5_grid_features_path: str, masks_path: str, guided_load_feature = dict, max_detections = int, vocab: Vocab = None, idx_by_filename = bool, tokenizer_name: Union[str, None] = None) -> None:
        super(FeatureDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path], tokenizer_name=tokenizer_name) if vocab is None else vocab

        # captions
        self.annotations = self.load_json(json_data)

        # Type features: 'region', 'grids'
        self.guided_load_feature = guided_load_feature

        # max detections
        self.max_detections = max_detections

        # index type
        self.idx_by_filename = idx_by_filename

        if self.guided_load_feature['region']:
            self.image_features_path = image_features_path
        
        if self.guided_load_feature['grid']:
            self.hdf5_grid_features_path = hdf5_grid_features_path
            self.f_features = h5py.File(self.hdf5_grid_features_path, 'r')

        if guided_load_feature['region'] and self.guided_load_feature['grid']:
            self.masks_path = masks_path

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

    def load_feature(self, image_id: str) -> np.ndarray:
        '''
        Load iterable data.
        '''

        delta = None # difference between max_detections and objects in region features.

        # Load grid features from directory.
        if self.guided_load_feature['grid']:
            grid_feature = self.f_features['%s_grids' % image_id][()]

        # Load region features by hdf5 file.
        if self.guided_load_feature['region']:
            feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
            feature = np.load(feature_file, allow_pickle=True)[()]
            boxes = feature["boxes"]
            region_features = feature['features']

            # Padding region features
            delta = self.max_detections - region_features.shape[0]
            if delta > 0:
                region_features = np.concatenate([region_features, np.zeros((delta, region_features.shape[1]))], axis=0)
                boxes = np.concatenate([boxes, np.zeros((delta, boxes.shape[1]))], axis=0)
            elif delta < 0:
                region_features = region_features[:self.max_detections]
                boxes = boxes[:self.max_detections]

        # Load masks (combine feature)
        if self.guided_load_feature['region'] and self.guided_load_feature['grid']:
            mask = np.load(self.masks_path + image_id + '.npy', allow_pickle=True)
            if delta > 0:
                mask = np.concatenate([mask, np.zeros((delta, mask.shape[1]))], axis=0)
            else:
                mask = mask[:self.max_detections]

        if self.guided_load_feature['region'] and self.guided_load_feature['grid']:
            return region_features.astype(np.float32), \
            boxes.astype(np.float32), \
            grid_feature.astype(np.float32), \
            mask.astype(np.float32)

        elif self.guided_load_feature['region']:
            return region_features.astype(np.float32), \
            boxes.astype(np.float32)
        
        elif self.guided_load_feature['grid']:
            return grid_feature.astype(np.float32), \

    def __getitem__(self, idx: int):
        caption = self.vocab.encode_caption(self.annotations[idx]["caption"])
        shifted_right_caption = torch.zeros_like(caption).fill_(self.vocab.padding_idx)
        shifted_right_caption[:-1] = caption[1:]
        caption = torch.where(caption == self.vocab.eos_idx, self.vocab.padding_idx, caption) # remove eos_token in caption
        
        region_features, boxes, grid_features, masks = None, None, None, None
        grid_size = 7

        if self.idx_by_filename:
            if self.guided_load_feature['grid'] and self.guided_load_feature['region']:
                region_features, boxes, grid_features, masks = self.load_feature(self.annotations[idx]["file_name"])
            elif self.guided_load_feature['grid']:
                grid_features = self.load_feature(self.annotations[idx]["file_name"])
            elif self.guided_load_feature['region']:
                region_features, boxes = self.load_feature(self.annotations[idx]["file_name"])
        else:
            if self.guided_load_feature['grid'] and self.guided_load_feature['region']:
                region_features, boxes, grid_features, masks = self.load_feature(self.annotations[idx]["image_id"])
            elif self.guided_load_feature['grid']:
                grid_features = self.load_feature(self.annotations[idx]["image_id"])
            elif self.guided_load_feature['region']:
                region_features, boxes = self.load_feature(self.annotations[idx]["image_id"])

        result_dict = {
            "region_features": region_features, 
            "boxes": boxes,
            "grid_features": grid_features,
            "grid_size": grid_size,
            "caption": caption, 
            "shifted_right_caption": shifted_right_caption,
            "masks": masks
        }

        returning_dict = defaultdict(lambda: None)
        for key, value in result_dict.items():
            returning_dict[key] = value

        return returning_dict

    def __len__(self) -> int:
        return len(self.annotations)