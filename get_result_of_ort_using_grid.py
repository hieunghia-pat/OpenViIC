from data_utils.vocab import Vocab
from data_utils.utils import dict_grid_feature_collate_fn, get_tokenizer, preprocess_caption

import evaluation

from models.modules.transformer import Transformer
from models.modules.attentions import AugmentedGeometryScaledDotProductAttention, ScaledDotProductAttention
from models.modules.encoders import Encoder
from models.modules.decoders import Decoder

import torch
from torch.utils import data
from tqdm import tqdm
import os
import pickle
import numpy as np
import itertools
import random
import config
import itertools
import json
from typing import Union, List, Dict, Tuple

class GridDictionaryDataset(data.Dataset):
    def __init__(self, json_path: str, image_features_path: str, tokenizer: Union[str, None] = None, vocab: Vocab = None) -> None:
        '''
            acceptable tokenizers:
                + None
                + "spacy"
                + "pyvi"
                + "vncorenlp"
        '''
        super(GridDictionaryDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path], tokenizer=tokenizer) if vocab is None else vocab

        # captions
        self.image_ids, self.image_files, self.captions_with_image = self.load_json(json_data)

        # images
        self.image_features_path = image_features_path

    @property
    def max_caption_length(self) -> int:
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.captions)) + 2
        
        return self._max_length

    def load_json(self, json_data: Dict) -> List[Dict]:
        examples = {}
        for image in json_data["images"]:
            examples[image["id"]] = {
                "captions": []
            }

        for ann in json_data["annotations"]:
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    examples[ann["image_id"]]["file_name"] = image["file_name"]
                    break
            caption = preprocess_caption(ann["caption"], self.vocab.bos_token, self.vocab.eos_token, self.vocab.tokenizer)
            caption = " ".join(caption[1:-1]) # ignore <bos> and <eos>
            examples[ann["image_id"]]["captions"].append(caption)

        image_ids = []
        image_files = []
        captions_with_image = []
        for image_id, ann in examples.items():
            image_ids.append(image_id)
            image_files.append(ann["file_name"])
            captions_with_image.append(ann["captions"])

        return image_ids, image_files, captions_with_image
    
    @property
    def captions(self) -> List[str]:
        return [caption for caption in self.captions]

    def load_feature(self, image_id: int) -> np.ndarray:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        feature = np.load(feature_file, "r", allow_pickle=False)[:].copy()

        return feature

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[str]]:
        image_id = self.image_ids[idx]
        feature = self.load_feature(image_id)
        captions = self.captions_with_image[idx]
        filename = self.image_files[idx]

        return image_id, filename, torch.tensor(feature).unsqueeze(0), captions

    def __len__(self) -> int:
        return len(self.image_ids)

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_metrics(model: Transformer, dataloader: data.DataLoader, vocab: Vocab):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation ', unit='it', total=len(dataloader)) as pbar:
        for it, (features, caps_gt) in enumerate(dataloader):
            bs, c, h, w = features.shape
            features = features.reshape(bs, -1, c)
            features = features.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(features, grid_size=(h, w), max_len=vocab.max_caption_length, eos_idx=vocab.eos_idx, 
                                            beam_size=config.beam_size, out_size=1)
            caps_gen = vocab.decode_caption(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores

def get_predictions(model: Transformer, dataset: GridDictionaryDataset, vocab: Vocab):
    model.eval()
    results = []
    with tqdm(desc='Evaluation ', unit='it', total=len(dataset)) as pbar:
        for it, (image_id, filename, feature, caps_gt) in enumerate(dataset):
            bs, c, h, w = feature.shape
            feature = feature.reshape(bs, -1, c)
            feature = feature.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(feature, grid_size=(h, w), max_len=vocab.max_caption_length, eos_idx=vocab.eos_idx, 
                                            beam_size=config.beam_size, out_size=1)
            caps_gen = vocab.decode_caption(out, join_words=False)
            gens = []
            gts = []
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gens.append(gen_i)
                gts.append(gts_i)

            results.append({
                "image_id": image_id,
                "filename": filename,
                "gen": gens,
                "gts": gts
            })
            pbar.update()

    return results

vocab: Vocab = pickle.load(open(os.path.join("saved_models/ort_using_grid/vocab.pkl"), "rb"))
vocab.tokenizer = get_tokenizer(None)

dict_dataset = GridDictionaryDataset(json_path="features/annotations/vieCap4H/viecap4h_captions_private_test2017.json",
                                        image_features_path="features/grid_features/vieCap4H/resnext152++", 
                                        vocab=vocab)
dict_dataloader = data.DataLoader(
            dataset=dict_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=dict_grid_feature_collate_fn
        )

# Defining the Object Relation Transformer method
encoder = Encoder(N=config.nlayers, padding_idx=vocab.padding_idx, d_model=config.d_model, d_k=config.d_k, d_v=config.d_v,
                            d_ff=config.d_ff, dropout=config.dropout, attention_module=AugmentedGeometryScaledDotProductAttention)
decoder = Decoder(vocab_size=len(vocab), max_len=vocab.max_caption_length, N_dec=config.nlayers, padding_idx=vocab.padding_idx,
                        d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, d_ff=config.d_ff, dropout=config.dropout,
                        self_att_module=ScaledDotProductAttention, enc_att_module=ScaledDotProductAttention)
model = Transformer(vocab.bos_idx, encoder, decoder).to(device)

fname = "saved_models/ort_using_grid/best_val_model.pth"
checkpoint = torch.load(fname)
torch.set_rng_state(checkpoint['torch_rng_state'])
torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
np.random.set_state(checkpoint['numpy_rng_state'])
random.setstate(checkpoint['random_rng_state'])
model.load_state_dict(checkpoint['state_dict'], strict=False)

results = get_predictions(model, dict_dataset, vocab)

json.dump(results, open("private_test_predictions_.json", "w+"), ensure_ascii=False)