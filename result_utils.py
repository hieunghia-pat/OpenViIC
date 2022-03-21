from data_utils.vocab import Vocab
from data_utils.dataset import *

from models.modules.transformer import Transformer
from models.modules.attentions import *
from models.modules.encoders import *
from models.modules.decoders import *

import torch
from torch.utils import data
from tqdm import tqdm
import os
import itertools
import config
import itertools
import json

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_predictions_region_feature(model: Transformer, dataset: data.Dataset, vocab: Vocab):
    model.eval()
    results = []
    with tqdm(desc='Evaluating: ', unit='it', total=len(dataset)) as pbar:
        for it, (image_id, filename, feature, boxes, caps_gt) in enumerate(dataset):
            feature = torch.tensor(feature).unsqueeze(0).to(device)
            boxes = torch.tensor(boxes).unsqueeze(0).to(device)
            with torch.no_grad():
                out, _ = model.beam_search(feature, boxes=boxes, max_len=vocab.max_caption_length, eos_idx=vocab.eos_idx, 
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

def convert_results(sample_submisison_json, results, split="public"):
    sample_json_data = json.load(open(sample_submisison_json))
    for sample_item in tqdm(sample_json_data, desc="Converting results: "):
        for item in results:
            if sample_item["id"] == item["filename"]:
                sample_item["captions"] = item["gen"][0]

    json.dump(sample_json_data, open(os.path.join(config.checkpoint_path, config.model_name, f"{split}_results.json"), "w+"), ensure_ascii=False)
