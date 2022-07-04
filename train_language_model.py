import torch
import os
import dill as pickle
import numpy as np
import random
import argparse

from training_utils.captioning_model_trainer import Trainer
from configs.utils import get_config, get_language_model
from data_utils.vocab import Vocab
from data_utils.dataset import FeatureDataset
from data_utils.utils import collate_fn

random.seed(13)
torch.manual_seed(13)
np.random.seed(13)

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)
args = parser.parse_args()

config = get_config(args.config_file)

if not os.path.isdir(os.path.join(config.training.checkpoint_path, config.model.name)):
    os.makedirs(os.path.join(config.training.checkpoint_path, config.model.name))

device = "cuda" if torch.cuda.is_available() else "cpu"

# creating checkpoint directory
if not os.path.isdir(os.path.join(config.training.checkpoint_path, config.model.name)):
    os.makedirs(os.path.join(config.training.checkpoint_path, config.model.name))

# Creating vocabulary and dataset
if not os.path.isfile(os.path.join(config.training.checkpoint_path, config.model.name, "vocab.pkl")):
    vocab = Vocab([config.path.train_json_path, config.path.dev_json_path], tokenizer_name=config.dataset.tokenizer, 
                    pretrained_language_model_name=config.model.pretrained_language_model_name)
    pickle.dump(vocab, open(os.path.join(config.training.checkpoint_path, config.model.name, "vocab.pkl"), "wb"))
else:
    vocab = pickle.load(open(os.path.join(config.training.checkpoint_path, config.model.name, "vocab.pkl"), "rb"))

# creating iterable dataset
train_dataset = FeatureDataset(config.path.train_json_path, config.path.image_features_path, vocab) # for training with cross-entropy loss

val_dataset = FeatureDataset(config.path.dev_json_path, config.path.image_features_path, vocab) # for training with cross-entropy loss

model = get_language_model(vocab, config).to(device)

trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                    test_dataset=(None, None), vocab=vocab, collate_fn=collate_fn)

# Training
if config.training.start_from:
    trainer.train(os.path.join(config.training.checkpoint_path, config.model.name, config.training.start_from))
else:
    trainer.train()