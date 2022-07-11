import torch
import os
import dill as pickle
import numpy as np
import random
import argparse

from training_utils.language_model_trainer import Trainer
from configs.utils import get_config
from data_utils.vocab import Vocab
from data_utils.dataset import FeatureDataset
from data_utils.utils import collate_fn
from models.language_models import get_language_model

random.seed(13)
torch.manual_seed(13)
np.random.seed(13)

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)
args = parser.parse_args()

config = get_config(args.config_file)

device = "cuda" if torch.cuda.is_available() else "cpu"

# creating checkpoint directory
if not os.path.isdir(os.path.join(config.training.checkpoint_path, 
                                    f"{config.model.name}_using_{config.training.using_features}")):
    os.makedirs(os.path.join(config.training.checkpoint_path, 
                                f"{config.model.name}_using_{config.training.using_features}"))

# Creating vocabulary and dataset
if not os.path.isfile(os.path.join(config.training.checkpoint_path, 
                                    f"{config.model.name}_using_{config.training.using_features}", "vocab.pkl")):
    print("Creating vocab ...")
    vocab = Vocab([config.path.train_json_path, config.path.dev_json_path], tokenizer_name=config.dataset.tokenizer, 
                    pretrained_language_model_name=config.model.transformer.decoder.args.pretrained_language_model_name)
    pickle.dump(vocab, open(os.path.join(config.training.checkpoint_path, 
                            f"{config.model.name}_using_{config.training.using_features}", "vocab.pkl"), "wb"))
else:
    print("Loading vocab ...")
    vocab = pickle.load(open(os.path.join(config.training.checkpoint_path, 
                                            f"{config.model.name}_using_{config.training.using_features}", "vocab.pkl"), "rb"))

# creating iterable dataset
print("Creating datasets ...")
train_dataset = FeatureDataset(config.path.train_json_path, config.path.image_features_path, vocab) # for training with cross-entropy loss

val_dataset = FeatureDataset(config.path.dev_json_path, config.path.image_features_path, vocab) # for training with cross-entropy loss

if config.path.public_test_json_path is not None:
    public_test_dataset = FeatureDataset(config.path.public_test_json_path, config.path.image_features_path, vocab) # for training with cross-entropy loss
else:
    public_test_dataset = None

# init Transformer model.
print("Creating the language model ...")
model = get_language_model(vocab, config).to(device)

# Define Trainer
trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                    test_dataset=public_test_dataset, vocab=vocab, config=config, collate_fn=collate_fn)

# Training
if config.training.start_from:
    trainer.train(os.path.join(config.training.checkpoint_path, 
                                f"{config.model.name}_using_{config.training.using_features}", 
                                config.training.start_from))
else:
    trainer.train()