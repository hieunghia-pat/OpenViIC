import torch
import os
import dill as pickle
import numpy as np
import random
import json

from training_utils.captioning_model_trainer import Trainer
from configs.utils import get_encoder, get_decoder
from data_utils.vocab import Vocab
from data_utils.dataset import FeatureDataset, DictionaryDataset
from data_utils.utils import collate_fn
from configs.encoder_decoder_transformer_base import get_default_config

from models.modules.transformers import EncoderDecoderTransformer

random.seed(13)
torch.manual_seed(13)
np.random.seed(13)

config = get_default_config()

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

if config.path.public_test_json_path is not None:
    public_test_dataset = FeatureDataset(config.path.public_test_json_path, config.path.image_features_path, vocab) # for training with cross-entropy loss
else:
    public_test_dataset = None

# creating dictionary dataset
train_dict_dataset = DictionaryDataset(config.path.train_json_path, config.path.image_features_path, vocab) # for training with self-critical learning
val_dict_dataset = DictionaryDataset(config.path.dev_json_path, config.path.image_features_path, vocab) # for calculating metricsn validation set

if config.path.public_test_json_path is not None:
    public_test_dict_dataset = DictionaryDataset(config.path.public_test_json_path, config.path.image_features_path, vocab=vocab)
else:
    public_test_dict_dataset = None

if config.path.private_test_json_path is not None:
    private_test_dict_dataset = DictionaryDataset(config.path.private_test_json_path, config.path.image_features_path, vocab=vocab)
else:
    private_test_dict_dataset = None

# init encoder.
encoder = get_encoder(vocab, config)

# init decoder.
decoder = get_decoder(vocab, config)

# init Transformer model.
model = EncoderDecoderTransformer(vocab.bos_idx, encoder, decoder).to(device)

# Define Trainer
trainer = Trainer(model=model, train_datasets=(train_dataset, train_dict_dataset), val_datasets=(val_dataset, val_dict_dataset),
                    test_datasets=(public_test_dataset, public_test_dict_dataset), vocab=vocab, collate_fn=collate_fn)

# Training
if config.training.start_from:
    trainer.train(os.path.join(config.training.checkpoint_path, config.model.name, config.training.start_from))
else:
    trainer.train()

# Inference on Public test (if available)
if public_test_dict_dataset is not None:
    public_results = trainer.get_predictions(public_test_dict_dataset,
                                                checkpoint_filename=os.path.join(config.path.checkpoint_path, config.model.name, config.training.start_from),
                                                get_scores=config.get_scores)
    json.dump(public_results, open(os.path.join(config.path.checkpoint_path, config.model.name, "scored_public_results.json"), "w+"), ensure_ascii=False)

# Inference on Private test (if available)
if private_test_dict_dataset is not None:
    private_results = trainer.get_predictions(private_test_dict_dataset,
                                                checkpoint_filename=os.path.join(config.path.checkpoint_path, config.model.name, config.training.start_from),
                                                get_scores=config.training.get_scores)
    json.dump(private_results, open(os.path.join(config.path.checkpoint_path, config.model.name, "scored_private_results.json"), "w+"), ensure_ascii=False)

# Convert to the image order of sample submission
if config.path.sample_public_test_json_path is not None:
    trainer.convert_results(config.path.sample_public_test_json_path, public_results, split="public")

if config.path.sample_private_test_json_path is not None:
    trainer.convert_results(config.path.sample_private_test_json_path, private_results, split="private")