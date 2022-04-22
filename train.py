import torch
import os
import pickle
import numpy as np
import random
import json
import config

from training_utils.trainer import Trainer
from data_utils.vocab import Vocab
from data_utils.dataset import FeatureDataset, DictionaryDataset
from data_utils.utils import collate_fn

from models.modules.transformer import Transformer

random.seed(13)
torch.manual_seed(13)
np.random.seed(13)

device = "cuda" if torch.cuda.is_available() else "cpu"

# creating checkpoint directory
if not os.path.isdir(os.path.join(config.checkpoint_path, config.model_name)):
    os.makedirs(os.path.join(config.checkpoint_path, config.model_name))

# Creating vocabulary and dataset
if not os.path.isfile(os.path.join(config.checkpoint_path, config.model_name, "vocab.pkl")):
    vocab = Vocab([config.train_json_path, config.val_json_path], tokenizer=config.tokenizer)
    pickle.dump(vocab, open(os.path.join(config.checkpoint_path, config.model_name, "vocab.pkl"), "wb"))
else:
    vocab = pickle.load(open(os.path.join(config.checkpoint_path, config.model_name, "vocab.pkl"), "rb"))

# creating iterable dataset
train_dataset = FeatureDataset(config.train_json_path, config.feature_path, vocab) # for training with cross-entropy loss
val_dataset = FeatureDataset(config.val_json_path, config.feature_path, vocab) # for calculating evaluation loss
if config.public_test_json_path is not None:
    public_test_dataset = FeatureDataset(config.public_test_json_path, config.feature_path, vocab=vocab)
else:
    public_test_dataset = None

# creating dictionary dataset
train_dict_dataset = DictionaryDataset(config.train_json_path, config.feature_path, vocab) # for training with self-critical learning
val_dict_dataset = DictionaryDataset(config.val_json_path, config.feature_path, vocab) # for calculating metricsn validation set
if config.public_test_json_path is not None:
    public_test_dict_dataset = DictionaryDataset(config.public_test_json_path, config.feature_path, vocab=vocab)
else:
    public_test_dict_dataset = None
if config.private_test_json_path is not None:
    private_test_dict_dataset = DictionaryDataset(config.private_test_json_path, config.feature_path, vocab=vocab)
else:
    private_test_dict_dataset = None

encoder = config.encoder(N=config.nlayers, padding_idx=vocab.padding_idx, d_in=config.d_feature, d_model=config.d_model, d_k=config.d_k, d_v=config.d_v,
                            d_ff=config.d_ff, dropout=config.dropout, attention_module=config.encoder_self_attention,
                            attention_module_kwargs=config.encoder_self_attention_args, **config.encoder_args)
decoder = config.decoder(vocab_size=len(vocab), max_len=vocab.max_caption_length, N_dec=config.nlayers, padding_idx=vocab.padding_idx,
                        d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, d_ff=config.d_ff, dropout=config.dropout,
                        self_att_module=config.decoder_self_attention, enc_att_module=config.decoder_enc_attention,
                        self_att_module_kwargs=config.decoder_enc_attention_args, enc_att_module_kwargs=config.decoder_enc_attention_args, **config.decoder_args)
model = Transformer(vocab.bos_idx, encoder, decoder, **config.transformer_args).to(device)

trainer = Trainer(model=model, train_datasets=(train_dataset, train_dict_dataset), val_datasets=(val_dataset, val_dict_dataset),
                    test_datasets=(public_test_dataset, public_test_dict_dataset), vocab=vocab, collate_fn=collate_fn)

if config.start_from:
    trainer.train(os.path.join(config.checkpoint_path, config.model_name, config.start_from))
else:
    trainer.train()

if public_test_dict_dataset is not None:
    public_results = trainer.get_predictions(public_test_dict_dataset,
                                                checkpoint_filename=os.path.join(config.checkpoint_path, config.model_name, config.start_from),
                                                get_scores=config.get_scores)
    json.dump(public_results, open(os.path.join(config.checkpoint_path, config.model_name, "scored_public_results.json"), "w+"), ensure_ascii=False)

if private_test_dict_dataset is not None:
    private_results = trainer.get_predictions(private_test_dict_dataset,
                                                checkpoint_filename=os.path.join(config.checkpoint_path, config.model_name, config.start_from),
                                                get_scores=config.get_scores)
    json.dump(private_results, open(os.path.join(config.checkpoint_path, config.model_name, "scored_private_results.json"), "w+"), ensure_ascii=False)

if config.sample_public_test_json_path is not None:
    trainer.convert_results(config.sample_public_test_json_path, public_results, split="public")

if config.sample_private_test_json_path is not None:
    trainer.convert_results(config.sample_private_test_json_path, private_results, split="private")