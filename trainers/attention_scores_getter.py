import torch
from torch.utils.data import DataLoader
from data_utils.vocab import Vocab
from data_utils.dataset import FeatureDataset, DictionaryDataset

from data_utils.utils import collate_fn
from utils.logging_utils import setup_logger
from builders.model_builder import build_model
from builders.trainer_builder import META_TRAINER

import os
import numpy as np
import pickle
import random
from tqdm import tqdm
import itertools

logger = setup_logger()

@META_TRAINER.register()
class AttentionScoreGetter:
    def __init__(self, config):

        self.checkpoint_path = os.path.join(config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
        if not os.path.isdir(self.checkpoint_path):
            logger.info("Creating checkpoint path")
            os.makedirs(self.checkpoint_path)

        if not os.path.isfile(os.path.join(self.checkpoint_path, "vocab.bin")):
            logger.info("Creating vocab")
            self.vocab = self.load_vocab(config)
            logger.info("Saving vocab to %s" % os.path.join(self.checkpoint_path, "vocab.bin"))
            pickle.dump(self.vocab, open(os.path.join(self.checkpoint_path, "vocab.bin"), "wb"))
        else:
            logger.info("Loading vocab from %s" % os.path.join(self.checkpoint_path, "vocab.bin"))
            self.vocab = pickle.load(open(os.path.join(self.checkpoint_path, "vocab.bin"), "rb"))

        logger.info("Loading data")
        self.train_dataset, self.dev_dataset, self.test_dataset = self.load_feature_datasets(config.DATASET)
        self.train_dict_dataset, self.dev_dict_dataset, self.test_dict_dataset = self.load_dict_datasets(config.DATASET)
        
        # creating iterable-dataset data loader
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.DATASET.FEATURE_BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=config.DATASET.FEATURE_BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=config.DATASET.FEATURE_BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn
        )

        # creating dictionary iterable-dataset data loader
        self.train_dict_dataloader = DataLoader(
            dataset=self.train_dict_dataset,
            batch_size=config.DATASET.DICT_BATCH_SIZE // config.TRAINING.TRAINING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.val_dict_dataloader = DataLoader(
            dataset=self.dev_dict_dataset,
            batch_size=config.DATASET.DICT_BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.test_dict_dataloader = DataLoader(
            dataset=self.test_dict_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn
        )

        logger.info("Building model")
        self.model = build_model(config.MODEL, self.vocab)
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)

        logger.info("Defining optimizer and objective function")
        self.configuring_hyperparameters(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.warmup = config.TRAINING.WARMUP
        self.score = config.TRAINING.SCORE
        self.rl_learning_rate = config.TRAINING.RL_LEARNING_RATE
        self.get_scores = config.TRAINING.GET_SCORES
        self.training_beam_size = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience = config.TRAINING.PATIENCE

    def load_vocab(self, config):
        vocab = Vocab(config.DATASET)

        return vocab

    def load_feature_datasets(self, config):
        train_dataset = FeatureDataset(config.JSON_PATH.TRAIN, self.vocab, config)
        dev_dataset = FeatureDataset(config.JSON_PATH.DEV, self.vocab, config)
        test_dataset = FeatureDataset(config.JSON_PATH.TEST, self.vocab, config)

        return train_dataset, dev_dataset, test_dataset

    def load_dict_datasets(self, config):
        train_dataset = DictionaryDataset(config.JSON_PATH.TRAIN, self.vocab, config)
        dev_dataset = DictionaryDataset(config.JSON_PATH.DEV, self.vocab, config)
        test_dataset = DictionaryDataset(config.JSON_PATH.TEST, self.vocab, config)

        return train_dataset, dev_dataset, test_dataset

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None

        logger.info("Loading checkpoint from %s", fname)

        checkpoint = torch.load(fname)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        self.model.load_state_dict(checkpoint['state_dict'], strict=True)

        logger.info(f"Resuming from epoch %s", checkpoint['epoch'])

        return checkpoint

    def start(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            logger.error("Getting attention scores require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        attention_scores = {}
        with tqdm(desc='Getting attention score', unit='it', total=len(self.test_dict_dataloader)) as pbar:
            for it, items in enumerate(self.test_dict_dataloader):
                items = items.to(self.device)
                image_id = items.id[0]
                with torch.no_grad():
                    outs, att_scores = self.model.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size, out_size=1)
                
                tokens = self.vocab.decode_caption(outs, join_words=False)[0]
                tokens = [self.vocab.bos_token] + [k for k, g in itertools.groupby(tokens)] + [self.vocab.eos_token]
                attention_scores[image_id] = {
                    "attention_scores": att_scores,
                    "tokens": tokens
                }

                pbar.update()

        torch.save(attention_scores, "transformer_attention_score.bin")
