from data_utils.dataset import RegionFeatureDataset, RegionDictionaryDataset
from data_utils.vocab import Vocab
from data_utils.utils import region_feature_collate_fn, dict_region_feature_collate_fn

import evaluation
from evaluation import PTBTokenizer, Cider

from models.modules.transformer import Transformer
from models.modules.attentions import AugmentedGeometryScaledDotProductAttention, ScaledDotProductAttention
from models.modules.encoders import Encoder
from models.modules.decoders import Decoder

from result_utils import *

import torch
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
import os
import pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import random
import config
import itertools

random.seed(13)
torch.manual_seed(13)
np.random.seed(13)

def evaluate_loss(model: Transformer, dataloader: data.DataLoader, loss_fn: NLLLoss, vocab: Vocab):
    # Calculating validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - Validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (features, boxes, tokens, shifted_right_tokens) in enumerate(dataloader):
                features = features.to(device)
                boxes = boxes.to(device)
                tokens = tokens.to(device)
                shifted_right_tokens = shifted_right_tokens.to(device)
                out = model(features, tokens, boxes=boxes).contiguous()
                loss = loss_fn(out.view(-1, len(vocab)), shifted_right_tokens.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)

    return val_loss

def evaluate_metrics(model: Transformer, dataloader: data.DataLoader, vocab: Vocab):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - Evaluation' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (features, boxes, caps_gt) in enumerate(dataloader):
            features = features.to(device)
            boxes = boxes.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(features, boxes=boxes, max_len=vocab.max_caption_length, eos_idx=vocab.eos_idx, 
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

def train_xe(model: Transformer, dataloader: data.DataLoader, optim: Adam, vocab: Vocab):
    # Training with cross-entropy loss
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - Training with cross-entropy loss' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (features, boxes, tokens, shifted_right_tokens) in enumerate(dataloader):
            features = features.to(device)
            boxes = boxes.to(device)
            tokens = tokens.to(device)
            shifted_right_tokens = shifted_right_tokens.to(device)
            out = model(features, tokens, boxes=boxes).contiguous()
            optim.zero_grad()
            loss = loss_fn(out.view(-1, len(vocab)), shifted_right_tokens.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)

    return loss

def train_scst(model: Transformer, dataloader: data.DataLoader, optim: Adam, cider: Cider, vocab: Vocab):
    # Training with self-critical learning
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0

    model.train()

    running_loss = .0

    with tqdm(desc='Epoch %d - Training with self-critical learning' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (features, boxes, caps_gt) in enumerate(dataloader):
            features = features.to(device)
            boxes = boxes.to(device)
            outs, log_probs = model.beam_search(features, boxes=boxes, max_len=vocab.max_caption_length, eos_idx=vocab.eos_idx,
                                                beam_size=config.batch_size, out_size=config.beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = vocab.decode_caption(outs.contiguous().view(-1, vocab.max_caption_length), join_words=True)
            caps_gt = list(itertools.chain(*([c, ] * config.beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(features.shape[0], config.beam_size)
            reward_baseline = torch.mean(reward, dim=-1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)

    return loss, reward, reward_baseline

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # creating checkpoint directory
    if not os.path.isdir(os.path.join(config.checkpoint_path, config.model_name)):
        os.makedirs(os.path.join(config.checkpoint_path, config.model_name))

    # Creating vocabulary and dataset
    if not os.path.isfile(os.path.join(config.checkpoint_path, config.model_name, "vocab.pkl")):
        vocab = Vocab([config.train_json_path, config.val_json_path])
        pickle.dump(vocab, open(os.path.join(config.checkpoint_path, config.model_name, "vocab.pkl"), "wb"))
    else:
        vocab = pickle.load(open(os.path.join(config.checkpoint_path, config.model_name, "vocab.pkl"), "rb"))

    # creating iterable dataset
    train_dataset = RegionFeatureDataset(config.train_json_path, config.feature_path, vocab) # for training with cross-entropy loss
    val_dataset = RegionFeatureDataset(config.val_json_path, config.feature_path, vocab) # for calculating evaluation loss
    public_test_dict_dataset = RegionDictionaryDataset(config.public_test_json_path, config.feature_path, vocab=vocab)
    private_test_dict_dataset = RegionDictionaryDataset(config.private_test_json_path, config.feature_path, vocab=vocab)

    # creating dictionary dataset
    train_dict_dataset = RegionDictionaryDataset(config.train_json_path, config.feature_path, vocab) # for training with self-critical learning
    val_dict_dataset = RegionDictionaryDataset(config.val_json_path, config.feature_path, vocab) # for calculating metrics on validation set
    test_dict_dataset = RegionDictionaryDataset(config.test_json_path, config.feature_path, vocab) # for calculating metrics on test set

    # Defining the Object Relation Transformer method
    encoder = Encoder(N=config.nlayers, padding_idx=vocab.padding_idx, d_model=config.d_model, d_k=config.d_k, d_v=config.d_v,
                                use_aoa=True, d_ff=config.d_ff, dropout=config.dropout, attention_module=AugmentedGeometryScaledDotProductAttention)
    decoder = Decoder(vocab_size=len(vocab), max_len=vocab.max_caption_length, N_dec=config.nlayers, padding_idx=vocab.padding_idx,
                            d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, d_ff=config.d_ff, dropout=config.dropout,
                            self_att_module=ScaledDotProductAttention, enc_att_module=ScaledDotProductAttention)
    model = Transformer(vocab.bos_idx, encoder, decoder).to(device)
    # for evaluating self-critical learning
    cider_train = Cider(PTBTokenizer.tokenize(train_dataset.captions))

    def lambda_lr(s):
        warm_up = config.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=vocab.padding_idx)
    use_rl = False
    best_val_cider = .0
    best_test_cider = .0
    patience = 0
    start_epoch = 0

    if config.start_from:
        fname = os.path.join(config.checkpoint_path, config.model_name, config.start_from)

        if os.path.exists(fname):
            checkpoint = torch.load(fname)
            torch.set_rng_state(checkpoint['torch_rng_state'])
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['random_rng_state'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            """
            optim.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            """
            start_epoch = checkpoint['epoch'] + 1
            best_val_cider = checkpoint['best_val_cider']
            best_test_cider = checkpoint['best_test_cider']
            patience = checkpoint['patience']
            use_rl = checkpoint['use_rl']
            optim.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            print(f"resuming from epoch {checkpoint['epoch']} - validation loss {checkpoint['val_loss']} - best cider on val {checkpoint['best_val_cider']} - best cider on test {checkpoint['best_test_cider']}")

    for epoch in range(start_epoch, start_epoch + config.epochs):
        # creating iterable-dataset data loader
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=region_feature_collate_fn
        )
        val_dataloader = data.DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=region_feature_collate_fn
        )

        # creating dictionary iterable-dataset data loader
        train_dict_dataloader = data.DataLoader(
            dataset=train_dict_dataset,
            batch_size=config.batch_size // config.beam_size,
            shuffle=True,
            collate_fn=dict_region_feature_collate_fn
        )
        val_dict_dataloader = data.DataLoader(
            dataset=val_dict_dataset,
            batch_size=config.batch_size // config.beam_size,
            shuffle=True,
            collate_fn=dict_region_feature_collate_fn
        )

        if not use_rl:
            train_loss = train_xe(model, train_dataloader, optim, vocab)
        else:
            train_loss, reward, reward_baseline = train_scst(model, train_dict_dataloader, optim, cider_train, vocab=vocab)

        val_loss = evaluate_loss(model, val_dataloader, loss_fn, vocab)

        # val scores
        scores = evaluate_metrics(model, val_dict_dataloader, vocab)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']

        # Prepare for next epoch
        best = False
        if val_cider >= best_val_cider:
            best_val_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False

        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            checkpoint = torch.load(os.path.join(config.checkpoint_path, config.model_name, "best_val_model.pth"))
            torch.set_rng_state(checkpoint['torch_rng_state'])
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['random_rng_state'])
            model.load_state_dict(checkpoint['state_dict'])
            print('Resuming from epoch %d, validation loss %f, best_val_cider %f, and best test_cider %f' % (
                checkpoint['epoch'], checkpoint['val_loss'], checkpoint['best_val_cider'], checkpoint['best_test_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict() if not use_rl else optim.state_dict(),
            'scheduler': scheduler.state_dict() if not use_rl else scheduler.state_dict(),
            'patience': patience,
            'best_val_cider': best_val_cider,
            'best_test_cider': best_test_cider,
            'use_rl': use_rl,
        }, os.path.join(config.checkpoint_path, config.model_name, "last_model.pth"))

        if best:
            copyfile(os.path.join(config.checkpoint_path, config.model_name, "last_model.pth"), os.path.join(config.checkpoint_path, config.model_name, "best_val_model.pth"))
            public_test_results = get_predictions_region_feature(model, public_test_dict_dataset, vocab=vocab)
            convert_results(config.sample_public_test_json_path, public_test_results, split="public")
            private_test_results = get_predictions_region_feature(model, private_test_dict_dataset, vocab=vocab)
            convert_results(config.sample_private_test_json_path, private_test_results, split="private")

        if exit_train:
            break

        print("+"*10)