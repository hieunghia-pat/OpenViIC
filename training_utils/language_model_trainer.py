from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from data_utils.vocab import Vocab
from data_utils.utils import *
from models.transformers import EncoderDecoderTransformer
from data_utils.dataset import *
from evaluation import compute_language_scores
from training_utils.utils import get_visual_getter

from tqdm import tqdm
from typing import Tuple
import random
from shutil import copyfile

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

class Trainer:
    def __init__(self,  model: EncoderDecoderTransformer, 
                        train_dataset: FeatureDataset,
                        val_dataset: FeatureDataset,
                        test_dataset: Tuple[FeatureDataset, None],
                        vocab: Vocab,
                        config,
                        collate_fn=collate_fn):
        self.model = model
        self.vocab = vocab
        self.config = config

        self.get_visual_features = get_visual_getter(config.training.using_features)

        self.optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        
        self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_idx)
        
        self.epoch = 0

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # creating iterable-dataset data loader
        self.train_dataloader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.workers,
            collate_fn=collate_fn
        )
        self.val_dataloader = data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.workers,
            collate_fn=collate_fn
        )
        
        self.test_dataset = test_dataset

        if self.test_dataset is not None:
            self.test_dataloader = data.DataLoader(
                dataset=self.test_dataset,
                batch_size=self.config.dataset.batch_size,
                shuffle=True,
                num_workers=self.config.dataset.workers,
                collate_fn=collate_fn
            )
        else:
            self.test_dataloader = None

    def evaluate_loss(self, dataloader: data.DataLoader):
        # Calculating validation loss
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % (self.epoch + 1), unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, sample in enumerate(dataloader):
                    tokens = sample["tokens"].to(device)
                    shifted_right_tokens = sample["shifted_right_tokens"].to(device)
                    out, _ = self.model(tokens)
                    out = out.contiguous()
                    loss = self.loss_fn(out.view(-1, len(self.vocab)), shifted_right_tokens.view(-1))
                    this_loss = loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss

    def evaluate_metrics(self, dataloader: data.DataLoader):
        self.model.eval()
        with tqdm(desc='Epoch %d - Evaluation' % (self.epoch + 1), unit='it', total=len(dataloader)) as pbar:
            for it, sample in enumerate(dataloader):
                gt_ids = sample["tokens"].to(device)
                with torch.no_grad():
                    out, _ = self.model(gt_ids)
                    out = out.contiguous()
                predicted_ids = out.argmax(dim=-1)
                captions_gt = self.vocab.decode_caption(gt_ids)
                captions_gen = self.vocab.decode_caption(predicted_ids)
                scores = compute_language_scores(captions_gt, captions_gen)
                pbar.update()

        return scores

    def train_xe(self):
        # Training with cross-entropy loss
        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % (self.epoch + 1), unit='it', total=len(self.train_dataloader)) as pbar:
            for it, sample in enumerate(self.train_dataloader):
                tokens = sample["tokens"].to(device)
                shifted_right_tokens = sample["shifted_right_tokens"].to(device)
                out, _ = self.model(tokens)
                out = out.contiguous()
                self.optim.zero_grad()
                loss = self.loss_fn(out.view(-1, len(self.vocab)), shifted_right_tokens.view(-1))
                loss.backward()

                self.optim.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()

    def lambda_lr(self, s):
        if s <= 3:
            lr = self.config.training.xe_base_lr * s / 4
        elif s <= 10:
            lr = self.config.training.xe_base_lr
        elif s <= 12:
            lr = self.config.training.xe_base_lr * 0.2
        else:
            lr = self.config.training.xe_base_lr * 0.2 * 0.2
        
        return lr

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None

        checkpoint = torch.load(fname)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        self.optim.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        print(f"resuming from epoch {checkpoint['epoch']} - validation loss {checkpoint['val_loss']} - best f1 on val {checkpoint['best_val_f1']} - best f1 on test {checkpoint['best_test_f1']}")

        return {
            "best_val_f1": checkpoint['best_val_f1'],
            "best_test_f1": checkpoint['best_test_f1'],
            "patience": checkpoint['patience'],
            "epoch": checkpoint["epoch"]
        }

    def save_checkpoint(self, dict_for_updating: dict) -> None:
        dict_for_saving = {
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }

        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value

        torch.save(dict_for_saving, os.path.join(self.config.training.checkpoint_path, 
                                                    f"{self.config.model.name}_using_{self.config.training.using_features}", 
                                                    "last_language_model.pth"))

    def train(self, checkpoint_filename: str = None):
        
        if checkpoint_filename is not None and os.path.isfile(checkpoint_filename):
            checkpoint = self.load_checkpoint(checkpoint_filename)
            best_val_f1 = checkpoint["best_val_f1"]
            best_test_f1 = checkpoint["best_test_f1"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"]
        else:
            best_val_f1 = .0
            best_test_f1 = .0
            patience = 0

        while True:
            self.train_xe()

            val_loss = self.evaluate_loss(self.val_dataloader)

            # val scores
            scores = self.evaluate_metrics(self.val_dict_dataloader)
            print("Validation scores", scores)
            val_f1 = scores["f1"]

            if self.test_dict_dataloader is not None:
                scores = self.evaluate_metrics(self.test_dict_dataloader)
                print("Evaluation scores", scores)

            # Prepare for next epoch
            best = False
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
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
                    self.optim = Adam(self.model.parameters(), lr=5e-6)
                    print("Switching to RL")
                else:
                    print('patience reached.')
                    exit_train = True

            if switch_to_rl and not best:
                self.load_checkpoint(os.path.join(self.config.training.checkpoint_path,
                                                    f"{self.config.model.name}_using_{self.config.training.using_features}",
                                                    "best_language_model.pth"))

            self.save_checkpoint({
                'val_loss': val_loss,
                'val_f1': val_f1,
                'patience': patience,
                'best_val_f1': best_val_f1,
                'best_test_f1': best_test_f1,
                'use_rl': use_rl,
            })

            if best:
                copyfile(os.path.join(self.config.training.checkpoint_path,
                                        f"{self.config.model.name}_using_{self.config.training.using_features}",
                                        "last_language_model.pth"), 
                            os.path.join(self.config.training.checkpoint_path, 
                                            f"{self.config.model.name}_using_{self.config.training.using_features}",
                                            "best_language_model.pth"))

            if exit_train:
                break

            self.epoch += 1
            
            print("+"*10)