import random
from data import ImageDetectionsFieldRegion, TextField, RawField
from data import COCO
from evaluation import PTBTokenizer, Cider
from models.camo_transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
import torch
from tqdm import tqdm
import argparse, os, pickle
import numpy as np

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def get_attention_scores(model, dataloader, text_field):
    model.eval()
    att_scores = {}
    with tqdm(desc='Getting attention scores', unit='it', total=len(dataloader)) as pbar:
        for it, ((images, image_id), _) in enumerate(iter(dataloader)):
            images = torch.tensor(images).unsqueeze(0).to(device)
            with torch.no_grad():
                out, _, dec_att_scores = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
                inds = out[0].tolist()
                tokens = [text_field.vocab.itos[idx] for idx in inds]

            att_scores[image_id] = {
                "attention_scores": dec_att_scores,
                "tokens": tokens
            }
            pbar.update()

    return att_scores


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()
    print(args)

    # Pipeline for image regions
    image_field = ImageDetectionsFieldRegion(detections_path=args.features_path, max_detections=100, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'openviic_images', args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, d_in=1024, attention_module=ScaledDotProductAttention)
    decoder = TransformerDecoderLayer(len(text_field.vocab), 130, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)

    # Test scores
    att_scores = get_attention_scores(model, dict_dataset_test, text_field)
    torch.save(att_scores, "attention_scores_%s" % args.exp_name)