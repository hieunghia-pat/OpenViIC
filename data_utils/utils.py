from collections import defaultdict
import torch
from torchvision import transforms
import re
from typing import Union

def get_tokenizer(tokenizer):
    if callable(tokenizer):
        return tokenizer
    elif tokenizer is None:
        return lambda s: s 
    elif tokenizer == "pyvi":
        try:
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize
        except ImportError:
            print("Please install PyVi package. "
                  "See the docs at https://github.com/trungtv/pyvi for more information.")
    elif tokenizer == "spacy":
        try:
            from spacy.lang.vi import Vietnamese
            return Vietnamese()
        except ImportError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
    elif tokenizer == "vncorenlp":
        try:
            from vncorenlp import VnCoreNLP
            annotator = VnCoreNLP(r"data_utils/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

            def tokenize(s: str):
                words = annotator.tokenize(s)[0]
                return " ".join(words)

            return tokenize
        except ImportError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise
        except AttributeError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise

def preprocess_caption(caption, tokenizer: Union[str, None]):
    caption = re.sub(r"[“”]", "\"", caption)
    caption = re.sub(r"!", " ! ", caption)
    caption = re.sub(r"\?", " ? ", caption)
    caption = re.sub(r":", " : ", caption)
    caption = re.sub(r";", " ; ", caption)
    caption = re.sub(r",", " , ", caption)
    caption = re.sub(r"\"", " \" ", caption)
    caption = re.sub(r"'", " ' ", caption)
    caption = re.sub(r"\(", " ( ", caption)
    caption = re.sub(r"\[", " [ ", caption)
    caption = re.sub(r"\)", " ) ", caption)
    caption = re.sub(r"\]", " ] ", caption)
    caption = re.sub(r"/", " / ", caption)
    caption = re.sub(r"\.", " . ", caption)
    caption = re.sub(r".\. *\. *\. *", " ... ", caption)
    caption = re.sub(r"\$", " $ ", caption)
    caption = re.sub(r"\&", " & ", caption)
    caption = re.sub(r"\*", " * ", caption)
    # tokenize the caption
    caption = get_tokenizer(tokenizer)(caption)
    caption = " ".join(caption.strip().split()) # remove duplicated spaces
    tokens = caption.strip().split()
    
    return tokens

def get_transform(target_size):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def reporthook(t):
    """
    https://github.com/tqdm/tqdm.
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

def unk_init(token, dim):
    '''
        For default:
            + <pad> is 0
            + <sos> is 1
            + <eos> is 2
            + <unk> is 3
    '''

    if token in ["<pad>", "<p>"]:
        return torch.zeros(dim)
    if token in ["<sos>", "<bos>", "<s>"]:
        return torch.ones(dim)
    if token in ["<eos>", "</s>"]:
        return torch.ones(dim) * 2
    return torch.ones(dim) * 3

def collate_fn(samples):
    image_ids = []
    filenames = []
    features = []
    boxes = []
    tokens = []
    captions = []
    shifted_right_tokens = []
    max_seq_len = 0
    for sample in samples:
        image_id = sample["image_id"]
        filename = sample["filename"]
        feature = sample["features"]
        box = sample["boxes"]
        token = sample["caption"] # for cross-entropy objective training
        shifted_right_token = sample["shifted_right_caption"] # for cross-entropy objective training
        caption = sample["captions"] # for self-critical sequential training

        if max_seq_len < feature.shape[0]:
            max_seq_len = feature.shape[0]

        if image_id is not None:
            image_ids.append(image_id)
        if filename is not None:
            filenames.append(filename)
        if box is not None:
            boxes.append(torch.tensor(box))
        if caption is not None:
            captions.append(caption)
        features.append(torch.tensor(feature))
        tokens.append(token)
        shifted_right_tokens.append(shifted_right_token)

    zero_feature = torch.zeros_like(features[-1][-1]).unsqueeze(0) # (1, dim)
    if len(boxes) > 0:
        zero_box = torch.zeros_like(boxes[-1][-1]).unsqueeze(0) # (1, 4)
    else:
        zero_box = None
    for batch_ith in range(len(samples)):
        for ith in range(features[batch_ith].shape[0], max_seq_len):
            features[batch_ith] = torch.cat([features[batch_ith], zero_feature], dim=0)
            if zero_box:
                boxes[batch_ith] = torch.cat([boxes[batch_ith], zero_box], dim=0)

    features = torch.cat([feature.unsqueeze_(0) for feature in features], dim=0)

    if len(image_ids) == 0:
        image_ids = None
    if len(filenames) == 0:
        filenames = None
    if len(boxes) > 0:
        boxes = torch.cat([box.unsqueeze_(0) for box in boxes], dim=0)
    else:
        boxes = None
    if len(captions) == 0:
        captions = None
    if len(captions) == 0:
        captions = None
    tokens = torch.cat([token.unsqueeze_(0) for token in tokens], dim=0)
    shifted_right_tokens = torch.cat([token.unsqueeze_(0) for token in shifted_right_tokens], dim=0)

    return defaultdict({
        "image_ids": image_ids,
        "filenames": filenames,
        "features": features, 
        "boxes": boxes, 
        "tokens": tokens, 
        "shifted_right_tokens": shifted_right_tokens,
        "captions": captions
    })