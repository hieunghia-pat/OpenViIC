from models.modules.attentions import *
from models.modules.language_models import *
from models.modules.encoders import *
from models.modules.decoders import *

encoders = {
    "encoder": Encoder,
    "dlct-encoder": DualCollaborativeLevelEncoder
}

decoders = {
    "decoder": Decoder,
    "meshed_decoder": MeshedDecoder,
    "apdaptive-decoder": AdaptiveDecoder,
    "meshed-adaptive-decoder": MeshedAdaptiveDecoder,
}

pretrained_language_model_names = {
    "phobert-base": "vinai/phobert-base",
    "phobert-large": "vinai/phobert-large",
    "bartpho-syllable": "vinai/bartpho-syllable",
    "bartpho-word": "vinai/bartpho-word",
    "gpt2": "NlpHUST/gpt-neo-vi-small",
    None: None
}

pretrained_language_model = {
    "bert-model": BERTModel,
    "pho-bert-model": PhoBERTModel,
    # "bart_pho_model": BARTPhoModel,
    # "gpt_2": GPT2Model,
    None: None
}

word_embedding = {
    "fasttex": "fasttext.vi.300d",
    "phow2v_syllable_100": "phow2v.syllable.100d",
    "phow2v_syllable_300": "phow2v.syllable.300d",
    "phow2v_word_100": "phow2v.word.100d",
    "phow2v_word_300": "phow2v.word.300d",
    None: None
}