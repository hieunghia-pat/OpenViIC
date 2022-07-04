from models.modules.attentions import *
from models.language_models import *
from models.modules.encoders import *
from models.modules.decoders import *

from yacs.config import CfgNode
import yaml

Encoders = {
    "encoder": Encoder,
    "augmented-memory-encoder": AugmentedMemoryEncoder,
    "augmented-geometry-encoder": AugmentedGeometryEncoder,
    "dlct-encoder": DualCollaborativeLevelEncoder
}

Decoders = {
    "decoder": Decoder,
    "meshed-decoder": MeshedDecoder,
    "apdaptive-decoder": AdaptiveDecoder,
    "meshed-adaptive-decoder": MeshedAdaptiveDecoder
}

Pretrained_language_model_names = {
    "phobert-base": "vinai/phobert-base",
    "phobert-large": "vinai/phobert-large",
    "bartpho-syllable": "vinai/bartpho-syllable",
    "bartpho-word": "vinai/bartpho-word",
    "gpt2": "NlpHUST/gpt-neo-vi-small"
}

Pretrained_language_models = {
    "bert": BERTModel,
    "phobert": PhoBERTModel,
    # "bart_pho_model": BARTPhoModel,
    # "gpt_2": GPT2Model
}

Tokenizers = {
    "vncorenlp": "vncorenlp",
    "pyvi": "pyvi",
    "spacy": "spacy"
}

Word_embeddings = {
    "fasttex": "fasttext.vi.300d",
    "phow2v_syllable_100": "phow2v.syllable.100d",
    "phow2v_syllable_300": "phow2v.syllable.300d",
    "phow2v_word_100": "phow2v.word.100d",
    "phow2v_word_300": "phow2v.word.300d"
}

def get_encoder(vocab, config):
    encoder = Encoders[config.model.transformer.encoder.module]

    return encoder(N=config.model.nlayers, padding_idx=vocab.padding_idx, d_in=config.model.d_feature, 
                    d_model=config.model.d_model, d_k=config.model.d_k, d_v=config.model.d_v,
                    d_ff=config.model.d_ff, dropout=config.model.dropout,
                    **config.model.transformer.encoder.args)

def get_decoder(vocab, config):
    decoder = Decoders[config.model.transformer.decoder.module]

    return decoder(vocab_size=len(vocab), max_len=vocab.max_caption_length, N_dec=config.model.nlayers, 
                    padding_idx=vocab.padding_idx, d_model=config.model.d_model, d_k=config.model.d_k,
                    d_v=config.model.d_v, d_ff=config.model.d_ff, dropout=config.model.dropout,
                    **config.model.transformer.decoder.args)

def get_language_model(vocab, config):
    language_model = Pretrained_language_models[config.model.pretrained_language_model]
    return language_model(vocab, config.model.pretrained_language_model_name, padding_idx=1, 
                            language_model_hidden_size=config.model.language_model_hidden_size,
                            d_model=config.model.d_model, d_k=config.model.d_k, d_v=config.model.d_v, h=config.model.nheads,
                            d_ff=config.model.d_ff, max_len=vocab.max_caption_length, dropout=config.model.dropout)

def get_tokenizer(tokenizer_name: str):
    return Tokenizers[tokenizer_name]

def get_word_embedding(word_embedding_name: str):
    return Word_embeddings[word_embedding_name]

def get_config(yaml_file):
    return CfgNode(init_dict=yaml.load(open(yaml_file, "r"), Loader=yaml.FullLoader))