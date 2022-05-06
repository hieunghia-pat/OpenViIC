from data_utils.dataset import *
from data_utils.utils import *
from models.modules.language_models import *
from models.modules.attentions import *
from models.modules.encoders import *
from models.modules.decoders import *

# training configuration
checkpoint_path = "saved_models"
start_from = None
learning_rate = 1.
epochs = 20
warmup = 10000
# xe_base_lr = 1e-4
# rl_base_lr = 5e-6
# refine_epoch_rl = 28
min_freq = 1
get_scores = False

# model configuration
total_memory = 40
nhead = 8
nlayers = 3
d_model = 512
d_k = 64
d_v = 64
d_ff = 2048
d_feature = 2048
dropout = .1
training_beam_size = 5
evaluating_beam_size = 3
model_name = "rstnet_using_region"
pretrained_language_model_name = "vinai/phobert-base"                   # vinai/phobert-base
                                                        # vinai/phobert-large
                                                        # vinai/bartpho-syllable
                                                        # vinai/bartpho-word
                                                        # NlpHUST/gpt-neo-vi-small
pretrained_language_model = PhoBERTModel    # PhoBERTModel
                                    # BARTPhoModel
                                    # ViGPTModel

language_model_hidden_size = 768
encoder_self_attention = ScaledDotProductAttention
encoder_self_attention_args = {}
encoder_args = {}
decoder_self_attention = ScaledDotProductAttention
decoder_enc_attention = ScaledDotProductAttention
decoder_self_attention_args = {}
decoder_enc_attention_args = {}
decoder_args = {
    "pretrained_language_model_name": pretrained_language_model_name,
    "pretrained_language_model": pretrained_language_model
}
encoder = Encoder
decoder = Decoder
transformer_args = {"use_img_pos": True}

# dataset configuration
train_json_path = "features/annotations/UIT-ViIC/uitviic_captions_train2017.json"
val_json_path = "features/annotations/UIT-ViIC/uitviic_captions_val2017.json"
public_test_json_path = "features/annotations/UIT-ViIC/uitviic_captions_test2017.json"
private_test_json_path = None
feature_path = "features/region_features/UIT-ViIC/faster_rcnn"
batch_size = 32
workers = 2
tokenizer = "vncorenlp"    # vncorenlp
                    # pyvi
                    # spacy
word_embedding = None   # "fasttext.vi.300d"
                        # "phow2v.syllable.100d"
                        # "phow2v.syllable.300d"
                        # "phow2v.word.100d"
                        # "phow2v.word.300d"

# sample submission configuration
sample_public_test_json_path = None
sample_private_test_json_path = None