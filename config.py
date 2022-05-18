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

# Pre-train language model

'''
vinai/phobert-base
vinai/phobert-large
vinai/bartpho-syllable
vinai/bartpho-word
NlpHUST/gpt-neo-vi-small
'''

pretrained_language_model_name = "vinai/phobert-base" 

'''
PhoBERTModel
BARTPhoModel
ViGPTModel
'''

pretrained_language_model = PhoBERTModel
# pretrained_language_model = BERTModel

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
    "pretrained_language_model": pretrained_language_model,
    "pretrained_language_model_path": '/content/drive/MyDrive/DoanhNghia/OpenViIC/saved_models/rstnet_using_region/best_language_model.pth'
}
encoder = Encoder
decoder = AdaptiveDecoder
transformer_args = {"use_img_pos": True, "use_box_embedd": True}

# dataset configuration
train_json_path = "/content/drive/MyDrive/DoanhNghia/uit-vlsp-viecap4h-solution/annotations/train.json"
val_json_path = "/content/drive/MyDrive/DoanhNghia/uit-vlsp-viecap4h-solution/annotations/val.json"
public_test_json_path = None
private_test_json_path = None

# feature path
region_features_path = '/content/drive/MyDrive/DoanhNghia/uit-vlsp-viecap4h-solution/features/'
grid_features_path = '/content/drive/MyDrive/DoanhNghia/uit-vlsp-viecap4h-solution/X152++_VieCap_feature.hdf5'
mask_features_path = '/content/drive/MyDrive/DoanhNghia/uit-vlsp-viecap4h-solution/DLCT_masks/'

# training configuration
batch_size = 32
workers = 2

# tokenizer:
'''
+ vncorenlp
+ pyvi
+ spacy
'''
tokenizer = "vncorenlp" 

# Word embedding:
'''
+ fasttext.vi.300d
+ phow2v.syllable.100d
+ phow2v.syllable.300d
+ phow2v.word.100d
+ phow2v.word.300d
'''
word_embedding = None   

# sample submission configuration
sample_public_test_json_path = None
sample_private_test_json_path = None

# Type features used
guided_load_feature = {
    'grid': True,
    'region': False
}

# idx by filename
idx_by_filename = True