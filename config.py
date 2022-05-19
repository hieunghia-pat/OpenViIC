from data_utils.dataset import *
from data_utils.utils import *
from models.modules.language_models import *
from models.modules.attentions import *
from models.modules.encoders import *
from models.modules.decoders import *

# Root dir
root_path = 'features_for_exp'

# Training configuration
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

# Model configuration
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

language_model_hidden_size = 768

# Encoder configuration
encoder_self_attention = ScaledDotProductWithBoxAttention
encoder_self_attention_args = {}
encoder_args = {'multi_level_output': True}

# Decoder configuration
decoder_self_attention = ScaledDotProductAttention
decoder_enc_attention = ScaledDotProductAttention
decoder_self_attention_args = {}
decoder_enc_attention_args = {}
decoder_args = {
    "pretrained_language_model_name": pretrained_language_model_name,
    "pretrained_language_model": pretrained_language_model,
    "pretrained_language_model_path": '/content/drive/MyDrive/viecap4h-experiments/OpenViIC/saved_models/rstnet_using_region/best_language_model.pth'
}

# Transformer configuration
encoder = DualCollaborativeLevelEncoder
decoder = MeshedAdaptiveDecoder
transformer_args = {"use_img_pos": True, "use_box_embedd": True}

# Dataset configuration
train_json_path = root_path + "/annotations/train.json"
val_json_path = root_path + "/annotations/val.json"
public_test_json_path = None
private_test_json_path = None

# Feature paths
region_features_path = root_path + '/region_features_butd/'
grid_features_path = root_path + '/X152++_VieCap_feature.hdf5'
mask_features_path = root_path + '/DLCT_masks/'

# Training configuration
batch_size = 32
workers = 2

# Tokenizer:
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

# Sample submission configuration
sample_public_test_json_path = None
sample_private_test_json_path = None

# Type features used
guided_load_feature = {
    'grid': True,
    'region': True
}

# Idx by filename
idx_by_filename = True