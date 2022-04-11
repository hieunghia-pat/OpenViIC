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
xe_base_lr = 1e-4
rl_base_lr = 5e-6
refine_epoch_rl = 28
xe_least = 15
xe_most = 20
min_freq = 1

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
beam_size = 5
model_name = "transformer_using_region"
pretrained_language_model_name = None                   # vinai/phobert-base
                                                        # vinai/phobert-large
                                                        # vinai/bartpho-syllable
                                                        # vinai/bartpho-word
                                                        # NlpHUST/gpt-neo-vi-small
pretrained_language_model = None    # PhoBERTModel
                                    # BARTPhoModel
                                    # ViGPTModel

language_model_hidden_size = 768
encoder_self_attention = AugmentedMemoryScaledDotProductAttention
encoder_self_attention_args = {"m": total_memory}
encoder_args = {}
decoder_self_attention = ScaledDotProductAttention
decoder_enc_attention = ScaledDotProductAttention
decoder_self_attention_args = {}
decoder_enc_attention_args = {}
decoder_args = {"N_enc": nlayers}
encoder = MultiLevelEncoder
decoder = MeshedDecoder
transformer_args = {}

# dataset configuration
train_json_path = "features/annotations/vieCap4H/viecap4h_captions_train2017.json"
val_json_path = "features/annotations/vieCap4H/viecap4h_captions_val2017.json"
public_test_json_path = "features/annotations/vieCap4H/viecap4h_captions_public_test2017.json"
private_test_json_path = "features/annotations/vieCap4H/viecap4h_captions_private_test2017.json"
feature_path = "features/region_features/vieCap4H/faster_rcnn"
batch_size = 16
workers = 2
tokenizer = None    # vncorenlp
                    # pyvi
                    # spacy
word_embedding = None   # "fasttext.vi.300d"
                        # "phow2v.syllable.100d"
                        # "phow2v.syllable.300d"
                        # "phow2v.word.100d"
                        # "phow2v.word.300d"

# sample submission configuration
sample_public_test_json_path = "features/annotations/vieCap4H/sample_public_test_submission.json"
sample_private_test_json_path = "features/annotations/vieCap4H/sample_private_test_submission.json"