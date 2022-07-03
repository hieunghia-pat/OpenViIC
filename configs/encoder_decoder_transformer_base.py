from yacs.config import CfgNode as CN

config = CN()

## configuration for paths
config.path = CN()
config.path.train_json_path = "features/annotations/vieCap4H/viecap4h_captions_train2017.json"
config.path.dev_json_path = "features/annotations/vieCap4H/viecap4h_captions_val2017.json"
config.path.public_test_json_path = "features/annotations/vieCap4H/viecap4h_captions_public_test2017.json"
config.path.private_test_json_path = "features/annotations/vieCap4H/viecap4h_captions_private_test2017.json"
config.path.image_features_path = "features/hybrid_features/vieCap4H/faster_rcnn_x152++"
config.path.images_path = None

## configuration for training
config.training = CN()
config.training.checkpoint_path = "saved_models"
config.training.start_from = None
config.training.learning_rate = 1.
config.training.warmup = 10000
config.training.get_scores = False
config.training.training_beam_size = 5
config.training.evaluating_beam_size = 5
config.training.using_features = "region" # region
                                          # grid
                                          # region+grid

## model configuration
config.model = CN()
config.model.name = "base_transformer"
config.model.nhead = 8
config.model.nlayers = 3
config.model.d_model = 512
config.model.d_k = 64
config.model.d_v = 64
config.model.d_ff = 2048
config.model.d_feature = 2048
config.model.dropout = .5
config.model.embedding_dim = 300

## pretrained language model components
config.model.pretrained_language_model_name = None  # phobert-base
                                                    # phobert-large
                                                    # bartpho-syllable
                                                    # bartpho-word    
                                                    # gpt2  
config.model.pretrained_language_model = None # bert-model
                                              # pho-bert-model
                                              # None
config.model.pretrained_language_model_path = None
config.model.language_model_hidden_size = 768

config.model.transformer = CN()
config.model.transformer.encoder = CN()
config.model.transformer.encoder.args = CN()
config.model.transformer.encoder.args.use_aoa = False
config.model.transformer.encoder.args.multi_level_output = False
config.model.transformer.encoder.module = "encoder" # encoder
                                                    # dlct-encoder

config.model.transformer.decoder = CN()
config.model.transformer.decoder.args = CN()
config.model.transformer.decoder.args.use_aoa = False
config.model.transformer.decoder.module = "decoder" # decoder
                                                    # meshed-decoder
                                                    # adaptive-decoder

## dataset configuration
config.dataset = CN()
config.dataset.batch_size = 32
config.dataset.workers = 0
config.dataset.tokenizer = None # vncorenlp
                                # pyvi
                                # spacy
                                # None
config.dataset.word_embedding = None  # fasttext.vi.300d
                                      # phow2v.syllable.100d
                                      # phow2v.syllable.300d
                                      # phow2v.word.100d
                                      # phow2v.word.300d
                                      # None
config.dataset.min_freq = 1

def get_default_config():
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return config.clone()