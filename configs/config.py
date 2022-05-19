from yacs.config import CfgNode as CN

config = CN()

config.path = CN()
config.path.train_json_path = "features/annotations/OpenViVQA/openvivqa_train.json"
config.path.dev_json_path = "features/annotations/OpenViVQA/openvivqa_dev.json"
config.path.test_json_path = "features/annotations/OpenViVQA/openvivqa_test.json"
config.path.image_features_path = "features/region_features/OpenViVQA/faster_rcnn"
config.path.images_path = ""

# training configuration
config.training = CN()
config.training.start_from = None
config.training.learning_rate = 1.
config.training.warmup = 10000
config.training.get_scores = False

# transformer-based model configuration
config.model = CN()
config.model.model_name = ""
config.model.nhead = 8
config.model.nlayers = 3
config.model.d_model = 512
config.model.d_k = 64
config.model.d_v = 64
config.model.d_ff = 2048
config.model.d_feature = 2048
config.model.dropout = .1
config.model.training_beam_size = 5
config.model.evaluating_beam_size = 3

config.model.transformer = CN()
config.model.transformer.transformer_args = CN()
config.model.transformer.transformer_args.use_img_pos = False
config.model.transformer.transformer_args.use_box_embedd = False

config.model.transformer.encoder = CN()
config.model.transformer.encoder.encoder_module = "none"  # encoder
                                                          # none
config.model.transformer.encoder.encoder_layer_module = "none" # encoder
                                                        # multilevel encoder
                                                        # none
config.model.transformer.encoder.encoder_self_attention_module = "none"  # scaled_dot_product_attention
                                                                    # augmented_geometry_scaled_dot_product_attention
                                                                    # augmented_memory_scaled_dot_product_attention
                                                                    # apdative_scaled_dot_product_attention
                                                                    # none
config.model.transformer.encoder.encoder_self_attention_args = CN()
config.model.transformer.encoder.encoder_args = CN()
config.model.transformer.encoder.encoder_args.multi_level_output = True


config.model.transformer.decoder = CN()
config.model.transformer.decoder.language_model = CN()   
config.model.transformer.decoder.pretrained_language_model_name = None  # phobert_base
                                                                        # phobert_large
                                                                        # bartpho_syllable
                                                                        # bartpho_word

config.model.transformer.decoder.language_model.language_model_hidden_size = 768
config.model.transformer.decoder.language_model.pretrained_language_model_path = ""

config.model.transformer.decoder.decoder_module = "none"  # decoder
                                                          # meshed_decoder
                                                          # adaptive_decoder
                                                          # none

config.model.transformer.decoder.decoder_layer_module = "none" # decoder
                                                        # meshed_decoder
                                                        # adaptive_decoder
                                                        # none

config.model.transformer.decoder.decoder_self_attention_module = "none"  # scaled_dot_product_attention
                                                                  # augmented_geometry_scaled_dot_product_attention
                                                                  # augmented_memory_scaled_dot_product_attention
                                                                  # apdative_scaled_dot_product_attention
                                                                  # none

config.model.transformer.decoder.decoder_enc_attention_module = "none" # scaled_dot_product_attention
                                                                # augmented_geometry_scaled_dot_product_attention
                                                                # augmented_memory_scaled_dot_product_attention
                                                                # apdative_scaled_dot_product_attention
                                                                # none
config.model.transformer.decoder.decoder_self_attention_args = CN()
config.model.transformer.decoder.decoder_enc_attention_args = CN()
config.model.transformer.decoder.decoder_args = CN()

# dataset configuration
config.dataset = CN()
config.dataset.batch_size = 32
config.dataset.workers = 2
config.dataset.tokenizer = "none" # vncorenlp
                                  # pyvi
                                  # spacy
                                  # none
config.dataset.word_embedding = "none"  # fasttext.vi.300d
                                        # phow2v.syllable.100d
                                        # phow2v.syllable.300d
                                        # phow2v.word.100d
                                        # phow2v.word.300d
                                        # none
config.dataset.min_freq = 1

def get_default_config():
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return config.clone()