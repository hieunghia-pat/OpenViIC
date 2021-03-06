import torch
from torch import nn

from data_utils.types import *
from configs.constants import *

import copy
from yacs.config import CfgNode

from data_utils.vocab import Vocab
from models.modules.transformer import Transformer

def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.size(0)
    else:
        b_s = x[0].size(0)
    return b_s

def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.device
    else:
        b_s = x[0].device
    return b_s

def positional_embedding(input, d_model) -> torch.Tensor:
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model, padding_idx=None) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32)
    out = positional_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

def clones(module, n):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def generate_padding_mask(sequences: torch.IntTensor, padding_idx: int) -> torch.BoolTensor:
    '''
        sequences: (bs, seq_len)
    '''
    mask = (sequences == padding_idx) # (b_s, seq_len)

    return mask

def generate_sequential_mask(seq_len: int) -> torch.BoolTensor:
    '''
        Mask out subsequent positions
    '''
    attn_shape = (seq_len, seq_len)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).to(torch.bool)

    return subsequent_mask

def get_relative_pos(x, batch_size, norm_len):
    x = x.view(1, -1, 1).expand(batch_size, -1, -1)
    return  x / norm_len

def get_grids_position(batch_size, seq_len, grid_size):
    assert seq_len == grid_size[0] * grid_size[1]

    # record the pos of each grid according to the form of region box
    x = torch.arange(0, grid_size[0]).float().cuda()
    y = torch.arange(0, grid_size[1]).float().cuda()

    px_min = x.view(-1, 1).expand(-1, grid_size[0]).contiguous().view(-1)
    py_min = y.view(1, -1).expand(grid_size[1], -1).contiguous().view(-1)

    px_max = px_min + 1
    py_max = py_min + 1

    # scale pos into the range (0 ~ 1)
    rpx_min = get_relative_pos(px_min, batch_size, grid_size[0])
    rpy_min = get_relative_pos(py_min, batch_size, grid_size[1])

    rpx_max = get_relative_pos(px_max, batch_size, grid_size[0])
    rpy_max = get_relative_pos(py_max, batch_size, grid_size[1])

    boxes = torch.cat([rpx_min, rpy_min, rpx_max, rpy_max], dim=-1) # (bs, n, 4)

    return boxes

def box_relational_embedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)

    x_min, y_min, x_max, y_max = torch.chunk(f_g, chunks=4, dim=-1) # each tensor has dimension of (batch_size, max_nr_bounding_boxes, 1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1, -1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).to(f_g.device)
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
        
    return embedding # (batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, dim_g)

def get_encoder(model_config: CfgNode, encoder_config: CfgNode, vocab: Vocab):
    encoder = encoders[encoder_config.encoder_module]
    encoder_self_attention = attentions[encoder_config.encoder_self_attention_module]
    encoder_self_attention_args = dict(encoder_config.encoder_self_attention_args)
    encoder_args = dict(encoder_config.encoder_args)

    return encoder(N=model_config.nlayers, padding_idx=vocab.padding_idx, d_in=model_config.d_feature, d_model=model_config.d_model, d_k=model_config.d_k, 
                    d_v=model_config.d_v, d_ff=model_config.d_ff, dropout=model_config.dropout, 
                    attention_module=encoder_self_attention, 
                    attention_module_kwargs=encoder_self_attention_args, 
                    **encoder_args)

def get_decoder(model_config: CfgNode, decoder_config: CfgNode, vocab: Vocab):
    decoder = decoders[decoder_config.decoder_module]
    decoder_self_attention = attentions[decoder_config.decoder_self_attention_module]
    decoder_self_attention_args = dict(decoder_config.decoder_self_attention_args)
    decoder_enc_attention = attentions[decoder_config.decoder_enc_attention_module]
    decoder_enc_attention_args = dict(decoder_config.decoder_enc_attention_args)
    decoder_args = {
        **dict(decoder_config.decoder_args),
        **dict(decoder_config.language_model)
    }

    return decoder(vocab_size=len(vocab), max_len=vocab.max_caption_length, N_dec=model_config.nlayers, padding_idx=vocab.padding_idx,
                            d_model=model_config.d_model, d_k=model_config.d_k, d_v=model_config.d_v, d_ff=model_config.d_ff, dropout=model_config.dropout,
                            self_att_module=decoder_self_attention, enc_att_module=decoder_enc_attention,
                            self_att_module_kwargs=decoder_self_attention_args, enc_att_module_kwargs=decoder_enc_attention_args, **decoder_args)

def get_captioning_model(config: CfgNode, vocab: Vocab):

    model_config = config.model
    transformer_config = config.transformer
    encoder_config = transformer_config.encoder
    decoder_config = transformer_config.decoder

    encoder = get_encoder(model_config, encoder_config, vocab)
    decoder = get_decoder(model_config, decoder_config, vocab)

    # init Transformer model.
    captioning_model = Transformer(vocab.bos_idx, encoder, decoder, **config.transformer_args)

    return captioning_model

def get_language_model(config: CfgNode, vocab: Vocab):
    model_config = config.model
    language_model_config = config.model.transformer.decoder.language_model
    language_model = pretrained_language_model(language_model_config.pretrained_language_model_name, padding_idx=vocab.padding_idx, 
                                language_model_hidden_size=language_model_config.language_model_hidden_size,
                                vocab_size=len(vocab), d_model=model_config.d_model, d_k=model_config.d_k, d_v=model_config.d_v, 
                                h=model_config.nhead, d_ff=model_config.d_ff, max_len=vocab.max_caption_length, dropout=model_config.dropout)

    return language_model