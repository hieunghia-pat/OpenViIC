UIT-OpenIC - UIT Open Image Captioning
====

This is an open-source repository for researching Image Captioning task (in Vietnamese currently). This repo implements most recent transformer-based state-of-the-art methods on the MS COCO Image Captioning datataset to evaluate them on the first two Vietnamese Image Captioning datasets [UIT-ViIC](https://arxiv.org/pdf/2002.00175.pdf) and [vieCap4H](https://people.cs.umu.se/sonvx/files/VieCap4H_VLSP21.pdf).

## Implemented modules
We implemented most recent state-of-the-art (SOTA) transformer-based methods for image captioning on the MS-COCO image captioning dataset. For more detail, in this repo we conducted various encoder and decoder modules which are proposed by recent SOTA methods and you can compose properly together to get a novel method for experiments. For example, you can combine the encoder module equipped AugmentedGeometryScaledDotProductAttention together with the decoder equipped ScaleDotProductAttention to get the [Object Relation Transformer](https://arxiv.org/pdf/1906.05963.pdf) method.

Specifically, our repo have implemented the following attention-based modules:
- [ScaledDotProductAttention]()
- [AugmentedGeometryScaledDotProductAttention]()
- [AugmentedMemoryScaledDotProductAttention]()
- [AdaptiveScaledDotProductAttention]()

For encoder modules, we have implemented:
- [Transformer-based Encoder](https://arxiv.org/pdf/1706.03762.pdf) module
- [Transformer-based MultiLevelEncoder](https://arxiv.org/pdf/1912.08226.pdf) module

For decoder modules, we have implemented:
- [Transformer-based Decoder](https://arxiv.org/pdf/1706.03762.pdf) module
- [Transformer-based MeshedDecoder](https://arxiv.org/pdf/1912.08226.pdf) module
- [Transformer-based AdaptiveDecoder]() module

## Data preparation

### Annotation files

## Configuring the training process
All configurations of training process are defined in [config.py](config.py). To conducted any transformer-based method, you must defined the encoder_self_attention module for the encoder and its additional arguments, defined the decoder_self_attention and decoder_enc_attention for the decoder module and also specify its additional arguments. For example, when you want to conduct the [Meshed-Memory Transformer](https://arxiv.org/pdf/1912.08226.pdf), you have to conduct the model in the configuration file as follow:

```python
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
```

For more information about the architecture and arguments of each module, please visit [models/modules](models/modules/) directory.

## Current approaches used in this project

### Feature Representation

#### Region-based visual feature using [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf)

#### [Grid-based visual feature](https://arxiv.org/pdf/2001.03615.pdf)

### Transformer-based methods

#### [Attention on Attention Network](https://arxiv.org/pdf/1908.06954.pdf)

#### [Object Relation Transformer](https://arxiv.org/pdf/1906.05963.pdf)

#### [Meshed-Memory Transformer](https://arxiv.org/pdf/1912.08226.pdf)

#### [RSTNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_RSTNet_Captioning_With_Adaptive_Attention_on_Visual_and_Non-Visual_Words_CVPR_2021_paper.pdf)

## Contact
This project constructed under instruction of the NLP@UIT research group. For more information about the NLP@UIT group or relevant publications, please visit [http://nlp.uit.edu.vn/](http://nlp.uit.edu.vn/).

 - Nghia Hieu Nguyen: [19520178@gm.uit.edu.vn](mailto:19520178@gm.uit.edu.vn)
 - Duong T.D Vo: [19520483@gm.uit.edu.vn](mailto:19520483@gm.uit.edu.vn)
 - Minh-Quan Ha: [19522076@gm.uit.edu.vn](mailto:19522076@gm.uit.edu.vn)
