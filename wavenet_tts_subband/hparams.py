# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    name="subband_wavenet_tts",
    # output of preprocessing; it's used in train.py
    NPY_DATAROOT="/data4/azam/Datasets/LJSpeech-1.1/sub_datasets_redundant_integrated/train",
    # NPY_DATAROOT="/media/cnsl/Datasets/LJSpeech-1.1/sub_datasets_redundant_integrated/train",
    # NPY_DATAROOT="/media/cnsl/Datasets/LJSpeech-1.1/sub_datasets_redundant_integrated/dev",
    # NPY_DATAROOT="/data4/azam/Datasets/LJSpeech-1.1/sub_datasets_redundant_integrated/dev",

    # subbands
    subband_max=[0.8399, 0.5664, 1.0406, 2.4114,  # from d1 to d8
                 2.4463, 1.1815, 0.1377, 0.0207],
    joint_post_processing=False,

    # Audio:
    sample_rate=16000,
    silence_threshold=2,  # in terms of quantization index
    num_mels=80,  # it's not number of mel anymore; it's number of phonemes
    fft_size=1024,
    # shift can be specified by either hop_size or frame_shift_ms
    hop_size=256,    # 256 equals 16 ms for sr=16kHz
    frame_shift_ms=None,
    min_level_db=-100,
    ref_level_db=20,

    # global condition if False set global channel to None
    gc_enable=False,    # True for CMU_ARCTIC; False for LJSpeech
    global_channel=None,  # 16 for CMU_ARCTIC
    global_cardinality=7,  # speaker num

    filter_width=2,
    dilations=[1, 2, 4, 8, 16],
    residual_channels=256,
    dilation_channels=256, # 128 is preferred
    quantization_channels=256,
    skip_channels=256,
    use_biases=True,
    scalar_input=False,
    initial_filter_width=32,

    MOVING_AVERAGE_DECAY=0.9999,

    # local conditions
    lc_average=True,
    lc_overlap=False,
    average_window_len=256,
    average_window_shift=256,
    lc_conv_layers=3,
    lc_bias=True,
    lc_channels=256,
    lc_skip_conncetion=False,
    lc_initial_channels=70,
    lc_fw=5,
    upsample_conditional_features=False,
    upsample_factor=[16, 16],

    LEARNING_RATE_DECAY_FACTOR=0.5,
    NUM_STEPS_RATIO_PER_DECAY=0.3,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
