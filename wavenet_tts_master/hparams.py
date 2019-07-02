# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    name="wavenet_tts",

    # output of preprocessing; it's used in train.py
    NPY_DATAROOT="/data4/azam/Datasets/LJSpeech-1.1/trimmed/wavenet_tts",
    # NPY_DATAROOT="/media/cnsl/Datasets/LJSpeech-1.1/trimmed/wavenet_tts",
    # NPY_DATAROOT="/media/cnsl/Datasets/LJSpeech-1.1/trimmed/wavenet_tts_triphone",

    # Audio:
    sample_rate=16000,
    silence_threshold=2,  # in terms of quantization index
    num_mels=80,
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
    # dilations=[1, 2, 4, 8, 16, 32, 64, 128,
    #            1, 2, 4, 8, 16, 32, 64, 128],
    dilations=[1, 2, 4, 8, 16, 32,
               1, 2, 4, 8, 16, 32,
               1, 2, 4, 8, 16, 32,
               1, 2, 4, 8, 16, 32],
    # dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    #            1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    #            1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    #            1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    #            1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    residual_channels=32,
    dilation_channels=128, # 128 is desired
    quantization_channels=256,
    skip_channels=256,
    use_biases=True,
    scalar_input=False,
    initial_filter_width=32,

    MOVING_AVERAGE_DECAY=0.9999,

    # local conditions
    lc_average=True,
    lc_overlap=False,
    average_window_len=256,     # 256(16ms) if lc_overlap=False, otherwise 400(25ms)
    average_window_shift=256,   # 160(10ms) if lc_overlap=True, otherwise average_window_len
    lc_conv_layers=3,  # number of local conditioning convolutional layers
    lc_bias=True,
    lc_channels=256,
    lc_skip_connection=False,
    lc_initial_channels=70,     # number of phonems; if triphone, times it by 3
    lc_fw=5,
    lc_causal_conv=False,
    triphone=False,
    upsample_conditional_features=False,  # with conv transpose
    upsample_factor=[16, 16],

    LEARNING_RATE_DECAY_FACTOR=0.5,
    NUM_STEPS_RATIO_PER_DECAY=0.3,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
