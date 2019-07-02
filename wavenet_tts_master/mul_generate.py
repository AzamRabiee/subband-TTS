# -*- coding: utf-8 -*-

import argparse
import time
import os
import audio
import math

import librosa
import numpy as np
import tensorflow as tf

from model import WaveNetModel
from hparams import hparams, hparams_debug_string
import nnmnkwii.preprocessing as P
from model.ops import lc_averaging, lc_upsampling_by_repeat
from tqdm import tqdm
from mlxtend.preprocessing import one_hot
import scipy.io as sio


LOGDIR = "./logdir"
CHECKPOINT = os.path.join(LOGDIR, "train/24/model.ckpt-199999")
WAV_OUT_PATH = os.path.join(LOGDIR, "generate/24")
# EVAL_TXT = os.path.join(hparams.NPY_DATAROOT+"_dev", "train.txt")
EVAL_TXT = os.path.join(hparams.NPY_DATAROOT+"_dev", "eval.txt")
SAVE_EVERY = None
TEMPERATURE = 1.0
GPU = "0,1,2,3,4,5,6,7"


def get_arguments():

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        '--checkpoint', type=str, default=CHECKPOINT, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=WAV_OUT_PATH,
        help='Directory in which to store the generated '
        'waves according to the eval txt file.')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--eval_txt',
        type=str,
        default=EVAL_TXT,
        help="the eval txt"
    )
    parser.add_argument(
        '--hparams',
        type=str,
        default=None,
        help="the override hparams"
    )
    arguments = parser.parse_args()
    return arguments


def write_wav(waveform, sample_rate, filename):
    while os.path.exists(filename):
        filename = "%s_.wav" % filename[:-4]
    y = np.array(waveform)
    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(filename, (y * maxv).astype(np.int16), sample_rate)
    # librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def main():
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    args = get_arguments()
    os.makedirs(args.wav_out_path, exist_ok=True)
    if args.hparams is not None:
        hparams.parse(args.hparams)
    if not hparams.gc_enable:
        hparams.global_cardinality = None
        hparams.global_channel = None
    print(hparams_debug_string())

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True)))

    net = WaveNetModel(
        batch_size=1,
        dilations=hparams.dilations,
        filter_width=hparams.filter_width,
        residual_channels=hparams.residual_channels,
        dilation_channels=hparams.dilation_channels,
        skip_channels=hparams.skip_channels,
        quantization_channels=hparams.quantization_channels,
        use_biases=hparams.use_biases,
        scalar_input=hparams.scalar_input,
        initial_filter_width=hparams.initial_filter_width,
        local_condition_channel=hparams.lc_channels,
        lc_initial_channels=hparams.lc_initial_channels,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_factor=hparams.upsample_factor,
        global_cardinality=hparams.global_cardinality,
        global_channel=hparams.global_channel,
        is_training=False
    )
    samples = tf.placeholder(tf.int32)
    local_ph = tf.placeholder(tf.float32, shape=(1, net.local_condition_channel))

    sess.run(tf.global_variables_initializer())
    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)
    print('Restore is done succesfully!')

    tmp_global_condition = None
    upsample_factor = audio.get_hop_size()

    generate_list = []
    with open(os.path.join(WAV_OUT_PATH+"_dev", args.eval_txt), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line is not None:
                line = line.strip().split('|')
                npy_path = os.path.join(hparams.NPY_DATAROOT+"_dev", line[1])
                local_condition = np.load(npy_path)
                if hparams.triphone:
                    pre_phone = one_hot(local_condition[0], num_labels=net.lc_initial_channels)
                    cur_phone = one_hot(local_condition[1], num_labels=net.lc_initial_channels)
                    nxt_phone = one_hot(local_condition[2], num_labels=net.lc_initial_channels)
                    tmp_local_condition = np.concatenate((pre_phone, cur_phone, nxt_phone), axis=1)
                else:
                    tmp_local_condition = one_hot(local_condition, num_labels=net.lc_initial_channels)

                h = tmp_local_condition
                tmp_local_condition = lc_averaging(tmp_local_condition)

                tmp_local_condition = tmp_local_condition.astype(np.float32)
                if len(line) == 5:
                    tmp_global_condition = int(line[4])
                if hparams.global_channel is None:
                    tmp_global_condition = None
                generate_list.append((tmp_local_condition, tmp_global_condition, line[1], h))

    for local_condition, global_condition, npy_path, h in generate_list:
        h_hat = local_condition
        wav_id = npy_path.split('-phone')[0]
        wav_out_path = os.path.join(args.wav_out_path, "{}_gen.wav".format(wav_id))

        if not hparams.upsample_conditional_features and hparams.lc_conv_layers < 1:
            local_condition = np.repeat(local_condition, upsample_factor, axis=0)
        elif hparams.upsample_conditional_features:
            local_condition = np.expand_dims(local_condition, 0)
            local_condition = net.create_upsample(local_condition)
            local_condition = tf.squeeze(local_condition, [0]).eval(session=sess)
        else:
            local_condition = np.expand_dims(local_condition, 0)
            local_condition, h_list = net._create_lc_conv_layer(local_condition)
            # h3 = local_condition.eval(session=sess)
            if hparams.lc_average:  # upsampling by repeat
                if hparams.lc_overlap:
                    lc_len = tf.shape(local_condition).eval(session=sess)
                    local_condition = lc_upsampling_by_repeat(local_condition, hparams.average_window_shift)
                    mod = math.ceil(lc_len[1] * hparams.average_window_shift / 256) * 256 - lc_len[1] * hparams.average_window_shift
                    edge = tf.slice(local_condition, [0, lc_len[1] * hparams.average_window_shift -1, 0], [-1, 1, -1])
                    edge = tf.tile(edge, [1, mod, 1])
                    local_condition = tf.concat([local_condition, edge], axis=1)
                else:
                    local_condition = lc_upsampling_by_repeat(local_condition, hparams.average_window_len)
            local_condition = tf.squeeze(local_condition).eval(session=sess)

        h1 = h_list[0].eval(session=sess)
        h2 = h_list[1].eval(session=sess)
        h3 = h_list[2].eval(session=sess)
        mat_out_path = os.path.join(args.wav_out_path, "{}.mat".format(wav_id))
        sio.savemat(mat_out_path, {'C': local_condition,
                                   'h': h,
                                   'h_hat': h_hat,
                                   'h1': h1,
                                   'h2': h2,
                                   'h3': h3
                                   })

        next_sample = net.predict_proba_incremental(samples, local_ph, global_condition)
        sess.run(net.init_ops)

        quantization_channels = hparams.quantization_channels

        # Silence with a single random sample at the end.
        waveform = [quantization_channels / 2] * (net.receptive_field - 1)
        waveform.append(np.random.randint(quantization_channels))

        sample_len = local_condition.shape[0]
        for step in tqdm(range(0, sample_len)):

            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window = waveform[-1]

            # Run the WaveNet to predict the next sample.
            prediction = sess.run(outputs, feed_dict={samples: window,
                                                      local_ph: local_condition[step:step+1, :]
                                                      })[0]

            # Scale prediction distribution using temperature.
            np.seterr(divide='ignore')
            scaled_prediction = np.log(prediction) / args.temperature
            scaled_prediction = (scaled_prediction -
                                 np.logaddexp.reduce(scaled_prediction))
            scaled_prediction = np.exp(scaled_prediction)
            np.seterr(divide='warn')
            # print(quantization_channels, scaled_prediction)
            sample = np.random.choice(
                np.arange(quantization_channels), p=scaled_prediction)
            waveform.append(sample)

            # If we have partial writing, save the result so far.
            if (wav_out_path and args.save_every and
                            (step + 1) % args.save_every == 0):
                out = P.inv_mulaw_quantize(np.array(waveform), quantization_channels)
                write_wav(out, hparams.sample_rate, wav_out_path)

                # Introduce a newline to clear the carriage return from the progress.
        print()
        # Save the result as a wav file.
        if wav_out_path:
            out = P.inv_mulaw_quantize(np.array(waveform).astype(np.int16), quantization_channels)
            # out = P.inv_mulaw_quantize(np.asarray(waveform), quantization_channels)
            write_wav(out, hparams.sample_rate, wav_out_path)
    end = time.time()
    print('Finished generating.')
    print('It took %.2f seconds' % (end - start))

if __name__ == '__main__':
    main()


