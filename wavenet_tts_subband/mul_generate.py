# -*- coding: utf-8 -*-

import argparse
import os
import audio
import librosa
import numpy as np
import tensorflow as tf
import queue
from model import WaveNetModel
from hparams import hparams, hparams_debug_string
import nnmnkwii.preprocessing as P
from model.ops import denormalize, lc_averaging, lc_upsampling_by_repeat, normalize
from tqdm import tqdm
from mlxtend.preprocessing import one_hot
import scipy.io as sio
import time, threading


LOGDIR = "./logdir"
CHECKPOINT = os.path.join(LOGDIR, "train/4/model.ckpt-200000")    # d1
MATH_OUT_PATH = os.path.join(LOGDIR, "generate/4")
EVAL_TXT = os.path.join(hparams.NPY_DATAROOT, "eval1.txt")
SAVE_EVERY = None
TEMPERATURE = 1.0
GPU = "5"

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
        default=MATH_OUT_PATH,
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
    y = np.array(waveform)
    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(filename, (y * maxv).astype(np.int16), sample_rate)
    # librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def generate_subband(subband_id, checkpoint, local_condition, global_condition, subband_queue, temperature=TEMPERATURE):
    print('generating subband d%d' % subband_id)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True)))
    restore_net_start = time.time()
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
    sess.run(tf.global_variables_initializer())
    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint)
    restore_net_end = time.time()
    print('Restore is done succesfully for d%d!' % subband_id)
    print('it took %.2f (sec.)' % (restore_net_end - restore_net_start))

    samples = tf.placeholder(tf.int32)  # 8 subbands
    local_ph = tf.placeholder(tf.float32, shape=(1, net.local_condition_channel))
    subband = 'd%d' % (subband_id + 1)
    next_sample = net.predict_proba_incremental(subband, samples, local_ph, global_condition)
    sess.run(net.init_ops)
    # Silence with a single random sample at the end.
    waveform = [net.quantization_channels / 2] * (net.receptive_field - 1)
    waveform.append(np.random.randint(net.quantization_channels))

    sample_len = local_condition.shape[0]
    for step in range(0, sample_len):
        outputs = [next_sample]
        outputs.extend(net.push_ops)
        window = waveform[-1]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window,
                                                  local_ph: local_condition[step:step + 1, :]
                                                  })[0]
        # Scale prediction distribution using temperature.
        # prediction = np.random.randint(0, 255, 256)
        np.seterr(divide='ignore')
        scaled_prediction = np.log(prediction) / temperature
        scaled_prediction = (scaled_prediction -
                             np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        np.seterr(divide='warn')
        # print(quantization_channels, scaled_prediction)
        sample = np.random.choice(
            np.arange(net.quantization_channels), p=scaled_prediction)

        waveform.append(sample)
    subband_queue.put([subband_id, waveform])
    subband_queue.task_done()

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
    restore_net_start = time.time()
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
    sess.run(tf.global_variables_initializer())
    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)
    print('Restore is done succesfully!')
    restore_net_end = time.time()
    print('it took %.2f (sec.)' % (restore_net_end - restore_net_start))

    tmp_global_condition = None
    upsample_factor = audio.get_hop_size()

    generate_list = []
    with open(args.eval_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line is not None:
                line = line.strip().split('|')
                target = np.load(os.path.join(hparams.NPY_DATAROOT, line[0]))
                npy_path = os.path.normpath(os.path.join(hparams.NPY_DATAROOT, line[1]))
                tmp_local_condition = one_hot(np.load(npy_path), num_labels=hparams.lc_initial_channels)
                h = tmp_local_condition
                tmp_local_condition = lc_averaging(tmp_local_condition)
                tmp_local_condition = tmp_local_condition.astype(np.float32)
                if len(line) == 5:
                    tmp_global_condition = int(line[4])
                if hparams.global_channel is None:
                    tmp_global_condition = None
                generate_list.append((tmp_local_condition, tmp_global_condition, line[1], target, h))

    for local_condition, global_condition, npy_path, target, h in generate_list:
        wav_id = npy_path.split('/')[-1]
        wav_id = wav_id.split('-phone')[0]
        wav_id = wav_id.replace("-", "_")
        mat_out_path = os.path.normpath(os.path.join(args.wav_out_path, "{}.mat".format(wav_id)))

        if not hparams.upsample_conditional_features and hparams.lc_conv_layers < 1:
            local_condition = np.repeat(local_condition, upsample_factor, axis=0)
        elif hparams.upsample_conditional_features:
            local_condition = np.expand_dims(local_condition, 0)
            local_condition = net.create_upsample(local_condition)
            local_condition = tf.squeeze(local_condition, [0]).eval(session=sess)
        else:
            local_condition = np.expand_dims(local_condition, 0)
            local_condition, h_list = net._create_lc_conv_layer(local_condition)
            if hparams.lc_average and not hparams.lc_overlap:
                local_condition = lc_upsampling_by_repeat(local_condition, hparams.average_window_len)
            local_condition = tf.squeeze(local_condition).eval(session=sess)
        print('conditional features are made sucessfully!')

        # option1: ----- without threading ----
        # for i in range(8):
        #     generate_subband(i, args.checkpoint, local_condition, global_condition, subband_q, args.temperature)

        # option2: ----- with threading ----
        subband_q = queue.Queue()
        threads = [threading.Thread(target=generate_subband,
                                    args=(i, args.checkpoint, local_condition, global_condition, subband_q, args.temperature))
                   for i in range(8)]
        for t in threads:
            t.daemon = True  # Thread will close when parent quits.
            t.start()
        subband_q.join()
        out1 = [None] * 8
        predicted1_256 = [None] * 8
        for _ in range(8):
            [subband_id, x] = subband_q.get()
            predicted1_256[subband_id] = x
            out1[subband_id] = P.inv_mulaw_quantize(np.array(x).astype(np.int16), net.quantization_channels)
            out1[subband_id] = denormalize(out1[subband_id], 'd%d' % (subband_id + 1))

        if mat_out_path:
            sio.savemat(mat_out_path, {
                                       'predicted1': out1,
                                       'target': target
                                       })
    end = time.time()
    print('Finished generating. Estimated time: %.3f sec.' % (end - start))

if __name__ == '__main__':
    main()


