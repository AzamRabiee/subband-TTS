# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from nnmnkwii.datasets import cmu_arctic
from nnmnkwii.io import hts
from nnmnkwii import preprocessing as P
from os.path import exists
import librosa

from hparams import hparams


def build_from_path(in_dir, out_dir, silence_threshold, fft_size, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    speakers = cmu_arctic.available_speakers

    wd = cmu_arctic.WavFileDataSource(in_dir, speakers=speakers)
    wav_paths = wd.collect_files()
    speaker_ids = wd.labels

    for index, (speaker_id, wav_path) in enumerate(
            zip(speaker_ids, wav_paths)):
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index + 1, speaker_id, wav_path, "N/A", silence_threshold, fft_size)))
    return [future.result() for future in tqdm(futures)]


def start_at(labels):
    has_silence = labels[0][-1] == "pau"
    if not has_silence:
        return labels[0][0]
    for i in range(1, len(labels)):
        if labels[i][-1] != "pau":
            return labels[i][0]
    assert False


def end_at(labels):
    has_silence = labels[-1][-1] == "pau"
    if not has_silence:
        return labels[-1][1]
    for i in range(len(labels) - 2, 0, -1):
        if labels[i][-1] != "pau":
            return labels[i][1]
    assert False


def _process_utterance(out_dir, index, speaker_id, wav_path, text, silence_threshold, fft_size, ):
    sr = hparams.sample_rate

    # Load the audio to a numpy array. Resampled if needed
    wav = audio.load_wav(wav_path)

    lab_path = wav_path.replace("wav/", "lab/").replace(".wav", ".lab")

    # Trim silence from hts labels if available
    # TODO
    if exists(lab_path) and False:
        labels = hts.load(lab_path)
        b = int(start_at(labels) * 1e-7 * sr)
        e = int(end_at(labels) * 1e-7 * sr)
        wav = wav[b:e]
        wav, _ = librosa.effects.trim(wav, top_db=20)
    else:
        wav, _ = librosa.effects.trim(wav, top_db=20)

    # Mu-law quantize
    quantized = P.mulaw_quantize(wav)

    # Trim silences
    start, end = audio.start_and_end_indices(quantized, silence_threshold)
    quantized = quantized[start:end]
    wav = wav[start:end]

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    # lws pads zeros internally before performing stft
    # this is needed to adjast time resolution between audio and mel-spectrogram
    l, r = audio.lws_pad_lr(wav, fft_size, audio.get_hop_size())

    # zero pad for quantized signal
    quantized = np.pad(quantized, (l, r), mode="constant",
                       constant_values=P.mulaw_quantize(0))
    N = mel_spectrogram.shape[0]
    assert len(quantized) >= N * audio.get_hop_size()

    # time resolution adjastment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    quantized = quantized[:N * audio.get_hop_size()]
    assert len(quantized) % audio.get_hop_size() == 0

    timesteps = len(quantized)

    wav_id = wav_path.split('/')[-1].split('.')[0]
    # Write the spectrograms to disk:
    audio_filename = '{}-audio.npy'.format(wav_id)
    mel_filename = '{}-mel.npy'.format(wav_id)
    np.save(os.path.join(out_dir, audio_filename),
            quantized.astype(np.int16), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return (audio_filename, mel_filename, timesteps, text, speaker_id)

