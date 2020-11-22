# -*- coding: utf-8 -*-
"""
Description: This the first stage of the processing pipeline that is used to predict, based on an audio sample of
respiratory sounds, the prevalence of COVID-19 infection.

Stage 1: Processing and extracting features out of a suitable Audio Sample - normally the audio sample should have
the following characteristics:

 - breathing sounds: recordings of your breathing as deeply as you can from your mouth for five breaths.
 - coughing: a recording of three coughs.

The breaths and coughs should be evenly spaced in the audio clip for better results.

Note:
    This is supplied as supplementary material of the paper: https://arxiv.org/abs/2006.05919 by Brown et al. If you
    use this code (or derivatives of it) please cite our work.
    
    The input json file 'android_breath2cough.json' is used to match breath and cough from the same user as multiple 
    sample are stored in one path. 

Authors:
    Original code: T. Xia
    Check: J. Han

Date last touched: 19/11/2020
"""

import json
import os
import sys
import warnings
from math import pi

import librosa
import numpy as np
import pandas as pd
from scipy.fftpack import fft, hilbert

warnings.filterwarnings("ignore")

SR = 22050  # sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
MFCC_dim = 13  # the MFCC dimension

# the dictionary which maps the labels used in the survey
# with their corresponding index
label_dict = {
    "covidandroidnocough": 1,
    "covidandroidwithcough": 2,
    "covidwebnocough": 3,
    "covidwebwithcough": 4,
    "healthyandroidnosymp": -1,
    "healthyandroidwithcough": -2,
    "healthywebnosymp": -3,
    "healthywebwithcough": -4,
    "asthmaandroidwithcough": 6,
    "asthmawebwithcough": 8
}


def sta_fun(np_data):
    """Extract various statistical features from the numpy array provided as input.

    :param np_data: the numpy array to extract the features from
    :type np_data: numpy.ndarray
    :return: The extracted features as a vector
    :rtype: numpy.ndarray
    """

    # perform a sanity check
    if np_data is None:
        raise ValueError("Input array cannot be None")

    # perform the feature extraction
    dat_min = np.min(np_data)
    dat_max = np.max(np_data)
    dat_mean = np.mean(np_data)
    dat_rms = np.sqrt(np.sum(np.square(np_data)) / len(np_data))
    dat_median = np.median(np_data)
    dat_qrl1 = np.percentile(np_data, 25)
    dat_qrl3 = np.percentile(np_data, 75)
    dat_lower_q = np.quantile(np_data, 0.25, interpolation="lower")
    dat_higher_q = np.quantile(np_data, 0.75, interpolation="higher")
    dat_iqrl = dat_higher_q - dat_lower_q
    dat_std = np.std(np_data)
    s = pd.Series(np_data)
    dat_skew = s.skew()
    dat_kurt = s.kurt()

    # finally return the features in a concatenated array (as a vector)
    return np.array([dat_mean, dat_min, dat_max, dat_std, dat_rms,
                     dat_median, dat_qrl1, dat_qrl3, dat_iqrl, dat_skew, dat_kurt])

def get_period(signal, signal_sr):
    """Extract the period from the the provided signal

    :param signal: the signal to extract the period from
    :type signal: numpy.ndarray
    :param signal_sr: the sampling rate of the input signal
    :type signal_sr: integer
    :return: a vector containing the signal period
    :rtype: numpy.ndarray
    """

    # perform a sanity check
    if signal is None:
        raise ValueError("Input signal cannot be None")

    # transform the signal to the hilbert space
    hy = hilbert(signal)

    ey = np.sqrt(signal ** 2 + hy ** 2)
    min_time = 1.0 / signal_sr
    tot_time = len(ey) * min_time
    pow_ft = np.abs(fft(ey))
    peak_freq = pow_ft[3: int(len(pow_ft) / 2)]
    peak_freq_pos = peak_freq.argmax()
    peak_freq_val = 2 * pi * (peak_freq_pos + 2) / tot_time
    period = 2 * pi / peak_freq_val

    return np.array([period])

def extract_signal_features(signal, signal_sr):
    """Extract part of handcrafted features from the input signal.

    :param signal: the signal the extract features from
    :type signal: numpy.ndarray
    :param signal_sr: the sample rate of the signal
    :type signal_sr: integer
    :return: the populated feature vector
    :rtype: numpy.ndarray
    """

    # normalise the sound signal before processing
    signal = signal / np.max(np.abs(signal))
    # trim the signal to the appropriate length
    trimmed_signal, idc = librosa.effects.trim(signal, frame_length=FRAME_LEN, hop_length=HOP)
    # extract the signal duration
    signal_duration = librosa.get_duration(y=trimmed_signal, sr=signal_sr)
    # use librosa to track the beats
    tempo, beats = librosa.beat.beat_track(y=trimmed_signal, sr=signal_sr)
    # find the onset strength of the trimmed signal
    o_env = librosa.onset.onset_strength(trimmed_signal, sr=signal_sr)
    # find the frames of the onset
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=signal_sr)
    # keep only the first onset frame
    onsets = onset_frames.shape[0]
    # decompose the signal into its magnitude and the phase components such that signal = mag * phase
    mag, phase = librosa.magphase(librosa.stft(trimmed_signal, n_fft=FRAME_LEN, hop_length=HOP))
    # extract the rms from the magnitude component
    rms = librosa.feature.rms(y=trimmed_signal)[0]
    # extract the spectral centroid of the magnitude
    cent = librosa.feature.spectral_centroid(S=mag)[0]
    # extract the spectral rolloff point from the magnitude
    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=signal_sr)[0]
    # extract the zero crossing rate from the trimmed signal using the predefined frame and hop lengths
    zcr = librosa.feature.zero_crossing_rate(trimmed_signal, frame_length=FRAME_LEN, hop_length=HOP)[0]

    # pack the extracted features into the feature vector to be returned
    signal_features = np.concatenate(
        (
            np.array([signal_duration, tempo, onsets]),
            get_period(signal, signal_sr=sr),
            sta_fun(rms),
            sta_fun(cent),
            sta_fun(rolloff),
            sta_fun(zcr),
        ),
        axis=0,
    )

    # finally, return the gathered features and the trimmed signal
    return signal_features, trimmed_signal

def extract_mfcc(signal, signal_sr=SR, n_fft=FRAME_LEN, hop_length=HOP, n_mfcc=MFCC_dim):
    """Extracts the Mel-frequency cepstral coefficients (MFCC) from the provided signal

    :param signal: the signal to extract the mfcc from
    :type signal: numpy.ndarray
    :param signal_sr: the signal sample rate
    :type signal_sr: integer
    :param n_fft: the fft window size
    :type n_fft: integer
    :param hop_length: the hop length
    :type hop_length: integer
    :param n_mfcc: the dimension of the mfcc
    :type n_mfcc: integer
    :return: the populated feature vector
    :rtype: numpy.ndarray
    """
    # compute the mfcc of the input signal
    mfcc = librosa.feature.mfcc(
        y=signal, sr=signal_sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc, dct_type=3
    )

    # extract the first and second order deltas from the retrieved mfcc's
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # create the mfcc array
    mfccs = []

    # populate it using the extracted features
    for i in range(n_mfcc):
        mfccs.extend(sta_fun(mfcc[i, :])) 
    for i in range(n_mfcc):
        mfccs.extend(sta_fun(mfcc_delta[i, :]))
    for i in range(n_mfcc):
        mfccs.extend(sta_fun(mfcc_delta2[i, :]))

    # finally return the coefficients
    return mfccs

def extract_features(signal, signal_sr): 
    """Extract all features from the input signal. 

    :param signal: the signal the extract features from
    :type signal: numpy.ndarray
    :param signal_sr: the sample rate of the signal
    :type signal_sr: integer
    :return: the extracted feature vector
    :rtype: numpy.ndarray
    """

    # extract the signal features
    signal_features, trimmed_signal = extract_signal_features(signal, signal_sr)

    # extract the mfcc's from the trimmed signal and get the statistical feature. 
    mfccs = extract_mfcc(trimmed_signal)

    return np.concatenate((signal_features, mfccs), axis=0)
    
def get_resort(files):
    """Re-sort the files under data path.

    :param files: file list
    :type files: list
    :return: alphabetic orders
    :rtype: list
    """
    name_dict = {}
    for sample in files:
        type,name,others = sample.split('_',2)  # the UID is a mixed of upper and lower characters
        name = name.lower()
        name_dict['_'.join([type,name,others])] = sample
    re_file = [name_dict[s] for s in sorted(name_dict.keys())]
    return re_file
    
    
if __name__ == "__main__":
    """The main stub and entry point for the pipeline"""

    path = sys.argv[1]  # data path
    meta_breath2cough = json.load(open(os.path.join(path,"android_breath2cough.json")))
    
    #feature extraction 
    x_data = []
    y_label = []
    y_uid = []

    #first extract features for samples from [Android] 
    file_path = os.listdir(path) 
    for files in sorted(file_path):
        print(files)
        if files not in [
            "covidandroidnocough",
            "covidandroidwithcough",
            "healthyandroidwithcough",
            "healthyandroidnosymp",
            "asthmaandroidwithcough",]:
            continue
        sample_path = os.path.join(path,files,"breath")
        samples = os.listdir(sample_path)
        for sample in get_resort(samples):
            if ".wav_aug" in sample or "mono" in sample:
                continue
            print(sample)

            # for breath
            sample_path = os.path.join(path,files,"breath")
            file_b = os.path.join(sample_path,sample)

            name, filetpye = file_b.split(".")
            soundtype = sample.split("_")[0]
            if soundtype != "breaths":
                continue
            if filetpye != "wav":
                continue

            #remove the beginning and ending silence
            y, sr = librosa.load(file_b, sr=SR, mono=True, offset=0.0, duration=None)
            yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=yt, sr=sr)
            #filter out the audio shorter then 2 seconds. 
            if duration < 2:
                print("breath too short")
                continue
            features_breath = extract_features(signal=y, signal_sr=sr)

            # for cough
            if sample in meta_breath2cough:
                sample_c = meta_breath2cough[sample]
                sample_path = os.path.join(path,files,"cough")
                file_c = os.path.join(sample_path,sample_c)
                try:
                    y, sr = librosa.load(
                        file_c, sr=SR, mono=True, offset=0.0, duration=None
                    )
                except IOError:
                    print("adroid cough doesn't exit")
                    continue
                else:
                    print("load")

                yt, index = librosa.effects.trim(
                    y, frame_length=FRAME_LEN, hop_length=HOP
                )
                duration = librosa.get_duration(y=yt, sr=sr)
                if duration < 2:
                    print("cough too short")
                    continue
                features_cough = extract_features(signal=y, signal_sr=sr)
                
                #combine breath and cough features for future use
                label = label_dict[files]
                uid = sample.split("_")[1]
                features = np.concatenate((features_breath, features_cough),
                                          axis=0
                )
                x_data.append(features.tolist())
                y_label.append(label)
                y_uid.append(uid)
                print("save features!")

    #then extract features for samples from [Web]
    file_path = os.listdir(path)
    for fold in sorted(file_path): 
        print(fold)
        if fold not in [
            "covidwebnocough",
            "covidwebwithcough",
            "healthywebwithcough",
            "healthywebnosymp",
            "asthmawebwithcough",
            ]:
            continue
        fold_path = os.listdir(os.path.join(path,fold))
        for files in sorted(fold_path):
            print(files)
            # for breathe
            try:
                sample_path = (os.path.join(path,fold,files,"audio_file_breathe.wav"))
                file_b = sample_path
                y, sr = librosa.load(
                    file_b, sr=SR, mono=True, offset=0.0, duration=None
                )
            except IOError:
                print("breath doesn't exit")
                continue
            else:
                print("load")

            yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=yt, sr=sr)
            if duration < 2:
                print("breath too short")
                continue  
            features_breath = extract_features(signal=y, signal_sr=sr)

            # for cough
            try:
                sample_path = (os.path.join(path,fold,files,"audio_file_cough.wav"))
                file_c = sample_path
                y, sr = librosa.load(
                    file_c, sr=SR, mono=True, offset=0.0, duration=None
                )
            except IOError:
                print("cough doesn't exit")
                continue
            else:
                print("load")

            yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=yt, sr=sr)
            if duration < 2:
                print("cough too short")
                continue 
            features_cough = extract_features(signal=y, signal_sr=sr)
            
            #combine breath and cough
            label = label_dict[fold]
            features = np.concatenate((features_breath, features_cough), 
                                      axis=0
            )
            x_data.append(features.tolist())
            y_label.append(label)
            y_uid.append(files)
            print("save features")

    x_data = np.array(x_data)
    y_label = np.array(y_label)
    y_uid = np.array(y_uid)

    #save features in numpy.array
    np.save("x_data_handcraft.npy", x_data)
    np.save("y_label_handcraft.npy", y_label)
    np.save("y_uid_handcraft.npy", y_uid)

    ##==================================================================================
    #Augmentation features extraction
    x_data = []
    y_label = []
    y_uid = []

    #firstly for android
    file_path = os.listdir(path)
    for files in sorted(file_path):
        print(files)
        if files not in ["healthyandroidwithcough", 
                         "asthmaandroidwithcough"]:
            continue
        sample_path = os.path.join(path,files,"breath")
        samples = os.listdir(sample_path)
        for sample in get_resort(samples):
            print(sample)
            # for breath
            sample_path = os.path.join(path,files,"breath")
            file_b = os.path.join(sample_path,sample)
            if ".wav_aug" not in sample:
                continue

            y, sr = librosa.load(file_b, sr=SR, mono=True, offset=0.0, duration=None)
            yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=yt, sr=sr)
            if duration < 2:
                print("breath too short")
                continue
            features_breath = extract_features(signal=y, signal_sr=sr)

            # for cough
            raw_sample = sample.split(".", 1)[0] + ".wav"
            tail = sample.split(".", 1)[1]
            sample_c = meta_breath2cough[raw_sample]
            sample_path = os.path.join(path,files,"cough")
            file_c = os.path.join(sample_path,sample_c[:-3] + tail)
            print(file_c)
 
            try:
                y, sr = librosa.load(
                    file_c, sr=SR, mono=True, offset=0.0, duration=None
                )
            except IOError:
                print("adroid cough doesn't exit")
                continue
            else:
                y, sr = librosa.load(
                    file_c, sr=SR, mono=True, offset=0.0, duration=None
                )

            yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=yt, sr=sr)
            if duration < 2:
                print("cough too short")
                continue
            features_cough = extract_features(signal=y, signal_sr=sr)
            label = label_dict[files]
            uid = sample.split("_")[1]

            features = np.concatenate((features_breath, features_cough), 
                                      axis=0
            )
            x_data.append(features.tolist())
            y_label.append(label)
            y_uid.append(uid)
            print("save features!")

    # Then extract features for web samples
    file_path = os.listdir(path)
    for fold in sorted(file_path): 
        print(fold)
        if fold not in ["healthywebwithcough", 
                         "asthmawebwithcough"]:
            continue
        fold_path = os.listdir(os.path.join(path ,fold))
        for files in sorted(fold_path):
            print(files)
            for tail in [
                "_aug_amp2.wav",
                "_aug_noise1.wav",
                "_aug_noise2.wav",
                "_aug_pitchspeed1.wav",
                "_aug_pitchspeed2.wav",
            ]:    # why not like the android one???
                try:
                    sample_path = (os.path.join(path,fold,files,
                                   "audio_file_breathe.wav"+tail)
                    )
                    file_b = sample_path
                    y, sr = librosa.load(
                        file_b, sr=SR, mono=True, offset=0.0, duration=None
                    )
                except IOError:
                    print("breath doesn't exit")
                    continue
                else:
                    print("load")

                yt, index = librosa.effects.trim(
                    y, frame_length=FRAME_LEN, hop_length=HOP
                )
                duration = librosa.get_duration(y=yt, sr=sr)
                if duration < 2:
                    continue
                features_breath = extract_features(signal=y, signal_sr=sr)

                # for cough
                try:
                    sample_path = (os.path.join(path,fold,files,
                                   "audio_file_cough.wav"+tail)
                    )
                    file_c = sample_path
                    y, sr = librosa.load(
                        file_c, sr=SR, mono=True, offset=0.0, duration=None
                    )
                except IOError:
                    print("cough doesn't exit")
                    continue
                else:
                    print("load")

                yt, index = librosa.effects.trim(
                    y, frame_length=FRAME_LEN, hop_length=HOP
                )
                duration = librosa.get_duration(y=yt, sr=sr)
                if duration < 2:
                    continue
                features_cough = extract_features(signal=y, signal_sr=sr)
                label = label_dict[fold]
                features = np.concatenate((features_breath, features_cough), 
                                          axis=0
                )
                x_data.append(features.tolist())
                y_label.append(label)
                y_uid.append(files)
                print("save features")

    
    x_data = np.array(x_data)
    y_label = np.array(y_label)
    y_uid = np.array(y_uid)

    #save features for augmentated samples in numpy.array
    np.save("x_data_handcraft_aug.npy", x_data)
    np.save("y_label_handcraft_aug.npy", y_label)
    np.save("y_uid_handcraft_aug.npy", y_uid)