# -*- coding: utf-8 -*-
"""
Description: This the second stage of the processing pipeline that is used to predict, based on an audio sample of
respiratory sounds, the prevalence of COVID-19 infection.

Stage 2: Processing and extracting features out of a suitable Audio Sample - normally the audio sample should have
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
from __future__ import print_function

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import json
import sys
import numpy as np
import librosa

import urllib
sys.path.append('../vggish')
import vggish_input
import vggish_params
import vggish_slim


SR = 22050  # sample rate
SR_VGG = 16000  # VGG pretrained model sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50%overlap, 5ms


# the dictionary which maps the labels used in the survey
# with their corresponding values
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



def download(url, dst_dir):
    """Download file.
    If the file not exist then download it.
    Args:url: Web location of the file.
    Returns: path to downloaded file.
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(dst_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        statinfo = os.stat(filepath)
        print('Successfully downloaded:', filename, statinfo.st_size, 'bytes.')
    return filepath

def sta_fun_2(npdata):  # 1D np array
    """Extract various statistical features from the numpy array provided as input.

    :param np_data: the numpy array to extract the features from
    :type np_data: numpy.ndarray
    :return: The extracted features as a vector
    :rtype: numpy.ndarray
    """

    # perform a sanity check
    if npdata is None:
        raise ValueError("Input array cannot be None")

    # perform the feature extraction
    Mean = np.mean(npdata, axis=0)
    Std = np.std(npdata, axis=0)
    
    # finally return the features in a concatenated array (as a vector)
    return np.concatenate((Mean, Std), axis=0).reshape(1, -1)

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
    
print("\nTesting your install of VGGish\n")
# Paths to downloaded VGGish files.
checkpoint_path = "../vggish/vggish_model.ckpt"

if not os.path.exists(checkpoint_path): #automatically download the checkpoint if not exist.
    url = 'https://storage.googleapis.com/audioset/vggish_model.ckpt'
    download(url, '../vggish/')
    

if __name__ == "__main__":
    path = sys.argv[1]  # data path
    meta_breath2cough = json.load(open(os.path.join(path,"android_breath2cough.json")))
   
   
    ##feature extraction  
    with tf.Graph().as_default(), tf.Session() as sess:
        # load pre-trained model
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME
        )
        
        x_data = []
        y_label = []
        y_uid = []

        # extract features for Anroid samples
        file_path = os.listdir(path)
        for files in sorted(file_path):
            print(files)
            if files not in [
                "covidandroidnocough",
                "covidandroidwithcough",
                "healthyandroidwithcough",
                "healthyandroidnosymp",
                "asthmaandroidwithcough"]:
                continue
            sample_path = os.path.join(path ,files,"breath")
            samples = os.listdir(sample_path)
            for sample in get_resort(samples):
                if ".wav_aug" in sample or "mono" in sample:
                    continue
                print(sample)
                # for breath
                sample_path = os.path.join(path ,files,"breath")
                file_b = os.path.join(sample_path,sample)
                name, filetpye = file_b.split(".")
                soundtype = sample.split("_")[0]
                if soundtype != "breaths":
                    print("no breath")
                    continue
                if filetpye != "wav":
                    print("no wav!")
                    continue

                y, sr = librosa.load(
                    file_b, sr=SR, mono=True, offset=0.0, duration=None
                )
                yt, index = librosa.effects.trim(
                    y, frame_length=FRAME_LEN, hop_length=HOP
                )
                duration = librosa.get_duration(y=yt, sr=sr)
                if duration < 2:
                    print("breath too short")
                    continue

                input_batch = vggish_input.waveform_to_examples(
                    yt, SR_VGG   
                )  # ?x96x64 --> ?x128 
                [features_breath] = sess.run(
                    [embedding_tensor], feed_dict={features_tensor: input_batch}
                )
                features_breath = sta_fun_2(features_breath)

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
                        print("android cough doesn't exit")
                        continue
                    else:
                        print("load")

                    #remove the beginning and ending silence
                    yt, index = librosa.effects.trim(
                        y, frame_length=FRAME_LEN, hop_length=HOP
                    )
                    duration = librosa.get_duration(y=yt, sr=sr)
                    # filter out samples shorter than 2 minutes after trimming
                    if duration < 2:
                        print("cough too short")
                        continue
                    input_batch = vggish_input.waveform_to_examples(
                        yt, SR_VGG
                    )  # ?x96x64 --> ?x128
                    [features_cough] = sess.run(
                        [embedding_tensor], feed_dict={features_tensor: input_batch}
                    )
                    features_cough = sta_fun_2(features_cough)
                    label = label_dict[files]

                    #combine breath features and cough features
                    uid = sample.split("_")[1]
                    features = np.concatenate((features_breath, features_cough), 
                                              axis=1 
                    )
                    x_data.append(features.tolist())
                    y_label.append(label)
                    y_uid.append(uid)
                    print("save features!")

        # extract features for web samples
        file_path = os.listdir(path)
        for fold in sorted(file_path):
            print(fold)
            if fold not in [
                "covidwebnocough",
                "covidwebwithcough",
                "healthywebwithcough",
                "healthywebnosymp",
                "asthmawebwithcough"]:
                continue
            fold_path = os.listdir(os.path.join(path,fold))
            for files in sorted(fold_path):
                print(files)
                # for breathe
                try:
                    sample_path = os.path.join(path ,fold ,files,"audio_file_breathe.wav")                      
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
                input_batch = vggish_input.waveform_to_examples(
                    yt, SR_VGG
                )  # ?x96x64 --> ?x128
                [features_breath] = sess.run(
                    [embedding_tensor], feed_dict={features_tensor: input_batch}
                )
                features_breath = sta_fun_2(features_breath)

                # for cough
                try:
                    sample_path = os.path.join(path ,fold ,files,"audio_file_cough.wav")
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
                input_batch = vggish_input.waveform_to_examples(
                    yt, SR_VGG
                )  # ?x96x64 --> ?x128
                [features_cough] = sess.run(
                    [embedding_tensor], feed_dict={features_tensor: input_batch}
                )
                features_cough = sta_fun_2(features_cough)
                    
                #combine breath and cough features for future use
                label = label_dict[fold]
                features = np.concatenate((features_breath, features_cough),
                                          axis=1
                )
                x_data.append(features.tolist())
                y_label.append(label)
                y_uid.append(files)
                print("save features")

        x_data = np.array(x_data)
        y_label = np.array(y_label)
        y_uid = np.array(y_uid)

        #save features in numpy.array
        np.save("x_data_vgg.npy", x_data)
        np.save("y_label_vgg.npy", y_label)
        np.save("y_uid_vgg.npy", y_uid)

        ## ===========================================================  
        ## agumentation features extraction
        x_data = []
        y_label = []
        y_uid = []

        # extract features for android samples
        file_path = os.listdir(path)
        for files in sorted(file_path):
            print(files)
            if files not in ["healthyandroidwithcough", 
                             "asthmaandroidwithcough"]:
                continue
            sample_path = os.path.join(path,files,"breath")
            samples = os.listdir(sample_path)
            for sample in get_resort(samples):
                if ".wav_aug" not in sample:
                    continue
                print(sample)
                # for breath
                sample_path = os.path.join(path,files,"breath")
                file_b = os.path.join(sample_path,sample)
                y, sr = librosa.load(
                    file_b, sr=SR, mono=True, offset=0.0, duration=None
                )
                yt, index = librosa.effects.trim(
                    y, frame_length=FRAME_LEN, hop_length=HOP
                )
                duration = librosa.get_duration(y=yt, sr=sr)
                if duration < 2:
                    print("breath too short")
                    continue

                input_batch = vggish_input.waveform_to_examples(
                    yt, SR_VGG
                )  # ?x96x64 --> ?x128
                [features_breath] = sess.run(
                    [embedding_tensor], feed_dict={features_tensor: input_batch}
                )
                features_breath = sta_fun_2(features_breath)

                # for cough
                raw_sample = sample.split(".", 1)[0] + ".wav"
                sample_c = meta_breath2cough[raw_sample]
                tail = sample.split(".", 1)[1]
                sample_path = os.path.join(path,files,"cough")
                file_c = os.path.join(sample_path,sample_c[:-3]+tail)
                try:
                    y, sr = librosa.load(
                        file_c, sr=SR, mono=True, offset=0.0, duration=None
                    )
                except IOError:
                    print("android cough doesn't exit")
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
                input_batch = vggish_input.waveform_to_examples(
                    yt, SR_VGG
                )  # ?x96x64 --> ?x128
                [features_cough] = sess.run(
                    [embedding_tensor], feed_dict={features_tensor: input_batch}
                )
                features_cough = sta_fun_2(features_cough)
                
                #combine breath and cough
                label = label_dict[files]
                uid = sample.split("_")[1]
                features = np.concatenate((features_breath, features_cough), 
                                          axis=1
                )
                x_data.append(features.tolist())
                y_label.append(label)
                y_uid.append(uid)
                print("save features!")

        # extract features for web samples
        file_path = os.listdir(path)
        for fold in sorted(file_path):
            print(fold)
            if fold not in ["healthywebwithcough", 
                             "asthmawebwithcough"]:
                continue
            fold_path = os.listdir(os.path.join(path,fold))
            for files in sorted(fold_path):
                print(files)
                for tail in [
                    "_aug_amp2.wav",
                    "_aug_noise1.wav",
                    "_aug_noise2.wav",
                    "_aug_pitchspeed1.wav",
                    "_aug_pitchspeed2.wav",
                ]:
                    # for breathe
                    try:
                        sample_path = os.path.join(path,fold,files,
                                    "audio_file_breathe.wav" + tail
                        )
                        file_b = sample_path
                        y, sr = librosa.load(
                            file_b, sr=SR, mono=True, offset=0.0, duration=None
                        )
                    except IOError:
                        print("breath doesn't exit")
                        continue
                    else:
                        print('load!')

                    yt, index = librosa.effects.trim(
                        y, frame_length=FRAME_LEN, hop_length=HOP
                    )
                    duration = librosa.get_duration(y=yt, sr=sr)
                    if duration < 2:
                        print("breath too short")
                        continue
                    input_batch = vggish_input.waveform_to_examples(
                        yt, SR_VGG
                    )  # ?x96x64 --> ?x128
                    [features_breath] = sess.run(
                        [embedding_tensor], feed_dict={features_tensor: input_batch}
                    )
                    features_breath = sta_fun_2(features_breath)

                    # for cough
                    try:
                        sample_path = os.path.join(path,fold,files,
                                    "audio_file_cough.wav" + tail
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
                        print("cough too short")
                        continue
                    input_batch = vggish_input.waveform_to_examples(
                        yt, SR_VGG
                    )  # ?x96x64 --> ?x128
                    [features_cough] = sess.run(
                        [embedding_tensor], feed_dict={features_tensor: input_batch}
                    )
                    features_cough = sta_fun_2(features_cough)

                    label = label_dict[fold]
                    features = np.concatenate((features_breath, features_cough), 
                                              axis=1
                    )
                    x_data.append(features.tolist())
                    y_label.append(label)
                    y_uid.append(files)
                    print("save features")

        x_data = np.array(x_data)
        y_label = np.array(y_label)
        y_uid = np.array(y_uid)

        #save features in numpy.array
        np.save("x_data_vgg_aug.npy", x_data)
        np.save("y_label_vgg_agu.npy", y_label)
        np.save("y_uid_vgg_agu.npy", y_uid)