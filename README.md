# covid19-sounds-kdd20
This repo is for our KDD paper *Exploring Automatic Diagnosis of COVID-19 from Crowdsourced Respiratory Sound Data. arXiv preprint arXiv:2006.05919.*

## Dataset

We do release our dataset used in this paper for research purpose. To obtain this, please visit our website  https://www.covid-19-sounds.org/en/blog/data_sharing.html for more details.

Basically, the dataset consists of ten groups of samples: 

- covidandroidnocough,
- covidandroidwithcough,
- covidwebnocough,
- covidwebwithcough,
- healthyandroidnosymp,
- healthyandroidwithcough,
- healthywebnosymp,
- healthywebwithcough,
- asthmaandroidwithcough,
- asthmawebwithcough.

Since we have two applications to collect the data, we use 'web' and 'android' to distinguish them.
Also, in the name format, 'nocough' and 'withcough' indicate whether the user reported a dry or wet cough symptom, while 'nosymp' means that the user had no symptoms at that time.
In each Android file, cough and breath samples are presented separately, and thus a `.json` file named as `android_breath2cough.json` is provided to match cough with breath of the same user. For example, `"breaths_CNz7PwFNQz_1588140467902.wav": "cough_CNz7PwFNQz_1588140467941.wav"` indicates those two audios were contributed by "the same" user  `CNz7PwFNQz`.

Unzip all the folds in this dataset before using it to this model!

## Codes

Our model is implemented by Python3 with Tensorflow. To reproduce the results, codes are provided.

### Dependencies (others may also work)

- tensorflow==1.15
- librosa==0.6.3
- numpy==1.18.5
- pandas==0.25.3
- scipy==1.4.1
- seaborn==0.91
- scikit-learn==0.22.2
- vggish (from https://modelzoo.co/model/audioset) .
- urllib

### Training

To train and evaluate our model, please `sh run_experiments.sh $path`. $path is the dataset path (use the full path). 

Specifically, this bash executes all the python scripts sequentially as follows, 

`python 01_extract_handcraft_features.py $path`

This code is to extract 477-dimensional handcraft features for each breathing or cough audio file. The returned tensor is a combination of features for corresponding breathing and cough pairs (dimension = 477 * 2 = 954).

`python 02_extract_vgg_features.py $path`

It is to obtain vggish features (dimension = 128 * 2 = 256): from a pretrained VGG network (Codes are in vggish folder). 

`python 03_classification_without_augmentation.py`

In this script, we research a variety of feature fusion strategies and hype-parameters to get the best performance for three tasks, respectively. Results are saved as readable CSV files.

`python 04_classification_with_augmentation.py`

This code is only for task2 and task3, where we add some augmented sounds to improve the performance of classification.

After parameter tuning, we represent the best model in Table1. The implemetation can be found in `show_results.ipynb`.

Besides, feature analysis process as shown in Figure1 is given in `feature_visulization.ipynb`.

### Issues

This code project is developed by Tong Xia with Github account *XTxiatong*. Any problem, please contact me by tx229@cam.ac.uk.
