# -*- coding: utf-8 -*-
"""
Description: This the fourth stage of the processing pipeline that is used to predict, based on an audio sample of
respiratory sounds, the prevalence of COVID-19 infection.

Stage 4: Combining features and training the binary classifier for prediction-normally two categories of features 
should be involved and agumented samples are utlized for task2 and task3.

 - handcrafted features: features like MFCCs from 01_extract_handcrafted_feautures.py.
 - VGG features: embeddings from the pretrained VGGish network, i.e., the output of 02_extract_vgg_features.py


Note:
    This is supplied as supplementary material of the paper: https://arxiv.org/abs/2006.05919 by Brown et al. If you
    use this code (or derivatives of it) please cite our work.
    
    You may find the results slight differ from what we reported in KDD paper in Table 2. That is because we forgot to 
    fix the order of adding augmentation data (see line 357). You can try different seeds, which can achieve similar 
    perforamnce.
 
Authors:
    Original code: T. Xia
    Check: J. Han

Date last touched: 19/11/2020
"""
import random
import warnings

import numpy as np
from sklearn import decomposition
from sklearn import metrics
from sklearn import preprocessing
from sklearn.externals import joblib #import joblib #if externals does not exits
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def RandomUnderSampler(np_data, np_label): 
    """downsample the majority class according to the given labels.

    :param np_data: extracted features as a array
    :type np_data: numpy.ndarray
    :param np_label: correspoinds labes as a vector
    :type np_data: numpy.ndarray
    :return: feature vectors and labels for balanced samples
    :rtype: numpy.ndarray
    """
    label = list(set(np_label))
    
    # perform a sanity check
    if len(label) < 2:
        raise ValueError("Less than two classed input")
        
    # seperate two class
    number_c0 = np.sum(np_label == label[0])
    number_c1 = np.sum(np_label == label[1])
    x_c0 = np_data[np_label == label[0], :]
    x_c1 = np_data[np_label == label[1], :]
    y_c0 = np_label[np_label == label[0]]
    y_c1 = np_label[np_label == label[1]]
    
    # downsample the majority class
    random.seed(0)
    if number_c0 < number_c1:
        index = random.sample(range(0, number_c1), number_c0)
        x_c1 = x_c1[index, :]
        y_c1 = y_c1[index]

    else:
        index = random.sample(range(0, number_c0), number_c1)
        x_c0 = x_c0[index, :]
        y_c0 = y_c0[index]
        
    new_data = np.concatenate((x_c0, x_c1), axis=0)
    new_label = np.concatenate((y_c0, y_c1), axis=0)
    
    #return the balanced class
    return new_data, new_label

if __name__ == "__main__":
    
    # try different feature combination 
    for features in [
        0,  #Vgg features only, denoted by TYPE=2 in our paer
        4,  #Vgg features + duration, tempo, onset, and period, denoted by TYPE=3(A) in our paer
        48, #Vgg features + siginal features  
        48 + 13 * 11, #Vgg features+ all features except Δ-MFCCs and Δ2-MFCCs, denoted by TYPE=3(B) in our paer
        48 + 13 * 11 * 3, #Vgg features+ all handcrafted features, denoted by TYPE=3(C) in our paer
    ]:  
        
        print("========feature group=======")
        Number = features + 256  # the feature dimension for one modal
        print("features dimension:", Number)
        
        # load features and labels got from step 1 and 2
        x_data_all_1 = np.load(
                "x_data_handcraft.npy", allow_pickle=True
        )
        x_data_all_2 = np.load(
                "x_data_vgg.npy", allow_pickle=True
        )
        x_data_all_2 = np.squeeze(x_data_all_2)
        
        x_data_all = np.concatenate(
            (
                x_data_all_1[:, :features],           #handcrafted features for breath
                x_data_all_2[:, :256],                #Vgg features for breath
                x_data_all_1[:, 477 : 477 + features],#Vgg features for cough
                x_data_all_2[:, 256:],                #Vgg for cough
            ),
            axis=1,
        )   
        y_label_all = np.load(
            "y_label_handcraft.npy", allow_pickle=True
        )  # labels are the same, eigher handcraft or vgg is correct
        y_uid_all = np.load("y_uid_handcraft.npy", allow_pickle=True)

    
        # split the data for different tasks
        x_data_all_1 = x_data_all[y_label_all == 1]  #covidandroidnocough
        x_data_all_2 = x_data_all[y_label_all == 2]  #covidandroidwithcough
        x_data_all_3 = x_data_all[y_label_all == 3]  #covidwebnocough
        x_data_all_4 = x_data_all[y_label_all == 4]  #covidwebwithcough
        x_data_all_6 = x_data_all[y_label_all == 6]  #asthmaandroidwithcough
        x_data_all_8 = x_data_all[y_label_all == 8]  #asthmawebwithcough
        x_data_all_m1 = x_data_all[y_label_all == -1] #healthyandroidnosymp
        x_data_all_m2 = x_data_all[y_label_all == -2] #healthyandroidwithcough
        x_data_all_m3 = x_data_all[y_label_all == -3] #healthywebnosymp
        x_data_all_m4 = x_data_all[y_label_all == -4] #healthywebwithcough
    
        y_label_all_1 = y_label_all[y_label_all == 1]
        y_label_all_2 = y_label_all[y_label_all == 2]
        y_label_all_3 = y_label_all[y_label_all == 3]
        y_label_all_4 = y_label_all[y_label_all == 4]
        y_label_all_6 = y_label_all[y_label_all == 6]
        y_label_all_8 = y_label_all[y_label_all == 8]
        y_label_all_m1 = y_label_all[y_label_all == -1]
        y_label_all_m2 = y_label_all[y_label_all == -2]
        y_label_all_m3 = y_label_all[y_label_all == -3]
        y_label_all_m4 = y_label_all[y_label_all == -4]
    
        y_uid_1 = y_uid_all[y_label_all == 1]
        y_uid_2 = y_uid_all[y_label_all == 2]
        y_uid_3 = y_uid_all[y_label_all == 3]
        y_uid_4 = y_uid_all[y_label_all == 4]
        y_uid_6 = y_uid_all[y_label_all == 6]
        y_uid_8 = y_uid_all[y_label_all == 8]
        y_uid_m1 = y_uid_all[y_label_all == -1]
        y_uid_m2 = y_uid_all[y_label_all == -2]
        y_uid_m3 = y_uid_all[y_label_all == -3]
        y_uid_m4 = y_uid_all[y_label_all == -4]
    
        #use agumentation for task2 and task3
        x_data_aug_1 = np.load(
                "x_data_handcraft_aug.npy", allow_pickle=True
        )
        x_data_aug_2 = np.load(
                "x_data_vgg_aug.npy", allow_pickle=True
        )
        x_data_aug_1 = np.squeeze(x_data_aug_1)
        x_data_aug_2 = np.squeeze(x_data_aug_2)
        x_data_all = np.concatenate(
            (
                x_data_aug_1[:, :features],                #handcrafted features for breath
                x_data_aug_2[:, :256],                     #Vgg features for breath
                x_data_aug_1[:, 477 : 477 + features],     #Vgg features for cough
                x_data_aug_2[:, 256:],                     #Vgg for cough
            ),
            axis=1,
        )
    
        y_label_all = np.load(
                "y_label_handcraft_aug.npy", allow_pickle=True
        )
        y_uid_all = np.load(
                "y_uid_handcraft_aug.npy", allow_pickle=True
        )
    
        ax_data_all_m2 = x_data_all[y_label_all == -2]
        ay_label_all_m2 = y_label_all[y_label_all == -2]
        ay_uid_m2 = y_uid_all[y_label_all == -2]
    
        ax_data_all_m4 = x_data_all[y_label_all == -4]
        ay_label_all_m4 = y_label_all[y_label_all == -4]
        ay_uid_m4 = y_uid_all[y_label_all == -4]
    
        ax_data_all_6 = x_data_all[y_label_all == 6]
        ay_label_all_6 = y_label_all[y_label_all == 6]
        ay_uid_6 = y_uid_all[y_label_all == 6]
    
        ax_data_all_8 = x_data_all[y_label_all == 8]
        ay_label_all_8 = y_label_all[y_label_all == 8]
        ay_uid_8 = y_uid_all[y_label_all == 8]
    
        # save csv for each group of experiments
        output = open("subset" + str(features) + "_tasks_svm_auc_aug.csv", "wb")
        head = [
            "Tasks",
            "Train",
            "Test",
            "Breathing_PCA",
            "Breathing_AUC",
            "Breathing_ACC",
            "Breathing_Pre",
            "Breathing_Rec",
            "Cough_PCA",
            "Cough_AUC",
            "Cough_ACC",
            "Cough_Pre",
            "Cough_Rec",
            "BreathingCough_PCA",
            "BreathingCough_AUC",
            "BreathingCough_ACC",
            "BreathingCough_Pre",
            "BreathingCough_Rec",
        ]
    
        output.write(",".join(head).encode(encoding="utf-8"))
        output.write("\n".encode(encoding="utf-8"))
    
        # the variance remained after PCA
        for n in [0.7, 0.8, 0.9, 0.95]:  
            print("PCA:", n)
            output.write("\n".encode(encoding="utf-8"))
            for i1 in ["task2", "task3"]:
                print("Conduct", i1)
                line = ['balance' + str(n) + i1]
                
                if i1 == "task1":
                    x_data_all_task = np.concatenate(
                        (
                            x_data_all_1,
                            x_data_all_2,
                            x_data_all_3,
                            x_data_all_4,
                            x_data_all_m1,
                            x_data_all_m3,
                        ),
                        axis=0,
                    )
                    y_label_all_task = np.concatenate(
                        (
                            y_label_all_1,
                            y_label_all_2,
                            y_label_all_3,
                            y_label_all_4,
                            y_label_all_m1,
                            y_label_all_m3,
                        ),
                        axis=0,
                    )
                    y_uid_all_task = np.concatenate(
                        (y_uid_1, y_uid_2, y_uid_3, y_uid_4, y_uid_m1, y_uid_m3), 
                        axis=0
                    )

                    y_label_all_task[y_label_all_task > 0] = 1  # covid positive
                    y_label_all_task[y_label_all_task < 0] = 0

                if i1 == "task2":
                    x_data_all_task = np.concatenate(
                        (x_data_all_2, x_data_all_4, x_data_all_m2, x_data_all_m4),
                        axis=0,
                    )
                    y_label_all_task = np.concatenate(
                        (y_label_all_2, y_label_all_4, y_label_all_m2, y_label_all_m4),
                        axis=0,
                    )
                    y_uid_all_task = np.concatenate(
                        (y_uid_2, y_uid_4, y_uid_m2, y_uid_m4), axis=0
                    )

                    y_label_all_task[y_label_all_task > 0] = 1  # covid positive
                    y_label_all_task[y_label_all_task < 0] = 0

                    ax_data_all_task = np.concatenate(
                        (ax_data_all_m2, ax_data_all_m4), axis=0
                    )
                    ay_label_all_task = np.concatenate(
                        (ay_label_all_m2, ay_label_all_m4), axis=0
                    )
                    a_uid_all_task = np.concatenate((ay_uid_m2, ay_uid_m4), 
                                                    axis=0
                    )

                    ay_label_all_task[ay_label_all_task > 0] = 1
                    ay_label_all_task[ay_label_all_task < 0] = 0

                if i1 == "task3":
                    x_data_all_task = np.concatenate(
                        (x_data_all_2, x_data_all_4, x_data_all_6, x_data_all_8), 
                        axis=0
                    )
                    y_label_all_task = np.concatenate(
                        (y_label_all_2, y_label_all_4, y_label_all_6, y_label_all_8),
                        axis=0,
                    )
                    y_uid_all_task = np.concatenate(
                        (y_uid_2, y_uid_4, y_uid_6, y_uid_8), axis=0
                    )

                    y_label_all_task[y_label_all_task < 5] = 1  # covid positive
                    y_label_all_task[y_label_all_task > 4] = 0

                    ax_data_all_task = np.concatenate(
                        (ax_data_all_6, ax_data_all_8), axis=0
                    )
                    ay_label_all_task = np.concatenate(
                        (ay_label_all_6, ay_label_all_8), axis=0
                    )
                    a_uid_all_task = np.concatenate((ay_uid_6, ay_uid_8), axis=0)
                    ay_label_all_task[ay_label_all_task < 5] = 1
                    ay_label_all_task[ay_label_all_task > 4] = 0

                for i2 in ["breath", "cough", "breath_cough"]:  # multi-modal
                    if i2 == "breath":
                        x_data_all_this = x_data_all_task[:, :Number]
                        ax_data_all_this = ax_data_all_task[:, :Number]
                    if i2 == "cough":
                        x_data_all_this = x_data_all_task[:, Number : Number * 2]
                        ax_data_all_this = ax_data_all_task[:, Number : Number * 2]
                    if i2 == "breath_cough":
                        x_data_all_this = x_data_all_task[:, : Number * 2]
                        ax_data_all_this = ax_data_all_task[:, : Number * 2]

                    dpca = []
                    acc = []
                    pre = []
                    rec = []
                    auc = []
                    prauc = []
                    train_ratio = []
                    test_ratio = []

                    for seed in [1, 2, 5, 10, 100, 200, 500, 1000, 2000, 5000]:

                        gss = GroupShuffleSplit(
                            n_splits=1, test_size=0.2, random_state=seed
                        )
                        idx1, idx2 = next(
                            gss.split(x_data_all_this, groups=y_uid_all_task)
                        )

                        # Get the split DataFrames.
                        train_x, test_x = x_data_all_this[idx1], x_data_all_this[idx2]
                        y_train, y_test = y_label_all_task[idx1], y_label_all_task[idx2]
                        uid_train, uid_test = y_uid_all_task[idx1], y_uid_all_task[idx2]

                        # merge training samples, adding augmented sample to the same user
                        print("before agumentation:", len(train_x))
                        if i1 != "task1":
                            train_users = sorted(set(uid_train))
                            random.seed(46) 
                            random.shuffle(train_users)
                            for u in train_users:
                                train_x = np.concatenate(
                                    (
                                        train_x,
                                        ax_data_all_this[
                                            a_uid_all_task == u
                                        ],
                                    ),
                                    axis=0,
                                )
                                y_train = np.concatenate(
                                    (
                                        y_train,
                                        ay_label_all_task[
                                            a_uid_all_task == u     
                                        ],
                                    ),
                                    axis=0,
                                )
                        print("after agumentation:", len(train_x))

                        train_x, y_train = RandomUnderSampler(train_x, y_train)
                        test_x, y_test = RandomUnderSampler(test_x, y_test)

                        train_ratio.append(train_x.shape[0])
                        test_ratio.append(test_x.shape[0])

                        #scale data
                        scaler = preprocessing.StandardScaler().fit(train_x)
                        x_train_n = scaler.transform(train_x)
                        x_test_n = scaler.transform(test_x)

                        # reduce feature dimension
                        pca = decomposition.PCA(n)
                        pca.fit(x_train_n)
                        x_train_n_pca = pca.fit_transform(x_train_n)
                        dpca.append(x_train_n_pca.shape[1])
                        x_train_n_pca = pca.fit_transform(x_train_n)
                        x_test_n_pca = pca.transform(x_test_n)

                        # for SVM
                        param_grid = [
                            {
                                "C": [10, 100, 1000],
                                "gamma": [0.1, 0.01, 0.001, 0.0001],
                                "kernel": ["rbf"],
                                "class_weight": ["balanced"],
                            }
                        ]

                        clf = SVC(probability=True)
                        gs = GridSearchCV(
                            clf,
                            param_grid,
                            scoring=metrics.make_scorer(metrics.roc_auc_score),
                            n_jobs=-1,
                            cv=5,
                        )

                        gs = gs.fit(x_train_n_pca, y_train)
                        #store and reload model
                        joblib.dump(gs.best_estimator_, "best_model_agu.pkl") 

                        clf = joblib.load("best_model_agu.pkl")
                        predicted = clf.predict(x_test_n_pca)
                        probs = clf.predict_proba(x_test_n_pca)
                        pre.append(metrics.precision_score(y_test, predicted))
                        acc.append(metrics.accuracy_score(y_test, predicted))
                        auc.append(metrics.roc_auc_score(y_test, probs[:, 1]))
                        precision, recall, _ = metrics.precision_recall_curve(
                            y_test, probs[:, 1]
                        )
                        prauc.append(metrics.auc(recall, precision))
                        rec.append(metrics.recall_score(y_test, predicted))

                    if i2 == "breath":
                        line.append(
                            "{:.2f}".format(np.mean(train_ratio)) + "("
                            "{:.2f}".format(np.std(train_ratio)) + ")"
                        )
                        line.append(
                            "{:.2f}".format(np.mean(test_ratio)) + "("
                            "{:.2f}".format(np.std(test_ratio)) + ")"
                        )

                    line.append(
                        "{:.2f}".format(np.mean(dpca)) + "("
                        "{:.2f}".format(np.std(dpca)) + ")"
                    )
                    line.append(
                        "{:.4f}".format(np.mean(auc)) + "("
                        "{:.4f}".format(np.std(auc)) + ")"
                    )
                    line.append(
                        "{:.4f}".format(np.mean(acc)) + "("
                        "{:.4f}".format(np.std(acc)) + ")"
                    )
                    line.append(
                        "{:.4f}".format(np.mean(pre)) + "("
                        "{:.4f}".format(np.std(pre)) + ")"
                    )
                    line.append(
                        "{:.4f}".format(np.mean(rec)) + "("
                        "{:.4f}".format(np.std(rec)) + ")"
                    )

                output.write(",".join(line).encode(encoding="utf-8"))
                output.write("\n".encode(encoding="utf-8"))
                print("---------------")
