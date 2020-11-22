#!/bin/bash
python 01_extract_handcrafted_features.py "${1}"
python 02_extract_vgg_features.py "${1}"
python 03_classification_without_augmentation.py
python 04_classification_with_augmentation.py
