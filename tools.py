# -*- coding: utf-8 -*-
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

CATEGORIES = ['alaska', 'bichons', 'chihuahua', 'golden', 'husky', 'labrador', 'papillon', 'samoyed', 'shepherd', 'teddy']
SAMPLE_PER_CATEGORY = 500
INPUT_SIZE = 224

# read the image
def read_img(data_dir, filepath, size):
    img = image.load_img(os.path.join(data_dir, filepath), target_size=size)
    img = image.img_to_array(img)
    return img


# read the training data
def get_train_val_data(data_dir, train_path):
    SEED = 1991
    train_dir = os.path.join(data_dir, train_path)

    train = []
    for category_id, category in enumerate(CATEGORIES):
        for file in os.listdir(os.path.join(train_dir, category)):
            train.append([train_path+'/{}/{}'.format(category, file), category_id, category])
    train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])
    # sampling
    train = pd.concat([train[train['category'] == c][:SAMPLE_PER_CATEGORY] for c in CATEGORIES])
    train = train.sample(frac=1)
    train.index = np.arange(len(train))

    # read in the image array
    x_train = np.zeros((len(train), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
    for i, file in tqdm(enumerate(train['file'])):
        img = read_img(data_dir, file, (INPUT_SIZE, INPUT_SIZE))
        x = preprocess_input(np.expand_dims(img.copy(), axis=0))
        x_train[i] = x

    # split train and validation
    np.random.seed(seed=SEED)
    rnd = np.random.random(len(train))
    train_idx = rnd < 0.8
    valid_idx = rnd >= 0.8

    Xtr = x_train[train_idx]
    Xv = x_train[valid_idx]
    ytr = train.loc[train_idx, 'category_id'].values
    yv = train.loc[valid_idx, 'category_id'].values

    # to categorical
    ytr = to_categorical(ytr, num_classes=len(CATEGORIES))
    yv = to_categorical(yv, num_classes=len(CATEGORIES))

    num_classes = len(CATEGORIES)

    return Xtr, ytr, Xv, yv, num_classes

# read the test data
def get_test_data(data_dir, test_path):
    test = []
    test_dir = os.path.join(data_dir, test_path)
    for file in os.listdir(test_dir):
        test.append([test_path+'/{}'.format(file), file])
    test = pd.DataFrame(test, columns=['filepath', 'file'])

    x_test = np.zeros((len(test), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
    for i, filepath in tqdm(enumerate(test['filepath'])):
        img = read_img(data_dir, filepath, (INPUT_SIZE, INPUT_SIZE))
        x = preprocess_input(np.expand_dims(img.copy(), axis=0))
        x_test[i] = x
    return x_test

