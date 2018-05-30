# -*- coding: utf-8 -*-
from keras.models import load_model
from tools import read_img
import numpy as np
import os

import keras

INPUT_SIZE = 224
breed = 'samoyed'

CATEGORIES = ['alaska', 'bichons', 'chihuahua', 'golden', 'husky', 'labrador', 'papillon', 'samoyed', 'shepherd', 'teddy']
NUM_CLASSES = len(CATEGORIES)
data_dir = os.path.join('./input/test1/', breed)
# filepath = '503d269759ee3d6d00be6d4543166d224e4adec0.jpg'


al_list = os.listdir(data_dir)
print(al_list)

# Load the model
model_path = './saved_model/xception_aug_BS32_weights.best.h5'
loaded_model = load_model(model_path)

# for filepath in al_list:
#     # Read one image
#     img = read_img(data_dir, filepath, (INPUT_SIZE, INPUT_SIZE))
#     # x = keras.applications.xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
#     x = keras.applications.vgg16.preprocess_input(np.expand_dims(img.copy(), axis=0))
#
#     preds = loaded_model.predict(x)
#     max_index = np.argmax(preds)
#     # print(preds)
#     # _,cate,prob =  keras.applications.xception.decode_predictions(prediction)
#     # _,cate,prob = keras.applications.vgg16.decode_predictions(prediction, top=1)[0][0]
#     print('prediction : {}, probability : {}'.format(CATEGORIES[max_index], preds[0][max_index]))

### Ploting the photos

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

j = int(np.sqrt(NUM_CLASSES))
i = int(np.ceil(1. * NUM_CLASSES / j))
print("i={}, j={}".format(i, j))
fig = plt.figure(1, figsize=(16, 16))
grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)

min_sum = 0
for i in range(len(al_list)):
    ax = grid[i]
    # Read one image
    img = read_img(data_dir, al_list[i], (INPUT_SIZE, INPUT_SIZE))
    ax.imshow(img / 255.)
    # x = keras.applications.xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    x = keras.applications.vgg16.preprocess_input(np.expand_dims(img.copy(), axis=0))

    preds = loaded_model.predict(x)
    max_index = np.argmax(preds)
    print('the min prob:{}'.format(preds[0][np.argmin(preds)]))
    print(preds)
    min_sum += preds[0][np.argmin(preds)]
    # print(preds)
    # _,cate,prob =  keras.applications.xception.decode_predictions(prediction)
    # _,cate,prob = keras.applications.vgg16.decode_predictions(prediction, top=1)[0][0]
    print('prediction : {}, probability : {}'.format(CATEGORIES[max_index], preds[0][max_index]))
    class_name = CATEGORIES[max_index]
    prob = preds[0][max_index]

    ax.text(10, 180, 'Xception: %s (%.2f)' % (class_name , prob), color='w', backgroundcolor='k', alpha=0.8)
    ax.text(10, 200, 'LABEL: %s' % breed, color='k', backgroundcolor='w', alpha=0.8)
    ax.axis('off')
print('the average of min probs: {}'.format(min_sum/len(al_list)))
plt.show()

