# -*- coding: utf-8 -*-
from keras.models import load_model
from tools import read_img
import numpy as np
import os

import keras

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_SIZE = 224
# breed = 'alaska'

# CATEGORIES = ['alaska', 'bichons', 'chihuahua', 'golden', 'husky', 'labrador', 'papillon', 'samoyed', 'shepherd', 'teddy']

CATEGORIES = ['alaska', 'bichons', 'french_bulldog', 'chihuahua', 'golden', 'husky', 'labrador', 'papillon', 'samoyed', 'shepherd',
              'teddy', 'basset_hound_dog', 'bull_terrier_dog', 'chinese_sharpei', 'chow',  'cocker_spaniel', 'corgi_dog', 'dachshund_dog',
              'dalmatian_dog', 'doberman', 'eskimo_dog', 'great_greyhound_dog', 'italian_greyhound', 'japanese_spitz_dog', 'lhasa', 'maltese',
              'miniature_pinscher', 'miniature_schnauzer', 'newfoundland', 'pekingese_dog', 'pomeranian', 'poodle', 'rough_collie_dog',
              'saint_bernard', 'shetland_sheepdog', 'shiba_inu_dog', 'shih_tzu_dog', 'tibetan_mastiff', 'wolf_dog']
NUM_CLASSES = len(CATEGORIES)



# Load the model
model_path = './saved_model/vgg16_200i_trainable10_aug_BS64_split0.7_weights.best.h5'
loaded_model = load_model(model_path)

def evulate_each_class(breed):
    acc = 0
    data_dir = os.path.join('./input/test1/', breed)

    al_list = os.listdir(data_dir)
    print(al_list)
    for filepath in al_list:
        # Read one image
        img = read_img(data_dir, filepath, (INPUT_SIZE, INPUT_SIZE))
        # x = keras.applications.xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
        x = keras.applications.vgg16.preprocess_input(np.expand_dims(img.copy(), axis=0))

        preds = loaded_model.predict(x)
        max_index = np.argmax(preds)

        class_name = CATEGORIES[max_index]
        prob = preds[0][max_index]
        if class_name == breed:
            acc += 1

        print('ground truth : {}, prediction : {}, probability : {}'.format(breed, class_name, prob))
    avg_acc = acc/len(al_list)*1.0
    print('accuracy on {} images of {} is {}'.format(len(al_list), breed, avg_acc))
    return avg_acc

sum_acc = 0
for breed in CATEGORIES:
    a_a = evulate_each_class(breed)
    sum_acc += a_a
print('{} dog breeds mean average accuracy : {}'.format(NUM_CLASSES, sum_acc/NUM_CLASSES))
### Ploting the photos

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
#
# j = int(np.sqrt(NUM_CLASSES))
# i = int(np.ceil(1. * NUM_CLASSES / j))
# print("i={}, j={}".format(i, j))
# fig = plt.figure(1, figsize=(16, 16))
# grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)
#
#
# for i in range(len(al_list)):
#     ax = grid[i]
#     # Read one image
#     img = read_img(data_dir, al_list[i], (INPUT_SIZE, INPUT_SIZE))
#     ax.imshow(img / 255.)
#     # x = keras.applications.xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
#     x = keras.applications.vgg16.preprocess_input(np.expand_dims(img.copy(), axis=0))
#
#     preds = loaded_model.predict(x)
#     max_index = np.argmax(preds)
#     # print('the min prob:{}'.format(preds[0][np.argmin(preds)]))
#     # print(preds)
#     # min_sum += preds[0][np.argmin(preds)]
#     # print(preds)
#     # _,cate,prob =  keras.applications.xception.decode_predictions(prediction)
#     # _,cate,prob = keras.applications.vgg16.decode_predictions(prediction, top=1)[0][0]
#     print('prediction : {}, probability : {}'.format(CATEGORIES[max_index], preds[0][max_index]))
#     class_name = CATEGORIES[max_index]
#     prob = preds[0][max_index]
#
#     ax.text(10, 180, 'Xception: %s (%.2f)' % (class_name , prob), color='w', backgroundcolor='k', alpha=0.8)
#     ax.text(10, 200, 'LABEL: %s' % breed, color='k', backgroundcolor='w', alpha=0.8)
#     ax.axis('off')
# # print('the average of min probs: {}'.format(min_sum/len(al_list)))
#
# plt.show()
#
