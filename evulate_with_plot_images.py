# -*- coding: utf-8 -*-
from keras.models import load_model
from tools import read_img
import numpy as np
import os

import keras

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_SIZE = 224
breed = 'bull_terrier_dog'

# CATEGORIES = ['alaska', 'bichons', 'chihuahua', 'golden', 'husky', 'labrador', 'papillon', 'samoyed', 'shepherd', 'teddy']

CATEGORIES = ['alaska', 'bichons', 'french_bulldog', 'chihuahua', 'golden', 'husky', 'labrador', 'papillon', 'samoyed', 'shepherd',
              'teddy', 'basset_hound_dog', 'bull_terrier_dog', 'chinese_sharpei', 'chow',  'cocker_spaniel', 'corgi_dog', 'dachshund_dog',
              'dalmatian_dog', 'doberman', 'eskimo_dog', 'great_greyhound_dog', 'italian_greyhound', 'japanese_spitz_dog', 'lhasa', 'maltese',
              'miniature_pinscher', 'miniature_schnauzer', 'newfoundland', 'pekingese_dog', 'pomeranian', 'poodle', 'rough_collie_dog',
              'saint_bernard', 'shetland_sheepdog', 'shiba_inu_dog', 'shih_tzu_dog', 'tibetan_mastiff', 'wolf_dog']
NUM_CLASSES = len(CATEGORIES)
data_dir = os.path.join('./input/test1/', breed)

al_list = os.listdir(data_dir)

# nb_test_images = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
nb_test_images = 25
print(nb_test_images)

# Load the model
model_path = './saved_model/vgg16_200i_trainable10_aug_BS64_split0.7_weights.best.h5'
loaded_model = load_model(model_path)

### Ploting the photos

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

j = int(np.sqrt(nb_test_images))
i = int(np.ceil(1. * nb_test_images / j))
print("i={}, j={}".format(i, j))
fig = plt.figure(1, figsize=(32, 32))
grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)

for i in range(nb_test_images):
    ax = grid[i]
    # Read one image
    img = read_img(data_dir, al_list[i], (INPUT_SIZE, INPUT_SIZE))
    ax.imshow(img / 255.)
    # x = keras.applications.xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    x = keras.applications.vgg16.preprocess_input(np.expand_dims(img.copy(), axis=0))

    preds = loaded_model.predict(x)
    max_index = np.argmax(preds)

    class_name = CATEGORIES[max_index]
    prob = preds[0][max_index]

    print('groud truth : {}, prediction : {}, probability : {}'.format(breed, class_name, prob))

    ax.text(10, 180, 'VGG16: %s (%.2f)' % (class_name , prob), color='w', backgroundcolor='k', alpha=0.8)
    ax.text(10, 200, 'LABEL: %s' % breed, color='k', backgroundcolor='w', alpha=0.8)
    ax.axis('off')
plt.show()

fig.savefig('./20_samples_test/'+breed+'.png', bbox_inches='tight')


