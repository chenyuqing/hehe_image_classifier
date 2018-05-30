# -*- coding: utf-8 -*-

from tools import get_train_val_data, get_test_data
from pretrained_models import vgg16_model,xception_model, res50_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from time import time
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'


data_dir = './input'
train_path = 'train_aug'

Xtr, ytr, Xv, yv, num_classes = get_train_val_data(data_dir, train_path)
print("Categories:{}".format(num_classes))
print('Xtr.shape:{}, Xv.shape:{}'.format(Xtr.shape, Xv.shape))

model = vgg16_model(num_classes)
# model = xception_model(num_classes)
# model = res50_model(num_classes)


# Tensorboard
tensorboard = TensorBoard(log_dir="logs/{}{}".format('vgg16_trainable10_aug_BS64', time()))

# checkpoibt
ck_fp = './saved_model/vgg16_trainable10_aug_BS64_weights.best.h5'
checkpoint = ModelCheckpoint(ck_fp, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

# early stopping
earlystop = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5, verbose=1, mode='auto')

batch_size = 64
nb_epoch = 50

# Start Fine-tuning
model.fit(Xtr, ytr,
          batch_size=batch_size,
          epochs=nb_epoch,
          shuffle=True,
          verbose=1,
          validation_data=(Xv, yv),
          callbacks=[tensorboard, checkpoint, earlystop]
          )

# Make predictions
predictions_valid = model.predict(Xv, batch_size=batch_size, verbose=1)

from sklearn.metrics import log_loss
# Cross-entropy loss score
score = log_loss(yv, predictions_valid)
print('score:',score)