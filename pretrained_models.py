# -*- coding: utf-8 -*-

from keras.applications import vgg16, xception, resnet50
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Model

import os

data_dir = os.path.join(os.getcwd(), 'pretrained_models')

def vgg16_model(num_classes=None, pooling='avg'):
    # create the base pre-trained model

    # download model from internet
    # base_model = vgg16.VGG16(weights='imagenet', include_top=False, pooling=POOLING)

    # load the model from local disk
    base_model = vgg16.VGG16(weights=None, include_top=False, pooling=pooling)
    base_model.load_weights(os.path.join(data_dir,'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'))
    # print the number of layers
    print('VGG16 # layers', len(base_model.layers))

    # add a global spatial average pooling layer
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # First train the only the top layers
    for layer in base_model.layers[10:]:
        layer.trainable = False

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def xception_model(num_classes=None, pooling='avg'):
    # Pre-trained CNN Model using imagenet dataset for pre-trained weights
    base_model = xception.Xception(weights=None, include_top=False, pooling=pooling)
    weight_path = os.path.join(data_dir,'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print(weight_path)
    base_model.load_weights(weight_path)

    # print the number of layers
    print('Xception # layers', len(base_model.layers))

    # Top Model Block
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Add your top layer block to your base model
    model = Model(base_model.input, predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    # complie the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def res50_model(num_classes=None, pooling='avg'):
    # load the model from local disk
    base_model = resnet50.ResNet50(weights=None, include_top=False, pooling=pooling)
    base_model.load_weights(os.path.join(data_dir, 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))
    # print the number of layers
    print('ResNet 50 # layers', len(base_model.layers))

    # add a global spatial average pooling layer
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # First train the only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model