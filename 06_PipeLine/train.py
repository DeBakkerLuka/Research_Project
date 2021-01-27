#importing tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import os
import config

folders = glob(config.train_dir + '/*')

def build_model(folders):
    model = model = tf.keras.applications.VGG19(input_shape = (config.image_height,config.image_width,config.channels), weights = 'imagenet', include_top = False)

    if config.MODEL == "vgg16":
        model = model = tf.keras.applications.VGG16(input_shape = (config.image_height,config.image_width,config.channels), weights = 'imagenet', include_top = False)
    if config.MODEL == "resnet152":
        model = model = tf.keras.applications.ResNet152(input_shape = (config.image_height,config.image_width,config.channels), weights = 'imagenet', include_top = False)
    if config.MODEL == "inceptionV3":
        model = model = tf.keras.applications.InceptionV3(input_shape = (config.image_height,config.image_width,config.channels), weights = 'imagenet', include_top = False)

    for layer in model.layers:
        layer.trainable = False

    x =  Flatten()(model.output)
    x1 = Dense(4096)(x)
    d1 = Dropout(.1)(x1)
    x2 = Dense(4096)(d1)
    d2 = Dropout(.1)(x2)
    x3 = Dense(4096)(d2)
    x4 = Dense(1024)(x3)
    prediction = Dense(len(folders), activation='sigmoid')(x4)
    model = Model(inputs = model.input, outputs = prediction)

    return model 



model = build_model(folders)


opt = Adam(lr=0.00001)
model.compile(
        loss = 'binary_crossentropy',
        optimizer = opt,
        metrics = ['accuracy']
        )


train_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range= 0.2,
                                   zoom_range= 0.2,
                                   horizontal_flip= True)
test_datagen = ImageDataGenerator(rescale= 1./255)
training_set = train_datagen.flow_from_directory(config.train_dir,
                                                       target_size= (config.image_height,config.image_width),
                                                       class_mode='categorical')
test_set = test_datagen.flow_from_directory(config.test_dir,
                                            target_size= (config.image_height, config.image_width),
                                            class_mode= 'categorical')


class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(training_set.classes), 
                training_set.classes)
class_weights = dict(enumerate(class_weights))


earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
r = model.fit(
        training_set,
        validation_data= test_set,
        epochs=config.EPOCHS,
        steps_per_epoch= len(training_set),
        validation_steps=len(test_set),
        class_weight=class_weights,
        callbacks=[earlystopping]
        )

model.save(config.save_model_dir + '/Final_Model_VGG19.h5')
print('Successfully saved')