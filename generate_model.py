import numpy as np
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import * 
from matplotlib import pyplot as plt


train_path = 'train'
valid_path = 'valid'
test_path = 'valid'

train_batches = ImageDataGenerator(rescale = 1/255.0).flow_from_directory(train_path, target_size=(300,300), classes= ['stopsign', 'not_stopsign'], batch_size = 10)
valid_batches = ImageDataGenerator(rescale = 1/255.0).flow_from_directory(valid_path, target_size=(300,300), classes= ['stopsign', 'not_stopsign'], batch_size = 10)
test_batches = valid_batches

print("Hello world")

model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (300,300,3)),
    Conv2D(32, (3,3), activation = 'relu'),
    Conv2D(64, (3,3), activation = 'relu'),
    Conv2D(32, (3,3), activation = 'relu'),
    Flatten(),
    Dense(2 , activation='softmax')
])

model.compile(Adam(lr = .001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=8, 
                    validation_data = valid_batches, validation_steps = 2, epochs = 5, verbose = 2)

