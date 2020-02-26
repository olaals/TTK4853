import keras
from keras.models import Model, Sequential
from keras import layers
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
import efficientnet.keras as efn
import cv2 as cv2
import tensorflow
from keras import backend as K
from tensorflow.python.client import device_lib


K.tensorflow_backend._get_available_gpus()


print(device_lib.list_local_devices())

print("hello")

train_data_dir = "train"
validation_data_dir = "valid"

img_width, img_height = 600, 300

num_epochs = 20
batch_size = 3
num_classes = 2

model = efn.EfficientNetB0(input_shape=(img_width, img_height, 3), weights=None, classes=2)

train_datagen = ImageDataGenerator(rotation_range=0.,
                                        width_shift_range=0,
                                        rescale= 1./255.,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical"
    )

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical"
    )


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps= validation_generator.n // batch_size,
        verbose=1
    )