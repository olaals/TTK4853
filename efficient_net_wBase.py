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

def make_model():
    """
    Creates an EfficientNet model with parameters specified `num_classes` outputs
    and `img_width` x `img_height` input shape.
    Args:
    Returns: an EfficientNet model with specified global parameters.
    """
    # create the base pre-trained model

    base_model = efn.EfficientNetB0(input_shape=(img_width, img_height, 3), include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return base_model, model


train_data_dir = "train"
validation_data_dir = "valid"

output_weights = "efficientnetb0.hdf5"

img_width, img_height = 600, 300

num_epochs = 40
batch_size = 3
num_classes = 2

_, model = make_model()

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

tensor_board = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
log_file_path = "logs/training.log"
csv_logger = CSVLogger(log_file_path, append=False)
reduce_lr = ReduceLROnPlateau("val_acc", factor=0.1, patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(output_weights, monitor="val_acc", verbose=1, save_best_only=True, save_weights_only=False)
callbacks = [tensor_board, model_checkpoint, csv_logger, reduce_lr]


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps= validation_generator.n // batch_size,
        callbacks=callbacks,
        verbose=1
    )