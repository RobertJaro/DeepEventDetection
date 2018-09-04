from pkg_resources import resource_filename
from tensorflow import keras
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, Conv2D, Dropout, Flatten, Dense
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import plot_model

from ded.config import resource_dir


def createModel(resolution=800):
    model = keras.Sequential()
    model.add(Reshape((resolution, resolution, 1), input_shape=(resolution, resolution)))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation="sigmoid"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

    return model


if __name__ == '__main__':
    model = createModel()
    plot_model(model, to_file=resource_filename(resource_dir, "model.png"), show_shapes=True)
    model.save(resource_filename(resource_dir, "classify_model.h5"))
