from pkg_resources import resource_filename
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import plot_model

from ded.config import resource_dir


def createModel():
    model: Model = InceptionV3(weights=None, input_shape=(800, 800, 1), classes=2)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

    return model


if __name__ == '__main__':
    model = createModel()
    plot_model(model, to_file=resource_filename(resource_dir, "model.png"), show_shapes=True)
    model.save(resource_filename(resource_dir, "classify_model.h5"))
