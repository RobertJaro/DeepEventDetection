from pkg_resources import resource_filename
from tensorflow.keras import models
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from ded.config import resource_dir
from ded.event.data import DataGenerator

size = 400
batch_size = 5


class Trainer:

    def __init__(self):
        self.model: Model = models.load_model(resource_filename(resource_dir, "classify_model.h5"))

        self.generator = DataGenerator(size=size, batch_size=batch_size, event_type="FL")

    def train(self, epochs):
        checkpoint = ModelCheckpoint(resource_filename(resource_dir, "classify_model_checkpoint.h5"))
        tensorboard = TensorBoard(resource_filename(resource_dir, "logs"))
        callbacks = [checkpoint, tensorboard]

        self.model.fit_generator(self.generator, epochs=epochs, callbacks=callbacks)

        self.model.save(resource_filename(resource_dir, "classify_model.h5"))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(20)
