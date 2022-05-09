import os
import pathlib
from datetime import datetime
from human_id import generate_id
from matplotlib import pyplot as plt
from tensorflow import keras
import uuid
import pickle


class Experiment:
    def __init__(self, name=None, base_path="experiments/"):
        self.base_path = base_path
        self.model = None
        self.params = None
        self.history = None

        if name is None:
            # Create an experiment
            self.name = "%s_%s" % (datetime.now().strftime("%d-%m-%Y_%H%M"), generate_id(seed=uuid.uuid4()))
            self.main_dir = os.path.join(self.base_path, self.name)
            pathlib.Path(self.main_dir).mkdir(parents=True, exist_ok=True)

            self.model_dir = os.path.join(self.main_dir, 'model')
            pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

            self.vis_dir = os.path.join(self.main_dir, 'vis')
            pathlib.Path(self.vis_dir).mkdir(parents=True, exist_ok=True)
        else:
            # Load experiment
            self.name = name
            self.main_dir = os.path.join(self.base_path, self.name)
            self.model_dir = os.path.join(self.main_dir, 'model')
            self.vis_dir = os.path.join(self.main_dir, 'vis')

            with open(os.path.join(self.main_dir, 'params.pckl'), 'rb') as f:
                self.params = pickle.load(f)

            with open(os.path.join(self.main_dir, 'history.pckl'), 'rb') as f:
                self.history = pickle.load(f)

    def load_model(self):
        self.model = keras.models.load_model(os.path.join(self.model_dir, 'model'))
        return self.model

    def save(self, params, model=None, history=None):
        if params is not None:
            with open(os.path.join(self.main_dir, 'params.pckl'), 'wb') as f:
                pickle.dump(params, f)
            with open(os.path.join(self.main_dir, 'params.txt'), 'w') as f:
                for key, value in params.items():
                    f.write("%s: %s\n" % (key, str(value)))
        if model is not None:
            model.encoder.save(os.path.join(self.model_dir, 'encoder'))
            model.decoder.save(os.path.join(self.model_dir, 'decoder'))
            model.save(os.path.join(self.model_dir, 'model'), save_format="tf")
        if history is not None:
            with open(os.path.join(self.main_dir, 'history.pckl'), 'wb') as f:
                pickle.dump(history, f)

    def plot(self, train_loss, val_loss):
        plt.figure()
        plt.semilogy(train_loss)
        plt.semilogy(val_loss)
        plt.title('Training and Validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig(os.path.join(self.vis_dir, 'losses' + '.png'))
        plt.close()


def load_experiments(with_params=None, base_path="experiments/"):
    # Load all experiments with basic configuration into a dictionary
    experiments = {}

    for directory in os.listdir(base_path):
        d = os.path.join(base_path, directory)
        if os.path.isdir(d):
            exp = Experiment(name=directory, base_path=base_path)

            if len(exp.params.items() & with_params.items()) == len(with_params.items()):
                # Save to dict
                experiments[directory] = {
                    'params': exp.params,
                    'experiment': exp
                }
    return experiments
