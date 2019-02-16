import os
from time import time as timer

from keras.callbacks import Callback

from checkpoints.utils import make_checkpoint_model_name
from training_utils.db import TrainingUtilsDB


def _update_training_run_config_last_chkpoint(training_run_config, last_chkpoint_path):
    db = TrainingUtilsDB()

    db.update_training_run_config(
        id = training_run_config.id,
        last_checkpoint_path = last_chkpoint_path
    )


class ModelSaver(Callback):
    """
    Keras callback responsible for saving models throughout training (and after training).
    Args:
      - checkpoint_model_dir  = path to the directory containing checkpoints
      - every_n_minutes       = how often to save the model
      - datetime_start        = Datetime object created at the start of (re)training
      - training_run_config   = Training run configuration fields from DB
    """

    def __init__(self, checkpoint_model_dir, every_n_minutes = None, *, datetime_start, training_run_config):
        self.every_n_minutes = every_n_minutes
        self.start = timer()
        self.batch = 0
        self.last_minute_saved = 0
        self.directory = checkpoint_model_dir
        self.datetime_start = datetime_start
        self.db_training_run_config = training_run_config

        os.makedirs(self.directory, exist_ok=True)

        print("ModelSaver.__init()__ ---> going to save model every %f minutes." % self.every_n_minutes)

    def save_model(self, stage):
        print("Saving model for stage = %s." % stage)
        path_to_check_points = os.path.join(os.path.dirname(__file__), '..', self.directory)

        if not os.path.isdir(path_to_check_points):
            os.makedirs(path_to_check_points)

        name = make_checkpoint_model_name(self.datetime_start, stage)
        model_checkpoint_path = os.path.join(path_to_check_points, name)
        self.model.save(model_checkpoint_path)

        _update_training_run_config_last_chkpoint(self.db_training_run_config, model_checkpoint_path)

    def on_batch_end(self, batch, logs={}):
        seconds_diff = timer() - self.start
        minutes_diff = seconds_diff // 60

        print(minutes_diff)

        if minutes_diff > 0 and ((minutes_diff - self.last_minute_saved) >= self.every_n_minutes):
            stage = str(int(minutes_diff)) + "_minutes"
            self.save_model(stage)
            self.last_minute_saved = minutes_diff
