from keras.callbacks import Callback

from training_utils.db import TrainingUtilsDB
from training_utils.db_tables_field_names import TrainingSessionMetricsFields
from data.openimages.constants import Constants

import os

class YoloV2MetricEvaluator(Callback):
    """
    Keras callback responsible for saving training progress throughout training in db.
    Args:
        - training_session_id       = id of the training session
        - start                     = timestamp of starting training
        - training_run_config_id    = id of the training run
        - metrics_to_evaluate       = dictionary of metrics to evaluate after each epoch
        - path_to_checkpoints       = path to folder containing checkpoints for current training session
    """

    def __init__(
        self,
        *,
        start,
        training_session_id,
        training_run_config_id,
        metrics_to_save = [],
        path_to_checkpoints
    ):
        self.start = start

        self.training_session_id = training_session_id
        self.training_run_config_id = training_run_config_id

        self.metrics_to_save = metrics_to_save
        self.path_to_checkpoints = path_to_checkpoints

    def update_num_of_epochs_processed(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(model_checkpoint_path)
