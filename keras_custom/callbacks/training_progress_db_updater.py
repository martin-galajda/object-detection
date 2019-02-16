from keras.callbacks import Callback

from training_utils.db import TrainingUtilsDB
from training_utils.db_tables_field_names import TrainingSessionMetricsFields
from data.openimages.constants import Constants

import os


class TrainingProgressDbUpdater(Callback):
    """
    Keras callback responsible for saving training progress throughout training in db.
    Args:
        - training_session_id       = id of the training session
        - start                     = timestamp of starting training
        - training_run_config_id    = id of the training run
        - metrics_to_update         = list of metrics to update in DB after each epoch
        - path_to_checkpoints       = path to folder containing checkpoints for current training session
    """

    def __init__(self, *, start, training_session_id, training_run_config_id, metrics_to_save = [], path_to_checkpoints):
        self.start = start

        self.training_session_id = training_session_id
        self.training_run_config_id = training_run_config_id
        self.db = TrainingUtilsDB()
        self.metrics_to_save = metrics_to_save
        self.path_to_checkpoints = path_to_checkpoints

    def update_num_of_epochs_processed(self):
        self.db.increment_training_session_num_of_epochs(self.training_session_id)

    def on_epoch_end(self, epoch, logs=None):
        self.db.increment_training_session_num_of_epochs(self.training_session_id)

        prepared_metrics_to_save = []
        for metric_to_update in self.metrics_to_save:
            prepared_metrics_to_save += [{
                TrainingSessionMetricsFields.metric_type: metric_to_update,
                TrainingSessionMetricsFields.value: logs[metric_to_update],
            }]

        chk_path = f'{self.start.strftime(Constants.DATETIME_FORMAT)}_epoch_{epoch}.hdf5'
        model_checkpoint_path = os.path.join(self.path_to_checkpoints, chk_path)

        self.model.save(model_checkpoint_path)

        checkpoints_metadata_id = self.db.save_checkpoints_metadata(
            training_run_config_id = self.training_run_config_id,
            checkpoint_path = model_checkpoint_path
        )

        self.db.save_training_metrics(
            training_session_id=self.training_session_id,
            training_run_config_id=self.training_run_config_id,
            metrics_to_save = prepared_metrics_to_save,
            checkpoints_metadata_id = checkpoints_metadata_id
        )
