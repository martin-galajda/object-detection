from collections import namedtuple

_training_session_fields = [
    'id',
    'num_of_epochs_processed',
    'num_of_examples_processed',
    'num_of_training_images',
    'num_of_validations_images',
    'batch_size',
    'model',
    'training_dataset_type',
    'optimizer',
    'unfreeze_top_k_layers',
    'use_multitarget_learning',
    'tensorboard_logs_path',
    'model_checkpoints_path',
    'created_at',
    'updated_at'
]
TrainingSessionFieldsConstructor = namedtuple('TrainingSessionFields', _training_session_fields)
TrainingSessionFields = TrainingSessionFieldsConstructor(
    *_training_session_fields
)


_training_run_config_fields = [
    'id',
    'training_session_id',
    'save_checkpoint_every_n_minutes',
    'job_id',
    'optimizer_lr',
    'generator_max_queue_size',
    'continue_training_allowed_different_config_keys',
    'continue_from_last_checkpoint',
    'validation_data_use_percentage',
    'last_checkpoint_path',
    'tensorboard_monitor_freq',
    'copy_db_to_scratch',
    'use_multiprocessing',
    'workers',
    'use_gpu',
    'created_at',
    'updated_at'
]
TrainingRunConfigConstructor = namedtuple('TrainingRunConfigurationFields', _training_run_config_fields)
TrainingRunConfigFields = TrainingRunConfigConstructor(
    *_training_run_config_fields
)


_multilabel_classification_metadata_fields = [
    'id',
    'training_session_id',
    'db_images_path',
    'db_image_labels_path'
]
MultilabelClassificationMetadataConstructor = namedtuple('MultilabelClassificationMetadataFields', _multilabel_classification_metadata_fields)
MultilabelClassificationMetadataFields = MultilabelClassificationMetadataConstructor(
    *_multilabel_classification_metadata_fields
)


_training_sess_metrics_fields = [
    'id',
    'training_session_id',
    'training_run_configuration_id',
    'checkpoints_metadata_id',
    'metric_type',
    'value'
]
TrainingSessionMetricsConstructor = namedtuple('TrainingSessionFields', _training_sess_metrics_fields)
TrainingSessionMetricsFields = TrainingSessionMetricsConstructor(
    *_training_sess_metrics_fields
)


_checkpoints_metadata_fields = [
    'id',
    'checkpoint_path',
    'training_run_config_id',
    'created_at',
    'updated_at'
]
CheckpointsMetadataConstructor = namedtuple('CheckpointsMetadataFields', _checkpoints_metadata_fields)
CheckpointsMetadataFields = CheckpointsMetadataConstructor(
    *_checkpoints_metadata_fields
)
