import sqlite3

from training_utils.constants import (TrainingUtilsConstants,
                                      TrainingUtilsTableNames)
from training_utils.db_tables_field_names import (MultilabelClassificationMetadataFields,
                                                  TrainingRunConfigFields,
                                                  TrainingSessionFields,
                                                  TrainingSessionMetricsFields,
                                                  CheckpointsMetadataFields)


def main():
    db_conn = sqlite3.connect(TrainingUtilsConstants.PATH_TO_TRAINING_METADATA_DB)
    db_cursor = db_conn.cursor()

    db_cursor.executescript(f'''
        CREATE TABLE IF NOT EXISTS {TrainingUtilsTableNames.TRAINING_SESSION} (
            {TrainingSessionFields.id} INTEGER PRIMARY KEY NOT NULL,
            {TrainingSessionFields.num_of_epochs_processed} INTEGER NOT NULL DEFAULT 0,
            {TrainingSessionFields.num_of_examples_processed} INTEGER NOT NULL DEFAULT 0,

            {TrainingSessionFields.num_of_training_images} INTEGER,
            {TrainingSessionFields.num_of_validations_images} INTEGER,

            {TrainingSessionFields.batch_size} INTEGER NOT NULL,

            {TrainingSessionFields.model} VARCHAR (100) NOT NULL,
            {TrainingSessionFields.training_dataset_type} VARCHAR (100) NOT NULL,

            {TrainingSessionFields.optimizer} VARCHAR (100) NOT NULL,

            {TrainingSessionFields.unfreeze_top_k_layers} VARCHAR(100) NOT NULL,

            {TrainingSessionFields.use_multitarget_learning} INTEGER NOT NULL,

            {TrainingSessionFields.tensorboard_logs_path} VARCHAR (500),
            {TrainingSessionFields.model_checkpoints_path} VARCHAR (500),
            
            {TrainingSessionFields.created_at} TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            {TrainingSessionFields.updated_at} TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION} (
            {TrainingRunConfigFields.id} INTEGER PRIMARY KEY NOT NULL,
            {TrainingRunConfigFields.training_session_id} INTEGER REFERENCES training_session ({TrainingSessionFields.id}) NOT NULL,
 
            {TrainingRunConfigFields.save_checkpoint_every_n_minutes} INTEGER,
            {TrainingRunConfigFields.job_id} VARCHAR (100) NOT NULL,

            {TrainingRunConfigFields.optimizer_lr} REAL,

            {TrainingRunConfigFields.generator_max_queue_size} INTEGER NOT NULL,
            {TrainingRunConfigFields.continue_training_allowed_different_config_keys} VARCHAR (1000) NOT NULL,
            {TrainingRunConfigFields.continue_from_last_checkpoint} INTEGER NOT NULL,
            {TrainingRunConfigFields.validation_data_use_percentage} REAL NOT NULL,

            {TrainingRunConfigFields.last_checkpoint_path} VARCHAR (1000),
            
            {TrainingRunConfigFields.tensorboard_monitor_freq} INTEGER,
            {TrainingRunConfigFields.copy_db_to_scratch} INTEGER,

            {TrainingRunConfigFields.use_multiprocessing} INTEGER NOT NULL,

            {TrainingRunConfigFields.workers} INTEGER NOT NULL,
            {TrainingRunConfigFields.use_gpu} INTEGER NOT NULL,

            {TrainingRunConfigFields.created_at} TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            {TrainingRunConfigFields.updated_at} TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {TrainingUtilsTableNames.MULTILABEL_CLASSIFICATION_METADATA} (
            {MultilabelClassificationMetadataFields.id} INTEGER PRIMARY KEY NOT NULL,
            {MultilabelClassificationMetadataFields.training_session_id} INTEGER 
                REFERENCES {TrainingUtilsTableNames.TRAINING_SESSION}(id) NOT NULL,
            {MultilabelClassificationMetadataFields.db_images_path} VARCHAR (200),
            {MultilabelClassificationMetadataFields.db_image_labels_path} VARCHAR (200)
        );

        CREATE TABLE IF NOT EXISTS {TrainingUtilsTableNames.TRAINING_SESSION_METRICS} (
            {TrainingSessionMetricsFields.id} INTEGER PRIMARY KEY NOT NULL,
            {TrainingSessionMetricsFields.training_session_id} INTEGER 
                REFERENCES {TrainingUtilsTableNames.TRAINING_SESSION}(id) NOT NULL,
            {TrainingSessionMetricsFields.training_run_configuration_id} INTEGER
                REFERENCES {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION}({TrainingRunConfigFields.id}) NOT NULL,
            {TrainingSessionMetricsFields.metric_type} VARCHAR(100) NOT NULL,
            {TrainingSessionMetricsFields.value} REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {TrainingUtilsTableNames.CHECKPOINTS_METADATA} (
            {CheckpointsMetadataFields.id} INTEGER PRIMARY KEY NOT NULL,
            {CheckpointsMetadataFields.checkpoint_path} VARCHAR (500) NOT NULL,
            {CheckpointsMetadataFields.training_run_config_id} INTEGER
                REFERENCES {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION}({TrainingRunConfigFields.id}) NOT NULL,
    
            {CheckpointsMetadataFields.created_at} TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            {CheckpointsMetadataFields.updated_at} TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {TrainingUtilsTableNames.TRAINING_SESSION_METRICS} (
            {TrainingSessionMetricsFields.id} INTEGER PRIMARY KEY NOT NULL,
            {TrainingSessionMetricsFields.training_session_id} INTEGER 
                REFERENCES {TrainingUtilsTableNames.TRAINING_SESSION}(id) NOT NULL,
            {TrainingSessionMetricsFields.training_run_configuration_id} INTEGER
                REFERENCES {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION}({TrainingRunConfigFields.id}) NOT NULL,

            {TrainingSessionMetricsFields.checkpoints_metadata_id} INTEGER
                REFERENCES {TrainingUtilsTableNames.CHECKPOINTS_METADATA}({CheckpointsMetadataFields.id}),

            {TrainingSessionMetricsFields.metric_type} VARCHAR(100) NOT NULL,
            {TrainingSessionMetricsFields.value} REAL NOT NULL
        );
    ''')

    db_cursor.executescript(f'''
        CREATE INDEX IF NOT EXISTS
            {TrainingUtilsTableNames.TRAINING_SESSION_METRICS}_training_session_idx
            ON {TrainingUtilsTableNames.TRAINING_SESSION_METRICS}({TrainingSessionMetricsFields.training_session_id});

        CREATE INDEX IF NOT EXISTS
            {TrainingUtilsTableNames.TRAINING_SESSION_METRICS}_training_run_config_idx
            ON {TrainingUtilsTableNames.TRAINING_SESSION_METRICS}({TrainingSessionMetricsFields.training_run_configuration_id});

        CREATE INDEX IF NOT EXISTS
            {TrainingUtilsTableNames.TRAINING_SESSION_METRICS}_checkpoints_metadata_idx
            ON {TrainingUtilsTableNames.TRAINING_SESSION_METRICS}({TrainingSessionMetricsFields.checkpoints_metadata_id});

        CREATE INDEX IF NOT EXISTS
            {TrainingUtilsTableNames.MULTILABEL_CLASSIFICATION_METADATA}_training_session_idx
            ON {TrainingUtilsTableNames.MULTILABEL_CLASSIFICATION_METADATA}({MultilabelClassificationMetadataFields.training_session_id});

        CREATE INDEX IF NOT EXISTS
            {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION}_training_session_idx
            ON {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION}({TrainingRunConfigFields.training_session_id});
    ''')

    db_conn.commit()


if __name__ == "__main__":
    main()
