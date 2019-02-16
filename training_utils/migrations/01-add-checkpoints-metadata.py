import sqlite3

from training_utils.constants import (TrainingUtilsConstants,
                                      TrainingUtilsTableNames)
from training_utils.db_tables_field_names import (TrainingSessionMetricsFields,
                                                  TrainingRunConfigFields,
                                                  CheckpointsMetadataFields)


def main():
    db_conn = sqlite3.connect(TrainingUtilsConstants.PATH_TO_TRAINING_METADATA_DB)
    db_cursor = db_conn.cursor()

    db_cursor.executescript(f'''

        CREATE TABLE IF NOT EXISTS {TrainingUtilsTableNames.CHECKPOINTS_METADATA} (
            {CheckpointsMetadataFields.id} INTEGER PRIMARY KEY NOT NULL,
            {CheckpointsMetadataFields.checkpoint_path} VARCHAR (500) NOT NULL,
            {CheckpointsMetadataFields.training_run_config_id} INTEGER
                REFERENCES {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION}({TrainingRunConfigFields.id}) NOT NULL,


            {CheckpointsMetadataFields.created_at} TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            {CheckpointsMetadataFields.updated_at} TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
        );

        ALTER TABLE {TrainingUtilsTableNames.TRAINING_SESSION_METRICS}
            ADD COLUMN {TrainingSessionMetricsFields.checkpoints_metadata_id} INTEGER
                REFERENCES {TrainingUtilsTableNames.CHECKPOINTS_METADATA}({CheckpointsMetadataFields.id})
        ;
    ''')

    db_cursor.executescript(f'''
        CREATE INDEX IF NOT EXISTS
            {TrainingUtilsTableNames.TRAINING_SESSION_METRICS}_checkpoints_metadata_idx
            ON {TrainingUtilsTableNames.TRAINING_SESSION_METRICS}({TrainingSessionMetricsFields.checkpoints_metadata_id});

        CREATE INDEX IF NOT EXISTS
            {TrainingUtilsTableNames.CHECKPOINTS_METADATA}_training_run_config_idx
            ON {TrainingUtilsTableNames.CHECKPOINTS_METADATA}({CheckpointsMetadataFields.training_run_config_id});
    ''')

    db_conn.commit()


if __name__ == "__main__":
    main()
