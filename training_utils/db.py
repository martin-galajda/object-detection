import sqlite3

from training_utils.constants import TrainingUtilsConstants, TrainingUtilsTableNames
from training_utils.db_tables_field_names import TrainingSessionFields, \
    TrainingSessionFieldsConstructor, \
    TrainingSessionMetricsFields, \
    TrainingRunConfigFields, \
    TrainingRunConfigConstructor, \
    CheckpointsMetadataFields


class TrainingUtilsDB():
    def __init__(
        self,
        db_path: str = TrainingUtilsConstants.PATH_TO_TRAINING_METADATA_DB
    ):
        self.db_path = db_path

    def _get_db_conn(self):
        conn = sqlite3.connect(self.db_path, isolation_level=None)

        return conn

    def get_training_session_by_id(
        self,
        id: int
    ) -> TrainingSessionFields:
        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.execute(f'''
            SELECT {', '.join(TrainingSessionFields._fields)}
            FROM {TrainingUtilsTableNames.TRAINING_SESSION}
            WHERE {TrainingSessionFields.id} = :{TrainingSessionFields.id}
        ''', {
            TrainingSessionFields.id: id
        })

        results = cursor.fetchone()

        if len(results) == 0:
            print(f'''No training session matched for
                id = {id}
            ''')
            return None

        training_session = TrainingSessionFieldsConstructor(*results)

        print(f'''
            Found training session by id: {training_session}
        ''')

        return training_session

    def get_training_session(
        self,
        *,
        model: str,
        optimizer: str,
        training_dataset_type: str,
        unfreeze_top_k_layers: int,
        use_multitarget_learning: int
        # num_of_training_images: int,
        # num_of_validations_images: int,
        # model_checkpoints_path: str,
        # tensorboard_logs_path: str,
        # batch_size: int
    ) -> TrainingSessionFields:
        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.execute(f'''
            SELECT {', '.join(TrainingSessionFields._fields)}
            FROM {TrainingUtilsTableNames.TRAINING_SESSION}
            WHERE {TrainingSessionFields.model} = :{TrainingSessionFields.model}
                AND {TrainingSessionFields.optimizer} = :{TrainingSessionFields.optimizer}
                AND {TrainingSessionFields.training_dataset_type} = :{TrainingSessionFields.training_dataset_type}
                AND {TrainingSessionFields.unfreeze_top_k_layers} = :{TrainingSessionFields.unfreeze_top_k_layers}
                AND {TrainingSessionFields.use_multitarget_learning} = :{TrainingSessionFields.use_multitarget_learning}
        ''', {
            TrainingSessionFields.model:                    model,
            TrainingSessionFields.optimizer:                optimizer,
            TrainingSessionFields.training_dataset_type:    training_dataset_type,
            TrainingSessionFields.unfreeze_top_k_layers:    unfreeze_top_k_layers,
            TrainingSessionFields.use_multitarget_learning: use_multitarget_learning,
        })

        results = cursor.fetchall()

        if len(results) > 1:
            raise ValueError(f'Expected to match one training session, but matched {len(results)} sessions.')

        if len(results) == 0:
            print(f'''No training session matched for
                model = {model}
                training_dataset_type = {training_dataset_type}
                unfreeze_top_k_layers = {unfreeze_top_k_layers}
                use_multitarget_learning = {use_multitarget_learning}
                optimizer = {optimizer}
            ''')
            return None

        training_session = TrainingSessionFieldsConstructor(*results[0])

        print(f'''
            Found training session: {training_session}
        ''')

        return training_session

    # 'id',
    # 'num_of_epochs',
    # 'num_of_training_images',
    # 'num_of_validations_images',
    # 'batch_size',
    # 'model',
    # 'training_dataset_type',
    # 'optimizer',
    # 'unfreeze_top_k_layers',
    # 'use_multitarget_learning',
    # 'tensorboard_logs_path',
    # 'model_checkpoints_path',
    # 'created_at',
    # 'updated_at'

    def create_training_session(
        self,
        *,
        model: str,
        optimizer: str,
        training_dataset_type: str,
        unfreeze_top_k_layers: int,
        use_multitarget_learning: int,
        num_of_training_images: int,
        num_of_validations_images: int,
        model_checkpoints_path: str,
        tensorboard_logs_path: str,
        batch_size: int,
        num_of_epochs_processed: int = 0
    ):
        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.execute(f'''
            INSERT INTO {TrainingUtilsTableNames.TRAINING_SESSION} (
                {TrainingSessionFields.model}, {TrainingSessionFields.optimizer}, {TrainingSessionFields.training_dataset_type},
                {TrainingSessionFields.unfreeze_top_k_layers}, {TrainingSessionFields.use_multitarget_learning}, 
                {TrainingSessionFields.num_of_training_images}, {TrainingSessionFields.num_of_validations_images},
                {TrainingSessionFields.model_checkpoints_path}, {TrainingSessionFields.tensorboard_logs_path},
                {TrainingSessionFields.batch_size}, {TrainingSessionFields.num_of_epochs_processed}
            ) VALUES (
                :{TrainingSessionFields.model}, :{TrainingSessionFields.optimizer}, :{TrainingSessionFields.training_dataset_type},
                :{TrainingSessionFields.unfreeze_top_k_layers}, :{TrainingSessionFields.use_multitarget_learning},
                :{TrainingSessionFields.num_of_training_images}, :{TrainingSessionFields.num_of_validations_images},
                :{TrainingSessionFields.model_checkpoints_path}, :{TrainingSessionFields.tensorboard_logs_path},
                :{TrainingSessionFields.batch_size}, :{TrainingSessionFields.num_of_epochs_processed})
        ''', {
            TrainingSessionFields.model:                        model,
            TrainingSessionFields.optimizer:                    optimizer,
            TrainingSessionFields.training_dataset_type:        training_dataset_type,
            TrainingSessionFields.unfreeze_top_k_layers:        unfreeze_top_k_layers,
            TrainingSessionFields.use_multitarget_learning:     use_multitarget_learning,
            TrainingSessionFields.num_of_training_images:       num_of_training_images,
            TrainingSessionFields.num_of_validations_images:    num_of_validations_images,
            TrainingSessionFields.model_checkpoints_path:       model_checkpoints_path,
            TrainingSessionFields.tensorboard_logs_path:        tensorboard_logs_path,
            TrainingSessionFields.batch_size:                   batch_size,
            TrainingSessionFields.num_of_epochs_processed:      num_of_epochs_processed,
        })

        last_training_session_row_id = cursor.lastrowid

        print(f'Inserted training session with last row id: {last_training_session_row_id}')

        training_session = self.get_training_session(
            model                       = model,
            optimizer                   = optimizer,
            training_dataset_type       = training_dataset_type,
            unfreeze_top_k_layers       = unfreeze_top_k_layers,
            use_multitarget_learning    = use_multitarget_learning
        )

        if not training_session:
            raise RuntimeError(f'Expected to find training session that was just inserted, but instead found nothing.')

        return training_session

    def update_training_session(
        self,
        *,
        id: int,
        model_checkpoints_path: str = None,
        tensorboard_logs_path: str = None,
        num_of_epochs_processed: int = None,
        num_of_examples_processed: int = None
    ):

        update_sql = f'''
            {
                f' {TrainingSessionFields.num_of_epochs_processed} = :{TrainingSessionFields.num_of_epochs_processed},'
                if num_of_epochs_processed is not None
                else ''
            }
            {
                f' {TrainingSessionFields.model_checkpoints_path} = :{TrainingSessionFields.model_checkpoints_path},'
                if model_checkpoints_path is not None
                else ''
            }

            {
                f' {TrainingSessionFields.tensorboard_logs_path} = :{TrainingSessionFields.tensorboard_logs_path},'
                if tensorboard_logs_path is not None
                else ''
            }

            {
                f' {TrainingSessionFields.num_of_examples_processed} = :{TrainingSessionFields.num_of_examples_processed},'
                if num_of_examples_processed is not None
                else ''
            }
        '''

        bound_args = {
            TrainingSessionFields.id: id,
        }

        if tensorboard_logs_path:
            bound_args[TrainingSessionFields.tensorboard_logs_path] = tensorboard_logs_path

        if num_of_examples_processed:
            bound_args[TrainingSessionFields.num_of_examples_processed] = num_of_examples_processed

        if model_checkpoints_path:
            bound_args[TrainingSessionFields.model_checkpoints_path] = model_checkpoints_path

        if num_of_epochs_processed:
            bound_args[TrainingSessionFields.num_of_epochs_processed] = num_of_epochs_processed

        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.execute(f'''
            UPDATE {TrainingUtilsTableNames.TRAINING_SESSION}
                SET {update_sql.strip().strip(',')}
            WHERE {TrainingSessionFields.id} = :{TrainingSessionFields.id}
        ''', bound_args)

        print(f'Updated training session with id: {id}')

        updated_training_session = self.get_training_session_by_id(id)

        if not updated_training_session:
            raise RuntimeError(f'''update_training_session_num_of_epochs:
                Expected to find training session that was just updated, but instead found nothing.
            ''')

        return updated_training_session

    def update_training_run_config(
        self,
        *,
        id: int,
        last_checkpoint_path: str = None
    ):

        update_sql = f'''
            {
                f' {TrainingRunConfigFields.last_checkpoint_path} = :{TrainingRunConfigFields.last_checkpoint_path},'
                if last_checkpoint_path is not None
                else ''
            }
        '''

        bound_args = {
            TrainingRunConfigFields.id: id,
        }

        if last_checkpoint_path:
            bound_args[TrainingRunConfigFields.last_checkpoint_path] = last_checkpoint_path

        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.execute(f'''
            UPDATE {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION}
                SET {update_sql.strip().strip(',')}
            WHERE {TrainingRunConfigFields.id} = :{TrainingRunConfigFields.id}
        ''', bound_args)

        print(f'Updated training run config with id: {id}')

    def increment_training_session_num_of_epochs(
        self,
        id: int,
        increment_by_value: int = 1
    ):
        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.execute(f'''
            UPDATE {TrainingUtilsTableNames.TRAINING_SESSION}
                SET {TrainingSessionFields.num_of_epochs_processed} =
                    {TrainingSessionFields.num_of_epochs_processed} + :increment_by_value
            WHERE {TrainingSessionFields.id} = :{TrainingSessionFields.id}
        ''', {
            TrainingSessionFields.id:   id,
            'increment_by_value':       increment_by_value
        })

        training_session = self.get_training_session_by_id(id)

        if not training_session:
            raise RuntimeError(f'''increment_training_session_num_of_epochs:
                Expected to find training session that was just updated, but instead found nothing.
            ''')

        return training_session

    def save_training_epoch_metrics(
        self,
        *,
        training_session_id: int,
        training_run_config_id: int,
        metrics_data: list
    ):
        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.executemany(f'''
            INSERT INTO {TrainingUtilsTableNames.TRAINING_SESSION_METRICS} (
                {TrainingSessionMetricsFields.training_session_id},
                {TrainingSessionMetricsFields.training_run_configuration_id},
                {TrainingSessionMetricsFields.metric_type},
                {TrainingSessionMetricsFields.value}
            )
            VALUES (
                :{TrainingSessionMetricsFields.training_session_id},
                :{TrainingSessionMetricsFields.training_run_configuration_id},
                :{TrainingSessionMetricsFields.metric_type},
                :{TrainingSessionMetricsFields.value}
            );
        ''', list(map(lambda metric_data: {
            TrainingSessionMetricsFields.training_session_id: training_session_id,
            TrainingSessionMetricsFields.metric_type: metric_data[TrainingSessionMetricsFields.metric_type],
            TrainingSessionMetricsFields.value: metric_data[TrainingSessionMetricsFields.value],
            TrainingSessionMetricsFields.training_run_configuration_id: training_run_config_id,
        }, metrics_data)))

        print(f'Inserted number of metrics: {cursor.rowcount}')

    def save_training_run_configuration(
        self,
        *,
        training_session_id: int,
        save_checkpoint_every_n_minutes: int,
        job_id: int,
        optimizer_lr: float,
        generator_max_queue_size: int,
        continue_training_allowed_different_config_keys: str,
        continue_from_last_checkpoint: int,
        last_checkpoint_path: str,
        tensorboard_monitor_freq: int,
        copy_db_to_scratch: int,
        use_multiprocessing: int,
        validation_data_use_percentage = float,
        workers: int,
        use_gpu: int
    ) -> TrainingRunConfigFields:
        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.execute(f'''
            INSERT INTO {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION} (
                {TrainingRunConfigFields.training_session_id},
                {TrainingRunConfigFields.save_checkpoint_every_n_minutes},
                {TrainingRunConfigFields.job_id},
                {TrainingRunConfigFields.optimizer_lr},
                {TrainingRunConfigFields.generator_max_queue_size},
                {TrainingRunConfigFields.continue_training_allowed_different_config_keys},
                {TrainingRunConfigFields.continue_from_last_checkpoint},
                {TrainingRunConfigFields.validation_data_use_percentage},
                {TrainingRunConfigFields.last_checkpoint_path},
                {TrainingRunConfigFields.tensorboard_monitor_freq},
                {TrainingRunConfigFields.copy_db_to_scratch},
                {TrainingRunConfigFields.use_multiprocessing},
                {TrainingRunConfigFields.workers},
                {TrainingRunConfigFields.use_gpu}
            ) VALUES (
                :{TrainingRunConfigFields.training_session_id},
                :{TrainingRunConfigFields.save_checkpoint_every_n_minutes},
                :{TrainingRunConfigFields.job_id},
                :{TrainingRunConfigFields.optimizer_lr},
                :{TrainingRunConfigFields.generator_max_queue_size},
                :{TrainingRunConfigFields.continue_training_allowed_different_config_keys},
                :{TrainingRunConfigFields.continue_from_last_checkpoint},
                :{TrainingRunConfigFields.validation_data_use_percentage},
                :{TrainingRunConfigFields.last_checkpoint_path},
                :{TrainingRunConfigFields.tensorboard_monitor_freq},
                :{TrainingRunConfigFields.copy_db_to_scratch},
                :{TrainingRunConfigFields.use_multiprocessing},
                :{TrainingRunConfigFields.workers},
                :{TrainingRunConfigFields.use_gpu}
            );
        ''', {
            TrainingRunConfigFields.training_session_id:                             training_session_id,
            TrainingRunConfigFields.save_checkpoint_every_n_minutes:                 save_checkpoint_every_n_minutes,
            TrainingRunConfigFields.job_id:                                          job_id,
            TrainingRunConfigFields.optimizer_lr:                                    optimizer_lr,
            TrainingRunConfigFields.generator_max_queue_size:                        generator_max_queue_size,
            TrainingRunConfigFields.continue_training_allowed_different_config_keys: continue_training_allowed_different_config_keys,
            TrainingRunConfigFields.continue_from_last_checkpoint:                   continue_from_last_checkpoint,
            TrainingRunConfigFields.validation_data_use_percentage:                  validation_data_use_percentage,
            TrainingRunConfigFields.last_checkpoint_path:                            last_checkpoint_path,
            TrainingRunConfigFields.tensorboard_monitor_freq:                        tensorboard_monitor_freq,
            TrainingRunConfigFields.copy_db_to_scratch:                              copy_db_to_scratch,
            TrainingRunConfigFields.use_multiprocessing:                             use_multiprocessing,
            TrainingRunConfigFields.workers:                                         workers,
            TrainingRunConfigFields.use_gpu:                                         use_gpu
        })

        return self.get_last_training_run_configration(training_session_id=training_session_id)

    def get_last_training_run_configration(
        self,
        *,
        training_session_id: int
    ) -> TrainingRunConfigFields:
        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.execute(f'''
            SELECT {', '.join(TrainingRunConfigFields._fields)}
            FROM {TrainingUtilsTableNames.TRAINING_RUN_CONFIGURATION}
            WHERE {TrainingRunConfigFields.training_session_id} = :{TrainingRunConfigFields.training_session_id}
            ORDER BY {TrainingRunConfigFields.id} DESC
            LIMIT 1;
        ''', {
            TrainingRunConfigFields.training_session_id: training_session_id,
        })

        sql_result = cursor.fetchone()

        if sql_result is None:
            print(f'No training run configuration for training session with id {training_session_id}')
            return None

        last_training_run_configuration = TrainingRunConfigConstructor(*sql_result)

        return last_training_run_configuration

    def save_training_metrics(
        self,
        *,
        training_session_id: int,
        training_run_config_id: int,
        metrics_to_save: list,
        checkpoints_metadata_id: int
    ):
        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        def map_metric_to_args(metric):
            return {
                TrainingSessionMetricsFields.training_session_id: training_session_id,
                TrainingSessionMetricsFields.training_run_configuration_id: training_run_config_id,
                TrainingSessionMetricsFields.checkpoints_metadata_id: checkpoints_metadata_id,
                TrainingSessionMetricsFields.metric_type: metric[TrainingSessionMetricsFields.metric_type],
                TrainingSessionMetricsFields.value: metric[TrainingSessionMetricsFields.value],
            }

        cursor.executemany(f'''
            INSERT INTO {TrainingUtilsTableNames.TRAINING_SESSION_METRICS} (
                {TrainingSessionMetricsFields.training_session_id},
                {TrainingSessionMetricsFields.training_run_configuration_id},
                {TrainingSessionMetricsFields.checkpoints_metadata_id},
                {TrainingSessionMetricsFields.metric_type},
                {TrainingSessionMetricsFields.value}
            ) VALUES (
                :{TrainingSessionMetricsFields.training_session_id},
                :{TrainingSessionMetricsFields.training_run_configuration_id},
                :{TrainingSessionMetricsFields.checkpoints_metadata_id},
                :{TrainingSessionMetricsFields.metric_type},
                :{TrainingSessionMetricsFields.value}
            );
        ''', map(map_metric_to_args, metrics_to_save))

        print(f'''
            Inserted {cursor.rowcount} metrics for training session with id {training_session_id}.
        ''')

    def save_checkpoints_metadata(
        self,
        *,
        training_run_config_id: int,
        checkpoint_path: str
    ):
        db_conn = self._get_db_conn()
        cursor = db_conn.cursor()

        cursor.execute(f'''
            INSERT INTO {TrainingUtilsTableNames.CHECKPOINTS_METADATA} (
                {CheckpointsMetadataFields.training_run_config_id},
                {CheckpointsMetadataFields.checkpoint_path}
            )
            VALUES (
                :{CheckpointsMetadataFields.training_run_config_id},
                :{CheckpointsMetadataFields.checkpoint_path}
            );
        ''', {
            CheckpointsMetadataFields.training_run_config_id: training_run_config_id,
            CheckpointsMetadataFields.checkpoint_path: checkpoint_path,
        })

        checkpoint_metadata_id = cursor.lastrowid

        print(f'''
            Created checkpoints metadata with id: {checkpoint_metadata_id}
        ''')

        return checkpoint_metadata_id
