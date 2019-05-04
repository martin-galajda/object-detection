import os
import argparse

from training_utils.db import TrainingUtilsDB
from training_utils.db_tables_field_names import TrainingSessionFieldsConstructor, TrainingSessionFields
from checkpoints.utils import make_checkpoint_model_dir, make_tensorboard_train_sess_dir
from utils.get_job_id import get_job_id


def parse_training_session_fields_from_args(args: argparse.Namespace, training_dataset_type: str) \
    -> TrainingSessionFields:
    training_session_fields = TrainingSessionFieldsConstructor(
        id=None,
        created_at=None,
        updated_at=None,
        num_of_examples_processed=None,
        model = args.model.strip(),
        training_dataset_type= training_dataset_type,
        unfreeze_top_k_layers= str(args.unfreeze_top_k_layers).strip(),
        use_multitarget_learning= args.use_multitarget_learning,
        optimizer= args.optimizer.strip(),
        num_of_training_images= args.images_num,
        num_of_validations_images= args.validation_images_num,
        model_checkpoints_path= None,
        tensorboard_logs_path= None,
        batch_size= args.batch_size,
        num_of_epochs_processed= 0
    )

    return training_session_fields


def load_training_session(args: argparse.Namespace, training_dataset_type: str) -> TrainingSessionFields:
    training_utils_db = TrainingUtilsDB()

    training_session_fields = parse_training_session_fields_from_args(args, training_dataset_type)

    if args.training_session_id:
        training_session = training_utils_db.get_training_session_by_id(id=int(args.training_session_id))
    else:
        training_session = training_utils_db.get_training_session(
            optimizer                   = training_session_fields.optimizer,
            model                       = training_session_fields.model,
            training_dataset_type       = training_session_fields.training_dataset_type,
            unfreeze_top_k_layers       = training_session_fields.unfreeze_top_k_layers,
            use_multitarget_learning    = training_session_fields.use_multitarget_learning
        )

    if training_session is None:
        training_session = initiate_new_training_session(training_session_fields)

    return training_session

def create_training_run(args: argparse.Namespace, training_session: TrainingSessionFields):
    training_utils_db = TrainingUtilsDB()
    training_run_config = training_utils_db.save_training_run_configuration(
        training_session_id=training_session.id,
        save_checkpoint_every_n_minutes=args.save_checkpoint_every_n_minutes,
        job_id= get_job_id(),
        optimizer_lr= args.optimizer_lr,
        generator_max_queue_size=args.generator_max_queue_size,
        continue_training_allowed_different_config_keys=args.continue_training_allowed_different_config_keys,
        continue_from_last_checkpoint=int(args.continue_from_last_checkpoint),
        last_checkpoint_path=None,
        tensorboard_monitor_freq=args.tensorboard_monitor_freq,
        copy_db_to_scratch=args.copy_db_to_scratch,
        use_multiprocessing=args.use_multiprocessing,
        validation_data_use_percentage=args.validation_data_use_percentage,
        workers=args.workers,
        use_gpu=int(args.use_gpu)
    )

    return training_run_config

def initiate_new_training_session(training_session_fields: TrainingSessionFields):
    training_utils_db = TrainingUtilsDB()
    training_session = training_utils_db.create_training_session(
        model=training_session_fields.model,
        optimizer=training_session_fields.optimizer,
        training_dataset_type=training_session_fields.training_dataset_type,
        unfreeze_top_k_layers=training_session_fields.unfreeze_top_k_layers,
        use_multitarget_learning=training_session_fields.use_multitarget_learning,
        num_of_training_images=training_session_fields.num_of_training_images,
        num_of_validations_images=training_session_fields.num_of_validations_images,
        batch_size=training_session_fields.batch_size,
        num_of_epochs_processed=training_session_fields.num_of_epochs_processed,
        model_checkpoints_path=None,
        tensorboard_logs_path=None
    )

    root_path = os.getcwd()

    model_checkpoints_path = make_checkpoint_model_dir(
        training_session.training_dataset_type,
        training_session.model,
        training_session.id)
    tensorboard_logs_path = make_tensorboard_train_sess_dir(
        training_session.training_dataset_type,
        training_session.model,
        training_session.id)

    model_checkpoints_path = os.path.abspath(f'{root_path}/{model_checkpoints_path}')
    tensorboard_logs_path = os.path.abspath(f'{root_path}/{tensorboard_logs_path}')

    os.makedirs(tensorboard_logs_path, exist_ok=True)
    os.makedirs(model_checkpoints_path, exist_ok=True)

    updated_training_session = training_utils_db.update_training_session(
        id = training_session.id,
        model_checkpoints_path = model_checkpoints_path,
        tensorboard_logs_path = tensorboard_logs_path
    )

    return updated_training_session
