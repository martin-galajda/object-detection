import os
from checkpoints.utils import \
    is_checkpoint_from_training_progress,\
    is_final_checkpoint,\
    get_timestamp_and_minutes_from_progress_chk,\
    get_timestamp_from_final_chk
from typing import Callable


def get_checkpoints_from_dir(chk_dir: str, *, filter_fn: Callable = None) -> (dict, dict):
    empty_res = ({}, {})
    if not os.path.isdir(chk_dir):
        return empty_res

    dir_files = os.listdir(chk_dir)

    if len(dir_files) == 0:
        return empty_res

    dir_files_by_date = {}
    final_models_by_date = {}

    print(dir_files)
    for dir_filename in dir_files:
        if filter_fn is not None and not filter_fn(dir_filename):
            continue

        if is_final_checkpoint(dir_filename):
            timestamp = get_timestamp_from_final_chk(dir_filename)
            final_models_by_date[timestamp] = dir_filename
            continue

        if is_checkpoint_from_training_progress(dir_filename):
            timestamp, mins = get_timestamp_and_minutes_from_progress_chk(dir_filename)

            if timestamp not in dir_files_by_date:
                dir_files_by_date[timestamp] = {}
            dir_files_by_date[timestamp][int(mins)] = dir_filename

    return dir_files_by_date, final_models_by_date


def get_last_checkpoint_from_dir(chk_dir: str):
    dir_files_by_date, final_models_by_date = get_checkpoints_from_dir(chk_dir)
    return _get_last_checkpoint(dir_files_by_date, final_models_by_date)


def get_checkpoint_for_retraining(checkpoint_dir: str):
    # def filter_checkpoints_fn(chkpoint_filename):
    #     if not is_final_checkpoint(chkpoint_filename) and not is_checkpoint_from_training_progress(chkpoint_filename):
    #         return False
    #     job_id = get_job_id_from_chk_name(chkpoint_filename)

    #     if not job_id:
    #         return True

    #     config_filename = make_config_name(job_id)
    #     config_file_path = os.path.join(checkpoint_dir, config_filename)
    #     config = load_training_config(config_file_path)
    #     return matching_training_configs(args, config)

    os.makedirs(checkpoint_dir, exist_ok=True)
    dir_files_by_date, final_models_by_date = get_checkpoints_from_dir(checkpoint_dir, filter_fn=None)
    return _get_last_checkpoint(dir_files_by_date, final_models_by_date)


def _get_last_checkpoint(dir_files_by_date: dict, final_models_by_date: dict):
    print(dir_files_by_date)
    sorted_dates = sorted(dir_files_by_date.keys())

    if len(sorted_dates) == 0:
        sorted_dates_final_models = sorted(final_models_by_date.keys())
        if len(sorted_dates_final_models) > 0:
            return final_models_by_date[sorted_dates_final_models[-1]]
        return None

    last_timestamp = sorted_dates[-1]

    checkpoints_for_last_date_dict = dir_files_by_date[last_timestamp]
    last_minute_checkpoint = sorted(checkpoints_for_last_date_dict.keys())[-1]
    last_checkpoint_filename = checkpoints_for_last_date_dict[last_minute_checkpoint]

    if last_timestamp in final_models_by_date:
        last_checkpoint_filename = final_models_by_date[last_timestamp]

    return last_checkpoint_filename
