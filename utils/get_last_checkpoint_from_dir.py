import os
import re

REGEX_EXTRACT_DATE_AND_MINUTES = r'(.*)_model_batch_(\d+)_minutes.hdf5$'


def get_last_checkpoint_from_dir(dir):
  if not os.path.isdir(dir):
    return None

  dir_files = os.listdir(dir)

  dir_files_by_date = {}
  dir_files_by_minutes = {}

  print(dir_files)
  for dir_filename in dir_files:
    match = re.match(REGEX_EXTRACT_DATE_AND_MINUTES, dir_filename)

    if match is None:
      print("match none")
      continue
    date, str_minutes = match.groups()

    dir_files_by_minutes[int(str_minutes)] = dir_filename

    if date not in dir_files_by_date:
      dir_files_by_date[date] = {}
    dir_files_by_date[date][int(str_minutes)] = dir_filename

  print(dir_files_by_date)
  sorted_dates = sorted(dir_files_by_date.keys())
  last_date = sorted_dates[-1]

  checkpoints_for_last_date_dict = dir_files_by_date[last_date]
  last_minute_checkpoint = sorted(checkpoints_for_last_date_dict.keys())[-1]
  last_checkpoint_filename = checkpoints_for_last_date_dict[last_minute_checkpoint]

  return last_checkpoint_filename