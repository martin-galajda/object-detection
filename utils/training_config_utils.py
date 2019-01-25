import argparse
import json
import os

def save_training_config(args: argparse.Namespace, dir: str, job_id: str):
  config = vars(args)
  path_to_filename = os.path.join(dir, make_config_name(job_id))
  with open(path_to_filename, 'w') as f:
    json.dump(config, f, sort_keys=True, indent=4)


def load_training_config(path_to_training_config: str):
  with open(path_to_training_config, 'r') as f:
    config_json = json.load(f)

  return config_json

def matching_training_configs(current_args: argparse.Namespace, old_config: dict):
  config = vars(current_args)

  ignore_keys = list(
    map(lambda x: x.strip(), current_args.continue_training_allowed_different_config_keys.split(',')))

  matching = True
  for key in old_config:
    if key in ignore_keys:
      print(f'Ignoring key "{key}" when comparing whether training configs match.')
      continue

    if config[key] != old_config[key]:
      print(f'Ignoring training config because key "{key}" doesnt match.')
      matching = False

  return matching

def make_config_name(job_id: str):
  return f'{job_id}-config.json'
