import os
from .get_last_checkpoint_from_dir import get_last_checkpoint_from_dir

def remove_old_checkpoints(dir):

  if not os.path.isdir(dir):
    print("%s is not valid path to directory." % (dir,))

  dir_files = os.listdir(dir)
  last_checkpoint_file = get_last_checkpoint_from_dir(dir)
  for dir_file in dir_files:
    if dir_file == last_checkpoint_file:
      continue
    path_to_file = os.path.join(dir, dir_file)
    os.remove(path_to_file)

  
