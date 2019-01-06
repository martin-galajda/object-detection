from keras.callbacks import Callback
from datetime import datetime
import os
from time import time as timer
import re

timestamp_of_training = re.sub(' ','_', str(datetime.utcnow()))

class ModelSaver(Callback):
    def __init__(self, after_every_batch = None, every_n_minutes = None, model = 'model'):
        self.save_every_batch = after_every_batch
        self.every_n_minutes = every_n_minutes
        self.start = timer()
        self.batch = 0
        self.last_minute_saved = 0
        self.directory = model
        os.makedirs(self.directory, exist_ok=True)

        if self.save_every_batch:
          print("ModelSaver.__init()__ ---> going to save model every %d batch processed." % self.save_every_batch)
        elif self.every_n_minutes:
          print("ModelSaver.__init()__ ---> going to save model every %d minutes." % self.every_n_minutes)


    def save_model(self, suffix):
      if self.save_every_batch:
        print("Saving model after %d batches." % self.batch)
      elif self.every_n_minutes:
        print("Saving model after %s." % suffix)

      path_to_check_points = os.path.join(os.path.dirname(__file__), '..', self.directory)

      if not os.path.isdir(path_to_check_points):
        os.makedirs(path_to_check_points)

      name = '%s_model_batch_%s.hdf5' % (timestamp_of_training, suffix)
      model_checkpoint_path = os.path.join(path_to_check_points, name)
      self.model.save(model_checkpoint_path)

    def on_batch_end(self, batch, logs={}):
        if self.save_every_batch:
          if self.batch > 0 and ((self.batch % self.save_every_batch) == 0):
              self.save_model(str(self.batch))
          self.batch += 1

        if self.every_n_minutes:
          seconds_diff = timer() - self.start
          minutes_diff = seconds_diff // 60

          print(minutes_diff)

          if minutes_diff > 0 and ((minutes_diff - self.last_minute_saved) >= self.every_n_minutes):
            self.save_model(str(int(minutes_diff)) + "_minutes")
            self.last_minute_saved = minutes_diff
