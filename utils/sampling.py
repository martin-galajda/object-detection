from numpy.random import randint

def sample_values(*, min_value = 1, max_value, batch_size):
  return randint(min_value, max_value, batch_size)
