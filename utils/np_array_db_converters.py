import numpy as np
import io
import sqlite3

def adapt_array(arr):
  """
  http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
  """

  out = io.BytesIO()
  out.seek(0)
  np.savez_compressed(out, np.array(arr))
  out.seek(0)

  bytes_compressed = out.read()

  return sqlite3.Binary(bytes_compressed)


def convert_array(text):
  out = io.BytesIO(text)
  out.seek(0)

  return np.load(out)['arr_0']
