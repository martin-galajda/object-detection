import os
from timeit import default_timer as timer

def progress_percentage(perc, width=None):
  # This will only work for python 3.3+ due to use of
  # os.get_terminal_size the print function etc.

  FULL_BLOCK = '#'
  # this is a gradient of incompleteness
  INCOMPLETE_BLOCK_GRAD = ['-', '▒', '▓']

  assert (isinstance(perc, float))
  assert (0. <= perc <= 100.)

  # if width unset use full terminal
  if width is None:
    width = os.get_terminal_size().columns

  # progress bar is block_widget separator perc_widget : ####### 30%
  max_perc_widget = '[100.00%]'  # 100% is max
  separator = ' '
  blocks_widget_width = width - len(separator) - len(max_perc_widget)
  assert (blocks_widget_width >= 10)  # not very meaningful if not

  perc_per_block = 100.0 / blocks_widget_width
  # epsilon is the sensitivity of rendering a gradient block
  epsilon = 1e-6

  # number of blocks that should be represented as complete
  full_blocks = int((perc + epsilon) / perc_per_block)

  # the rest are "incomplete"
  empty_blocks = blocks_widget_width - full_blocks

  # build blocks widget
  blocks_widget = ([FULL_BLOCK] * full_blocks)
  blocks_widget.extend([INCOMPLETE_BLOCK_GRAD[0]] * empty_blocks)

  # build perc widget
  str_perc = '%.2f' % perc
  # -1 because the percentage sign is not included
  perc_widget = '[%s%%]' % str_perc.ljust(len(max_perc_widget) - 3)

  # form progressbar
  progress_bar = '%s%s%s' % (''.join(blocks_widget), separator, perc_widget)

  # return progressbar as string
  return ''.join(progress_bar)


def copy_progress(copied, total, print_progress):
  if print_progress:
    print('\r' + progress_percentage(100 * copied / total, width=30), end='')


def copyfile(src, dst, print_progress):
  """Copy data from src to dst.

  If follow_symlinks is not set and src is a symbolic link, a new
  symlink will be created instead of copying the file it points to.

  """
  size = os.stat(src).st_size
  with open(src, 'rb') as fsrc:
    with open(dst, 'wb') as fdst:
      copyfileobj(fsrc, fdst, callback=lambda copied, total: copy_progress(copied,  total, print_progress), total=size)
  return dst


def copyfileobj(fsrc, fdst, callback, total, length=16 * 1024):
  copied = 0
  while True:
    buf = fsrc.read(length)
    if not buf:
      break
    fdst.write(buf)
    copied += len(buf)
    callback(copied, total=total)

def copy_file_with_progress_bar(src, dest, print_progress):
  copyfile(src, dest, print_progress)


def copy_file_to_scratch(src_file_path, print_progress = False):
  if not 'SCRATCH' in os.environ:
    raise RuntimeError(f'Env variable SCRATCH variable not set and calling copy_file_to_scratch().')

  scratch_dir = os.environ['SCRATCH'] + '/'
  file_name = os.path.split(src_file_path)[-1]
  scratch_path = scratch_dir + file_name

  if not os.path.exists(scratch_path):
    start = timer()
    print("")
    print("-- copying %s to %s" % (src_file_path, scratch_path))

    copy_file_with_progress_bar(src_file_path, scratch_path, print_progress)

    end = timer()
    print(f'\n\nTook {end - start} seconds to copy file to scratch')
  else:
    print(f'-- not copying file to {scratch_path} as it already exists.')

  return scratch_path
