import os
from random import randint

if 'JOBID' in os.environ:
  JOB_ID = os.environ['JOBID']
else:
  JOB_ID = str(randint(100000, 999999))

def get_job_id():
  return JOB_ID
