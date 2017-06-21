#!/bin/bash

export PYENV_VERSION=3.5.2
# Prepare args for computing
python prepare.py

N_JOBS_MINUS_1=$(($(eval python count_args.py) - 1))
INDICES="0-${N_JOBS_MINUS_1}"

echo "I will launch job with indices: ${INDICES}."

# qsub needs Python 2 on the avakas cluster
export PYENV_VERSION=2.7.12

CMD=$(eval "qsub -t ${INDICES} avakas_job.pbs")
echo $CMD >> job_name.txt
echo $CMD

unset PYENV_VERSION