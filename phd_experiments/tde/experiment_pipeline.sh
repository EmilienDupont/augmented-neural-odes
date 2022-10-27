#!/usr/bin/env bash
set -e
set -x
WORKDIR="/home/mbaddar/phd/augmented-neural-odes"

export PYTHONPATH=$PYTHONPATH:$WORKDIR
#
for i in 1 2 3 4 5 6 7 8 9 10
do
   echo "======================= Script run # ${i} ======================="
   python3 TensorODE_experiment_pipeline.py --config=tde_experiments_config.yaml
done