#!/bin/bash
#PBS -S /bin/sh
#PBS -N python
#PBS -j oe
#PBS -l nodes=1:ppn=128,walltime=01:00:00

module load "intelpython/3-2023.0.0"
source activate borg
python "/home/hoellinger/bias_challenge/bias-model-test-bench/investigate_borg_bias_models/perform_opt.py" --bias_model bias::BrokenPowerLaw --id v0.1 --nkeys 2 --nfields 2
exit 0
