wd = "/home/hoellinger/bias_challenge/bias-model-test-bench/investigate_borg_bias_models/"
data_dir = "/data73/hoellinger/bias_challenge/"
train_dir = data_dir+"Training/"
val_dir = data_dir+"Validation/"

train = train_dir+"data.hdf5"
val = val_dir+"data.hdf5"

av_seeds = ['aquila_bias_challenge_training_0',
            'aquila_bias_challenge_training_1',
            'aquila_bias_challenge_training_2',
            'aquila_bias_challenge_training_3',
            'aquila_bias_challenge_training_4',
            'aquila_bias_challenge_training_5',
            'aquila_bias_challenge_training_6',
            'aquila_bias_challenge_training_7',
            'aquila_bias_challenge_training_8',
            'aquila_bias_challenge_training_9',
            'aquila_bias_challenge_training_10']

bins_keys = ['counts_bin_0', 'counts_bin_1', 'counts_bin_2', 'counts_bin_3', 'counts_bin_4', 'counts_bin_5']

HEADER="#!/bin/bash\n#PBS -S /bin/sh\n#PBS -N python\n#PBS -j oe\n#PBS -l nodes=1:ppn=128,walltime=01:00:00\n"
