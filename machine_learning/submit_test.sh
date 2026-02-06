#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=yml_test
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --account=ai4pex
#SBATCH --qos=debug

export PATH="/home/users/twilder/Python/AI4PEX/tensorflow_env/bin:$PATH"

python yml_test.py --config ./config_cnn.yml --dropout_rate 0.1 --epochs 10