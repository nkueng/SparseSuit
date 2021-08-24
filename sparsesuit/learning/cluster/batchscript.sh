#!/bin/bash
#SBATCH --job-name=baseline-lre-3
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=2000M
# Run on 1 gpu, titanx
#SBATCH -p gpu --gres=gpu:titanrtx:1
# time d-h:m:s
#SBATCH --time=07:00:00
#SBATCH --mail-user=nickueng@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

#your script
python training.py hyperparams.initial_learning_rate=0.001
