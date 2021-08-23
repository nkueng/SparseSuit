#!/bin/bash
#SBATCH --job-name=baseline
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=6000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:titanx:2
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=03:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program

echo $CUDA_VISIBLE_DEVICES
python training.py
