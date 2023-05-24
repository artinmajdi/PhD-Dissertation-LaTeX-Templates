#!/bin/bash

#SBATCH --job-name=taxonomy
#SBATCH --time=01:00:00
#SBATCH --account=jjrodrig
#SBATCH --partition=standard

#SBATCH --nodes=2
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:1

#SBATCH -o %thresh_technique.out

# Let's run a few python tests

source /home/u29/mohammadsmajdi/.bashrc
conda activate main 


python aim1_1_2_label_conditional_taxonomy.py

