#!/bin/bash
#SBATCH -J lq
#SBATCH -p gpu4
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --error=%J.err   
#SBATCH --output=%J.out


# eurlex4k

python main_MLGN.py


