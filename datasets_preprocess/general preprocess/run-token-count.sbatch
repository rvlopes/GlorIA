#!/bin/bash
#SBATCH --job-name=token-count
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id
#SBATCH --nodelist=nscluster #novasearchdl nscluster
#SBATCH --output=slurmlogs/%x-%j.out
#SBATCH -e slurmlogs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12  # number cpus (threads) per task
#SBATCH --mem=64000 # 16384 Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

eval "$(conda shell.bash hook)"

# activate your anaconda environment
conda activate thesis

# change dir to where you want to run scripts
#cd $PWD

# run program
python token-count.py