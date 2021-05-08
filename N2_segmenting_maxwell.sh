#!/bin/bash
# call with: sbatch N2_segmenting_maxwell.sh

#SBATCH --partition=upex,cfel,cfel-cmi
#SBATCH --time=5-00:00:00                                   # request 5 days
#SBATCH --chdir=/home/brombh/timepix/FlashSep20      # directory must already exist!
#SBATCH --mail-type=END,FAIL                                # send mail when the job has finished or failed
#SBATCH --nodes=1                                           # number of nodes
#SBATCH --output=%x-%N-%j.out                               # per default slurm writes output to slurm-<jobid>.out. There are a number of options to customize the job
export LD_PRELOAD=""

source /etc/profile.d/modules.sh

source /home/brombh/miniconda3/etc/profile.d/conda.sh
conda activate timepix

python ./N2_segmenting.py