#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=isela1_UTR
#SBATCH --time=0-12:00:00 # request days-hours:minutes:seconds
#SBATCH --nodes=1 # request 1 node
#SBATCH --ntasks-per-node=1 # request 1 task (command) per node
#SBATCH --cpus-per-task=1 # request 1 cpu (core, thread) per task
#SBATCH --mem=32G # request 2GB total memory per node
#SBATCH --gres=gpu:a100:1 # request 1 GPU
#SBATCH --output=stdout.%x.%j
#SBATCH --error=stderr.%x.%j

# load software modules
ml purge
ml GCCcore/11.3.0 Python/3.10.4
ml CUDA/11.7.0
ml WebProxy

# run commands
source /scratch/user/isela1/myenv_gpu/bin/activate
python3 npz_to_csv.py
python3 npz_to_csv_train.py
deactivate
