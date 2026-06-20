#!/bin/bash
#SBATCH --job-name=taa_cfd
#SBATCH --array=0-999%100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

mkdir -p logs
python run_sample.py --sample_id $SLURM_ARRAY_TASK_ID --hpc --seed 42
