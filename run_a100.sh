#!/bin/bash

#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH -c 10
#SBATCH -t 0-01:00:00
#SBATCH -p public
#SBATCH -o bert_sst2_a100.%j.out
#SBATCH -e bert_sst2_a100.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yourmail@abc.xyz
#SBATCH --export=NONE

# Source bashrc for environment
source ~/.bashrc

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Load required modules
module load mamba/latest

# Set working directory
cd <path>

# Display GPU info
nvidia-smi

# Run the benchmark script using transformer-BERT environment
mamba run -n transformer-BERT \
    python bert_sst2_a100.py 2>&1 | tee bert_sst2_a100_benchmark.log


