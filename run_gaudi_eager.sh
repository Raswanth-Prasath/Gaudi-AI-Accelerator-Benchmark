#!/bin/bash

#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -c 10
#SBATCH -t 0-01:00:00
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -o bert_sst2_gaudi_eager.%j.out
#SBATCH -e bert_sst2_gaudi_eager.%j.err
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

# CRITICAL: Set environment variable for HPU Eager Mode
export PT_HPU_LAZY_MODE=0

# Display HPU info
echo "HPU Environment Variables:"
echo "PT_HPU_LAZY_MODE=$PT_HPU_LAZY_MODE"

# Use the Gaudi PyTorch environment's Python directly
GAUDI_ENV=/packages/envs/gaudi-pytorch-diffusion-1.22.0.740
PYTHON=$GAUDI_ENV/bin/python

# Prevent system site-packages from interfering
export PYTHONNOUSERSITE=1
export PYTHONPATH=$GAUDI_ENV/lib/python3.12/site-packages:$PYTHONPATH

# Run the benchmark script
$PYTHON bert_sst2_gaudi_eager.py 2>&1 | tee bert_sst2_gaudi_eager_benchmark.log