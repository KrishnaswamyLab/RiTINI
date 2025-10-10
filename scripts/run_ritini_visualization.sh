#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --time 6:00:00
#SBATCH --job-name ritini-viz
#SBATCH --output logs/ritini-natalia-%J.log

cd /home/jcr222/workspace/RiTINI

# Load uv (adjust based on your cluster setup)
module load uv  # or use: export PATH="$HOME/.cargo/bin:$PATH"

source .venv/bin/activate

python3 ritini/visualizations/visualize_natalia_ritini.py 