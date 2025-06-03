#!/bin/bash
#SBATCH --job-name=draft_bot_sae_training   # Job name
#SBATCH --output=/mnt/polished-lake/home/ian/logs/draft_bot_sae_training_%j.log       # Standard output log
#SBATCH --error=/mnt/polished-lake/home/ian/logs/draft_bot_sae_training_%j.err       # Standard error log
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (one per node)
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --gres=gpu:1                # GPUs per node

cd /mnt/polished-lake/home/ian/projects/draft_bot/src

# Get absolute path for conda environment
export PATH="/opt/conda/condabin:$PATH"
CONDA_ENV="env"
CONDA_ENV_PATH="${PWD}/${CONDA_ENV}"
 
# Print configuration for verification
echo "Job Configuration:"
echo "Number of Nodes: ${SLURM_NNODES}"
echo "GPUs per Node: ${SLURM_GPUS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Conda Environment Path: ${CONDA_ENV_PATH}"
echo "Master Node: ${SLURMD_NODENAME}"
echo "Node List: ${SLURM_NODELIST}"
 
# Initialize conda
eval "$(conda shell.bash hook)"
 
# Create conda environment if it doesn't exist
if [ ! -d "${CONDA_ENV_PATH}" ]; then
    echo "Creating new conda environment at: ${CONDA_ENV_PATH}"
    conda create --prefix "${CONDA_ENV_PATH}" python=3.9 pytorch torchvision cudatoolkit -c pytorch -y || {
        echo "Error: Failed to create conda environment"
        exit 1
    }
fi
 
# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV_PATH}"
conda activate "${CONDA_ENV_PATH}" || {
    echo "Error: Failed to activate conda environment at '${CONDA_ENV_PATH}'"
    exit 1
}
 
# Verify conda environment activation
if [[ "${CONDA_PREFIX}" != "${CONDA_ENV_PATH}" ]]; then
    echo "Error: Conda environment activation failed"
    echo "Expected: ${CONDA_ENV_PATH}"
    echo "Got: ${CONDA_PREFIX}"
    exit 1
fi
 
# Print Python and PyTorch versions for verification
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
 

python3 train_sae_for_draft_nnet.py \
   -o /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/models/draft_nets/draft_bot_nnet_2025-05-17\ 17_20_05.432717/saes \
   -d /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/processed \
   -s BRO \
   --draft-net-file /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/models/draft_nets/draft_bot_nnet_2025-05-17\ 17_20_05.432717/draft_bot_nnet_2025-05-17\ 17_20_05.432717.pt \
   --layer-idx 2