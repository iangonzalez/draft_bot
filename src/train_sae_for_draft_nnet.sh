#!/bin/bash
#SBATCH --job-name=draft_bot_sae_training   # Job name
#SBATCH --output=/mnt/polished-lake/home/ian/logs/draft_bot_sae_training_%j.log       # Standard output log
#SBATCH --error=/mnt/polished-lake/home/ian/logs/draft_bot_sae_training_%j.err       # Standard error log
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (one per node)
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --gres=gpu:1                # GPUs per node

# Change to the project root directory.
cd /mnt/polished-lake/home/ian/projects/draft_bot/src

# Source the common preamble.
source sbatch_job_preamble.sh

python3 train_sae_for_draft_nnet.py \
   -o /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/models/draft_nets/draft_bot_nnet_2025-05-17\ 17_20_05.432717/saes \
   -d /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/processed \
   -s BRO \
   --draft-net-file /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/models/draft_nets/draft_bot_nnet_2025-05-17\ 17_20_05.432717/draft_bot_nnet_2025-05-17\ 17_20_05.432717.pt \
   --layer-idx 3