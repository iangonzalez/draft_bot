#!/bin/bash
#SBATCH --job-name=draft_bot_autointerp_for_sae   # Job name
#SBATCH --output=/mnt/polished-lake/home/ian/logs/draft_bot_autointerp_for_sae_%j.log       # Standard output log
#SBATCH --error=/mnt/polished-lake/home/ian/logs/draft_bot_autointerp_for_sae_%j.err       # Standard error log
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (one per node)
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --gres=gpu:1                # GPUs per node

# Change to the project root directory.
cd /mnt/polished-lake/home/ian/projects/draft_bot/src

# Source the common preamble.
source sbatch_job_preamble.sh

python3 run_autointerp_for_sae.py \
   --data-dir /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/processed \
   --output-directory /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/models/draft_nets/draft_bot_nnet_2025-05-17\ 17_20_05.432717/autointerp_results/sae_sae_on_draft_bot_nnet_activation3_2025-05-27\ 21_13_09.307453.pt \
   --draft-net-file /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/models/draft_nets/draft_bot_nnet_2025-05-17\ 17_20_05.432717/draft_bot_nnet_2025-05-17\ 17_20_05.432717.pt \
   --sae-file /mnt/polished-lake/home/ian/projects/draft_bot/data/BRO/models/draft_nets/draft_bot_nnet_2025-05-17\ 17_20_05.432717/saes/sae_on_draft_bot_nnet_activation3_2025-05-27\ 21_13_09.307453.pt \
   --set-id BRO \
   --sample-size 10000 \
   --sae-layer-idx 3 \
   --mode AUTOINTERP_EVAL_OUTPUT_BASED_FEATURES

   #--mode AUTOINTERP_ON_OUTPUTS
