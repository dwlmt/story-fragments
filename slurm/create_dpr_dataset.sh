#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=64g  # Memory
#SBATCH --cpus-per-task=12  # number of cpus to use - there are 32 on each node.

# Set EXP_BASE_NAME and BATCH_FILE_PATH

echo "============"
echo "Initialize Env ========"

set -e # fail fast

export CURRENT_TIME=$(date "+%Y_%m_%d_%H%M%S")

# Activate Conda
source /home/${USER}/miniconda3/bin/activate allennlp


cd /home/s1569885/git/story-fragments/story_fragments/data_processing 

python create_dpr_dataset.py create --base-output-dir /home/s1569885/datasets/wikiplots_dpr/ \
--dataset-name wikiplots_20200701_dpr_window_4_step_2_exact \
--datasets /home/s1569885/datasets/wikiplots_dpr/wikiplots_20200701.jsonl \
--index-name exact --index-worlds 512 --window-size 4 --window-step 2 --skip-splitting False


