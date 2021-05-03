#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=64g  # Memory
#SBATCH --cpus-per-task=12  # number of cpus to use - there are 32 on each node.

echo "============"
echo "Initialize Env ========"

set -e # fail fast

export CURRENT_TIME=$(date "+%Y_%m_%d_%H%M%S")

# Activate Conda
source /home/${USER}/miniconda3/bin/activate allennlp

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y__%H_%M')
echo ${dt}

# Env variables
export STUDENT_ID=${USER}

# General training parameters
export CLUSTER_HOME="/home/${STUDENT_ID}"
export DATASET_SOURCE="${CLUSTER_HOME}/datasets/story_datasets/"

declare -a ScratchPathArray=(/disk/scratch_big/ /disk/scratch1/ /disk/scratch2/ /disk/scratch/ /disk/scratch_fast/)

# Iterate the string array using for loop
for i in "${ScratchPathArray[@]}"; do
  echo ${i}
  if [ -d ${i} ]; then
    export SCRATCH_HOME="${i}/${STUDENT_ID}"
    mkdir -p ${SCRATCH_HOME}
    if [ -w ${SCRATCH_HOME} ]; then
      break
    fi
  fi
done

echo ${SCRATCH_HOME}

export EXP_ROOT="${CLUSTER_HOME}/git/story-fragments"
export EXP_ID="${EXP_NAME}_${SLURM_JOB_ID}_${CURRENT_TIME}"
export SERIAL_DIR="${SCRATCH_HOME}/${EXP_ID}/"


# Ensure the scratch home exists and CD to the experiment root level.
cd "${EXP_ROOT}" # helps AllenNLP behave

cd /home/s1569885/git/story-fragments/story_fragments/data_processing 

python aligned_salience_processing.py events \
--src-json ${BATCH_FILE_PATH} \
--output-dir ${SERIAL_DIR} --match-type whole \
--more-k-diff-similarity 0.05 --plus-minus-percentile 7.25 --min-threshold 0.30 --earliest-k 3 --nearest-k 5

echo "============"
echo "Salience Extraction Task finished"

export HEAD_EXP_DIR="${CLUSTER_HOME}/runs/${EXP_ID}"
mkdir -p "${HEAD_EXP_DIR}"
rsync -avuzhP "${SERIAL_DIR}/" "${HEAD_EXP_DIR}/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}"

if [ ! -v COPY_DATASET ]; then
  echo "No dataset to delete"
else
  rm -rf ${DATASET_ROOT}
fi

echo "============"
echo "results synced"


