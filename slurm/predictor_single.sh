#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=32g  # Memory
#SBATCH --cpus-per-task=8  # number of cpus to use - there are 32 on each node.

# Set EXP_BASE_NAME and BATCH_FILE_PATH

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

export ALLENNLP_CACHE_ROOT="${SCRATCH_HOME}/allennlp_cache/"
export HF_DATASETS_CACHE="${SCRATCH_HOME}/huggingface_cache/"
export TMPDIR=${HOME}/tmp/
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

export EXP_ID="${EXP_NAME}_${SLURM_JOB_ID}_${CURRENT_TIME}"
export SERIAL_DIR="${SCRATCH_HOME}/${EXP_ID}"

export PREDICTION_STORY_FILE="${CLUSTER_HOME}/${BATCH_FILE_PATH}"

export MODEL_ZIP=${CLUSTER_HOME}/${MODEL_PATH}

# Ensure the scratch home exists and CD to the experiment root level.
cd "${EXP_ROOT}" # helps AllenNLP behave

mkdir -p ${SERIAL_DIR}

echo "============"
echo "ALLENNLP Task========"

echo "${OVERRIDES_JSON}"
if [ -z "${OVERRIDES_JSON}" ];
then

  allennlp predict --include-package story_fragments --predictor ${PREDICTOR} \
  ${MODEL_ZIP} \
  ${PREDICTION_STORY_FILE} --cuda-device 0 \
  --batch-size 1 \
  --output-file ${SERIAL_DIR}/${EXP_ID}_prediction_output.jsonl;

else
  allennlp predict --include-package story_fragments --predictor ${PREDICTOR} \
  ${MODEL_ZIP} \
  ${PREDICTION_STORY_FILE} --cuda-device 0 \
  --overrides "${OVERRIDES_JSON}" \
  --batch-size 1 \
  --output-file ${SERIAL_DIR}/${EXP_ID}_prediction_output.jsonl; fi


echo "============"
echo "ALLENNLP Task finished"

export HEAD_EXP_DIR="${CLUSTER_HOME}/runs/${EXP_ID}"
mkdir -p "${HEAD_EXP_DIR}"
rsync -avuzhP "${SERIAL_DIR}/" "${HEAD_EXP_DIR}/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}"

echo "============"
echo "results synced"