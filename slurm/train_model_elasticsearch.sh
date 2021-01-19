#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:2
#SBATCH --mem=48g  # Memory
#SBATCH --cpus-per-task=16  # number of cpus to use - there are 32 on each node.

# Set EXP_BASE_NAME and BATCH_FILE_PATH

echo "============"
echo "Initialize Env ========"

set -e # fail fast

export CURRENT_TIME=$(date "+%Y_%m_%d_%H%M%S")

# Activate Conda
source /home/${USER}/miniconda3/bin/activate allennlp

echo "Training ${EXP_NAME} with config ${EXP_CONFIG}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y__%H_%M')
echo ${dt}

# Env variables
export STUDENT_ID=${USER}

export TOKENIZERS_PARALLELISM=false
export MKL_SERVICE_FORCE_INTEL=true

export ES_INSTALL_ROOT="/home/s1569885/elasticsearch-7.10.2/"

# General training parameters
export CLUSTER_HOME="/home/${STUDENT_ID}"

declare -a ScratchPathArray=(/disk/scratch_big/ /disk/scratch1/ /disk/scratch2/ /disk/scratch/ /disk/scratch_fast/)

# Iterate the string array using for loop
for i in "${ScratchPathArray[@]}"; do
  echo ${i}
  if [ -d ${i} ] &&  [ -w ${i} ]; then
    export SCRATCH_HOME="${i}/${STUDENT_ID}"
    mkdir -p ${SCRATCH_HOME}
    break
  fi
done

# Deletes all scratch directories older than a week to cleanup
find ${SCRATCH_HOME} -type d -name "*" -mtime +8 -printf "%T+ %p\n" | sort | cut -d ' ' -f 2- | sed -e 's/^/"/' -e 's/$/"/' | xargs rm -rf

echo ${SCRATCH_HOME}

export EXP_ROOT="${CLUSTER_HOME}/git/story-fragments"

export EXP_ID="${EXP_NAME}_${SLURM_JOB_ID}_${CURRENT_TIME}"
export SERIAL_DIR="${SCRATCH_HOME}/${EXP_ID}"

export ALLENNLP_CACHE_ROOT="${SCRATCH_HOME}/allennlp_cache/"
rm -rf "${SCRATCH_HOME}/allennlp_cache/"

${ES_INSTALL_ROOT}/bin/elasticsearch -Epath.data=/${SCRATCH_HOME}/elasticsearch/data -d -p "${SCRATCH_HOME}/elasticsearch_pid"
sleep 60

# Ensure the scratch home exists and CD to the experiment root level.
cd "${EXP_ROOT}" # helps AllenNLP behave

mkdir -p ${SERIAL_DIR}

echo "============"
echo "ALLENNLP Task========"

allennlp train --file-friendly-logging --include-package story_fragments \
  --serialization-dir ${SERIAL_DIR}/ ${EXP_CONFIG}

echo "============"
echo "ALLENNLP Task finished"

export HEAD_EXP_DIR="${CLUSTER_HOME}/runs/${EXP_ID}"
mkdir -p "${HEAD_EXP_DIR}"
rsync -avuzhP "${SERIAL_DIR}/" "${HEAD_EXP_DIR}/" # Copy output onto headnode

kill $(cat ${SCRATCH_HOME}/elasticsearch_pid)

rm -rf "${SERIAL_DIR}"


echo "============"
echo "results synced"
