#!/bin/bash
SCRATCH=/scratch/str8775/conda_env

#SBATCH --job-name=gpt_j_inference
#SBATCH --open-mode=append
#SBATCH --output=$SCRATCH/nlu_ds_ga_1012sp22/final_project/Neural-Question-Generation-with-GPT-J/logs/%j_%x.out
#SBATCH --error=$SCRATCH/nlu_ds_ga_1012sp22/final_project/Neural-Question-Generation-with-GPT-J/logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4

###gpu:v100:1

# Install modules

singularity exec --nv --bind /scratch --overlay \
$SCRATCH/ccu_conda_env.ext3:rw \
$SCRATCH/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c \
"
source /ext3/env.sh
conda activate
cd $SCRATCH/nlu_ds_ga_1012sp22/final_project/Neural-Question-Generation-with-GPT-J
python ./eval/inference.py
"