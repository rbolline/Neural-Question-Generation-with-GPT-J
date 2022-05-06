#!/bin/bash
SCRATCH=/scratch/str8775/conda_env

#SBATCH --job-name=gptj_inf_num_ex1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/str8775/conda_env/nlu_ds_ga_1012sp22/final_project/Neural-Question-Generation-with-GPT-J/%j_%x.out
#SBATCH --error=/scratch/str8775/conda_env/nlu_ds_ga_1012sp22/final_project/Neural-Question-Generation-with-GPT-J/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

###gpu:v100:1

singularity exec --nv --bind /scratch --overlay /scratch/rb4987/conda_env/nlu_conda_env.ext3:rw /scratch/rb4987/conda_env/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/rb4987/NQG_GPTJ_eval/NQG_GPTJ/few_shot_train/fs_train.py
"
