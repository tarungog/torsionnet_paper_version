#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=ppo_gat_eval
#ppo_rtgn_pruning_fix_lignin_log_curr
#SBATCH --mail-user=tgog@umich.edu
#SBATCH --cpus-per-task=35
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --get-user-env
# The application(s) to execute along with its input arguments and options:
conda activate my-rdkit-env
module load cuda/10.1.243
module load gcc
python run_eval.py
# python run_batch_train2.py
