#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=PruningEXP_$1_$2
#SBATCH --output /home/ldery/jupyter_logs/jupyter-log-%x-%J.txt
#SBATCH --time 120:00:00
#SBATCH --exclude=tir-0-36
#SBATCH --gres gpu:v100:1

source ~/.bashrc
conda activate posthoc_mod

echo $1
echo $2
CUDA_VISIBLE_DEVICES=0 ./end_to_end.sh $1 TestBaseline bert-base-uncased BaselineRuns base $2 57     0
CUDA_VISIBLE_DEVICES=0 ./end_to_end.sh $1 TestBaseline bert-base-uncased BaselineRuns base $2 0      0
CUDA_VISIBLE_DEVICES=0 ./end_to_end.sh $1 TestBaseline bert-base-uncased BaselineRuns base $2 232323 0
