#!/bin/bash
#SBATCH --job-name=claude-mythios-unlocked
#SBATCH --partition=gpu-2080ti
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --time=01:30:00
#SBATCH --mail-user=mtran01@univ-lr.fr
#SBATCH --mail-type=END
#SBATCH --gres=gpu:2
#SBATCH --output=finetune-logs/bart-%j.log

cd /Utilisateurs/mtran01/hipe-ocrepair-llm

source .venv/bin/activate

nvidia-smi

echo "--- TESTING PYTORCH ---"
python -c "import torch; print('PyTorch loaded successfully!'); print('CUDA available:', torch.cuda.is_available())"
echo "--- TEST COMPLETE ---"

# python finetune/bart.py \
#     --model bart-base \
#     --data data/datasets/dta19_train.parquet \
#     --output_name bart-base-ocr-dta19

# python finetune/bart.py \
#     --model bart-base \
#     --data data/datasets/icdar2017_train.parquet \
#     --output_name bart-base-ocr-icdar2017

# python finetune/bart.py \
#     --model bart-base \
#     --data data/datasets/impresso-nzz_train.parquet \
#     --output_name bart-base-ocr-impresso-nzz

python finetune/bart.py \
    --model bart-large \
    --data data/datasets/impresso-snippets_train.parquet \
    --output_name bart-large-ocr-impresso-snippets

# python finetune/bart.py \
#     --model bart-base \
#     --data data/datasets/overproof-combined_train.parquet \
#     --output_name bart-base-ocr-overproof-combined