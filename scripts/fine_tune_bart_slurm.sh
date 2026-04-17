#!/bin/bash
#SBATCH --job-name=claude-mythios-unlocked
#SBATCH --partition=gpu-2080ti
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --time=06:00:00
#SBATCH --mail-user=mtran01@univ-lr.fr
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
#SBATCH --output=finetune-logs/bart-ft-%j.log

cd /Utilisateurs/mtran01/hipe-ocrepair-llm

source .venv/bin/activate

nvidia-smi

echo "--- TESTING PYTORCH ---"
python -c "import torch; print('PyTorch loaded successfully!'); print('CUDA available:', torch.cuda.is_available())"
echo "--- TEST COMPLETE ---"

# Uncomment to choose the experiment that we want:

# ============= BART-BASE VARIATIONS ====================

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

# python finetune/bart.py \
#     --model bart-base \
#     --data data/datasets/impresso-snippets_train.parquet \
#     --output_name bart-base-ocr-impresso-snippets

# python finetune/bart.py \
#     --model bart-base \
#     --data data/datasets/overproof-combined_train.parquet \
#     --output_name bart-base-ocr-overproof-combined

# python finetune/bart.py \
#     --model bart-base \
#     --data data/languages/english_train.parquet \
#     --output_name bart-base-ocr-english

# python finetune/bart.py \
#     --model bart-base \
#     --data data/languages/french_train.parquet \
#     --output_name bart-base-ocr-french

# python finetune/bart.py \
#     --model bart-base \
#     --data data/languages/german_train.parquet \
#     --output_name bart-base-ocr-german

# python finetune/bart.py \
#     --model bart-base \
#     --data data/hipe_aggregated_train.parquet \
#     --output_name bart-base-ocr-hipe_aggregated

# ============ BART-LARGE VARIATIONS =================

# python finetune/bart.py \
#     --model bart-large \
#     --data data/datasets/dta19_train.parquet \
#     --output_name bart-large-ocr-dta19 \

# python finetune/bart.py \
#     --model bart-large \
#     --data data/datasets/icdar2017_train.parquet \
#     --output_name bart-large-ocr-icdar2017

# python finetune/bart.py \
#     --model bart-large \
#     --data data/datasets/impresso-nzz_train.parquet \
#     --output_name bart-large-ocr-impresso-nzz

# python finetune/bart.py \
#     --model bart-large \
#     --data data/datasets/impresso-snippets_train.parquet \
#     --output_name bart-large-ocr-impresso-snippets

# python finetune/bart.py \
#     --model bart-large \
#     --data data/datasets/overproof-combined_train.parquet \
#     --output_name bart-large-ocr-overproof-combined

# python finetune/bart.py \
#     --model bart-large \
#     --data data/languages/english_train.parquet \
#     --output_name bart-large-ocr-english

# python finetune/bart.py \
#     --model bart-large \
#     --data data/languages/french_train.parquet \
#     --output_name bart-large-ocr-french

# python finetune/bart.py \
#     --model bart-large \
#     --data data/languages/german_train.parquet \
#     --output_name bart-large-ocr-german

python finetune/bart.py \
    --model bart-large \
    --data data/hipe_aggregated_train.parquet \
    --output_name bart-large-ocr-hipe_aggregated