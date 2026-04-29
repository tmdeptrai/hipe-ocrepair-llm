#!/bin/bash
#SBATCH --job-name=qwen_fine_tune
#SBATCH --partition=gpu-a40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32768
#SBATCH --time=20:00:00
#SBATCH --mail-user=minhduongqo@gmail.com
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
#SBATCH --output=finetune-logs/qwen-ft-%j.log

cd /Utilisateurs/mtran01/hipe-ocrepair-llm

source .venv/bin/activate

nvidia-smi

echo "--- TESTING PYTORCH ---"
python -c "import torch; print('PyTorch loaded successfully!'); print('CUDA available:', torch.cuda.is_available())"
echo "--- TEST COMPLETE ---"

# Uncomment to choose the experiment that we want:

# ==================== QWEN3-4B BASE ===========================

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --data data/datasets/dta19_train.parquet\
#     --output_name qwen3-4B-ocr-dta19

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --data data/datasets/icdar2017_train.parquet\
#     --output_name qwen3-4B-ocr-icdar2017

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --data data/datasets/impresso-nzz_train.parquet\
#     --output_name qwen3-4B-ocr-impresso-nzz

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --data data/datasets/impresso-snippets_train.parquet\
#     --output_name qwen3-4B-ocr-impresso-snippets

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --data data/datasets/overproof-combined_train.parquet\
#     --output_name qwen3-4B-ocr-overproof-combined

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --data data/languages/english_train.parquet\
#     --output_name qwen3-4B-ocr-english

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --data data/languages/french_train.parquet\
#     --output_name qwen3-4B-ocr-french

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --data data/languages/german_train.parquet\
#     --output_name qwen3-4B-ocr-german

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --data data/hipe_aggregated_train.parquet\
#     --output_name qwen3-4B-ocr-hipe_aggregated

# ==================== QWEN3-4B (WITH METADATA) ==================

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --with_metadata \
#     --data data/datasets/dta19_train.parquet \
#     --output_name qwen3-4B-ocr-meta-dta19

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --with_metadata \
#     --data data/datasets/icdar2017_train.parquet \
#     --output_name qwen3-4B-ocr-meta-icdar2017

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --with_metadata \
#     --data data/datasets/impresso-nzz_train.parquet \
#     --output_name qwen3-4B-ocr-meta-impresso-nzz

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --with_metadata \
#     --data data/datasets/impresso-snippets_train.parquet \
#     --output_name qwen3-4B-ocr-meta-impresso-snippets

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --with_metadata \
#     --data data/datasets/overproof-combined_train.parquet \
#     --output_name qwen3-4B-ocr-meta-overproof-combined

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --with_metadata \
#     --data data/languages/english_train.parquet \
#     --output_name qwen3-4B-ocr-meta-english

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --with_metadata \
#     --data data/languages/french_train.parquet \
#     --output_name qwen3-4B-ocr-meta-french

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --with_metadata \
#     --data data/languages/german_train.parquet \
#     --output_name qwen3-4B-ocr-meta-german

# python finetune/qwen3.py \
#     --model qwen3-4B \
#     --with_metadata \
#     --data data/hipe_aggregated_train.parquet \
#     --output_name qwen3-4B-ocr-meta-hipe_aggregated

# ==================== QWEN3-8B BASE ===========================

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --data data/datasets/dta19_train.parquet \
#     --output_name qwen3-8B-ocr-dta19

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --data data/datasets/icdar2017_train.parquet \
#     --output_name qwen3-8B-ocr-icdar2017

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --data data/datasets/impresso-nzz_train.parquet \
#     --output_name qwen3-8B-ocr-impresso-nzz

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --data data/datasets/impresso-snippets_train.parquet \
#     --output_name qwen3-8B-ocr-impresso-snippets

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --data data/datasets/overproof-combined_train.parquet \
#     --output_name qwen3-8B-ocr-overproof-combined

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --data data/languages/english_train.parquet \
#     --output_name qwen3-8B-ocr-english

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --data data/languages/french_train.parquet \
#     --output_name qwen3-8B-ocr-french

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --data data/languages/german_train.parquet \
#     --output_name qwen3-8B-ocr-german

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --data data/hipe_aggregated_train.parquet \
#     --output_name qwen3-8B-ocr-hipe_aggregated

# ==================== QWEN3-8B (WITH METADATA) ==================

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --with_metadata \
#     --data data/datasets/dta19_train.parquet \
#     --output_name qwen3-8B-ocr-meta-dta19

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --with_metadata \
#     --data data/datasets/icdar2017_train.parquet \
#     --output_name qwen3-8B-ocr-meta-icdar2017

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --with_metadata \
#     --data data/datasets/impresso-nzz_train.parquet \
#     --output_name qwen3-8B-ocr-meta-impresso-nzz

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --with_metadata \
#     --data data/datasets/impresso-snippets_train.parquet \
#     --output_name qwen3-8B-ocr-meta-impresso-snippets

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --with_metadata \
#     --data data/datasets/overproof-combined_train.parquet \
#     --output_name qwen3-8B-ocr-meta-overproof-combined

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --with_metadata \
#     --data data/languages/english_train.parquet \
#     --output_name qwen3-8B-ocr-meta-english

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --with_metadata \
#     --data data/languages/french_train.parquet \
#     --output_name qwen3-8B-ocr-meta-french

# python finetune/qwen3.py \
#     --model qwen3-8B \
#     --with_metadata \
#     --data data/languages/german_train.parquet \
#     --output_name qwen3-8B-ocr-meta-german

python finetune/qwen3.py \
    --model qwen3-8B \
    --with_metadata \
    --data data/hipe_aggregated_train.parquet \
    --output_name qwen3-8B-ocr-meta-hipe_aggregated