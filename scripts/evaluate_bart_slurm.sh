#!/bin/bash
#SBATCH --job-name=gpt7.0
#SBATCH --partition=gpu-2080ti
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --time=03:00:00
#SBATCH --mail-user=mtran01@univ-lr.fr
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
#SBATCH --output=model_eval_logs/bart-eval-%j.log

cd /Utilisateurs/mtran01/hipe-ocrepair-llm

source .venv/bin/activate

nvidia-smi

echo "--- TESTING PYTORCH ---"
python -c "import torch; print('PyTorch loaded successfully!'); print('CUDA available:', torch.cuda.is_available())"
echo "--- TEST COMPLETE ---"

# Uncomment to choose the evaluation that we want:

# ============ BART-BASE VARIATIONS =================

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-base-ocr-dta19 \
#     --data_path data/datasets/dta19_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-base-ocr-icdar2017 \
#     --data_path data/datasets/icdar2017_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-base-ocr-impresso-nzz \
#     --data_path data/datasets/impresso-nzz_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-base-ocr-impresso-snippets \
#     --data_path data/datasets/impresso-snippets_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-base-ocr-overproof-combined \
#     --data_path data/datasets/overproof-combined_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-base-ocr-english \
#     --data_path data/languages/english_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-base-ocr-french \
#     --data_path data/languages/french_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-base-ocr-german \
#     --data_path data/languages/german_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-base-ocr-hipe_aggregated \
#     --data_path data/hipe_aggregated_test.parquet \
#     --batch_size 4

# ============ BART-LARGE VARIATIONS =================

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-large-ocr-dta19 \
#     --data_path data/datasets/dta19_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-large-ocr-icdar2017 \
#     --data_path data/datasets/icdar2017_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-large-ocr-impresso-nzz \
#     --data_path data/datasets/impresso-nzz_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-large-ocr-impresso-snippets \
#     --data_path data/datasets/impresso-snippets_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-large-ocr-overproof-combined \
#     --data_path data/datasets/overproof-combined_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-large-ocr-english \
#     --data_path data/languages/english_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-large-ocr-french \
#     --data_path data/languages/french_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_bart.py \
#     --model_path model/bart-large-ocr-german \
#     --data_path data/languages/german_test.parquet \
#     --batch_size 4

python model_eval/evaluate_bart.py \
    --model_path model/bart-large-ocr-hipe_aggregated \
    --data_path data/hipe_aggregated_test.parquet \
    --batch_size 4