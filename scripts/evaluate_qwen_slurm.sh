#!/bin/bash
#SBATCH --job-name=qwen_eval
#SBATCH --partition=gpu-a40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --time=03:00:00
#SBATCH --mail-user=minhduongqo@gmail.com
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
#SBATCH --output=model_eval_logs/qwen-eval-%j.log

cd /Utilisateurs/mtran01/hipe-ocrepair-llm

source .venv/bin/activate

nvidia-smi

echo "--- TESTING PYTORCH ---"
python -c "import torch; print('PyTorch loaded successfully!'); print('CUDA available:', torch.cuda.is_available())"
echo "--- TEST COMPLETE ---"

# Uncomment to choose the evaluation that we want:

# ======================= QWEN3-4B ZERO-SHOT =========================

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/dta19_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/icdar2017_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/impresso-nzz_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/impresso-snippets_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/overproof-combined_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/languages/english_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/languages/french_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/languages/german_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/hipe_aggregated_test.parquet \
#     --batch_size 4

# ======================= QWEN3-4B ZERO-SHOT METADATA =========================

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/dta19_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/icdar2017_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/impresso-nzz_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/impresso-snippets_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/datasets/overproof-combined_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/languages/english_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/languages/french_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/languages/german_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --data_path data/hipe_aggregated_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# ======================= QWEN3-4B FINE-TUNED =========================

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-dta19-base \
#     --data_path data/datasets/dta19_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-icdar2017-base \
#     --data_path data/datasets/icdar2017_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-impresso-nzz-base \
#     --data_path data/datasets/impresso-nzz_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-impresso-snippets-base \
#     --data_path data/datasets/impresso-snippets_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-overproof-combined-base \
#     --data_path data/datasets/overproof-combined_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-english-base \
#     --data_path data/languages/english_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-french-base \
#     --data_path data/languages/french_test.parquet \
#     --batch_size 4

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-german-base \
#     --data_path data/languages/german_test.parquet \
#     --batch_size 4

python model_eval/evaluate_qwen.py \
    --base_model Qwen/Qwen3-4B \
    --adapter_path model/qwen3-4B-ocr-hipe_aggregated-base \
    --data_path data/hipe_aggregated_test.parquet \
    --batch_size 4

# ======================= QWEN3-4B FINE-TUNED (WITH METADATA) =========================

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-dta19-meta \
#     --data_path data/datasets/dta19_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-icdar2017-meta \
#     --data_path data/datasets/icdar2017_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-impresso-nzz-meta \
#     --data_path data/datasets/impresso-nzz_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-impresso-snippets-meta \
#     --data_path data/datasets/impresso-snippets_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-overproof-combined-meta \
#     --data_path data/datasets/overproof-combined_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-english-meta \
#     --data_path data/languages/english_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-french-meta \
#     --data_path data/languages/french_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-german-meta \
#     --data_path data/languages/german_test.parquet \
#     --batch_size 4 \
#     --with_metadata

# python model_eval/evaluate_qwen.py \
#     --base_model Qwen/Qwen3-4B \
#     --adapter_path model/qwen3-4B-ocr-hipe_aggregated-meta \
#     --data_path data/hipe_aggregated_test.parquet \
#     --batch_size 4 \
#     --with_metadata