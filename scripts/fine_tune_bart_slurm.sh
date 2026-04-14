#!/bin/bash
#SBATCH --job-name=ireallyamfantastic
#SBATCH --partition=gpu-2080ti
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --time=01:00:00
#SBATCH --mail-user=mtran01@univ-lr.fr
#SBATCH --mail-type=END
#SBATCH --gres=gpu:2
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

cd /Utilisateurs/mtran01/hipe-ocrepair-llm

source .venv/bin/activate

nvidia-smi

echo "--- TESTING PYTORCH ---"
python -c "import torch; print('PyTorch loaded successfully!'); print('CUDA available:', torch.cuda.is_available())"
echo "--- TEST COMPLETE ---"

echo "Training started, please wait :D"

python finetune/bart.py --model bart-base #change to bart-large later