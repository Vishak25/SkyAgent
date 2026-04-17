#!/bin/bash
#SBATCH --job-name=skyagent-gnn-train
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/%u/skyagent/logs/gnn_train_%j.log
#SBATCH --error=/scratch/%u/skyagent/logs/gnn_train_%j.err

# ─── Train the ST-GNN on IEM METAR data ────────────────────────────────────

cd /scratch/$USER/skyagent

module load python/3.10
source /scratch/$USER/skyagent/venv/bin/activate

echo "Starting ST-GNN training..."
echo "Data: data/raw/iem_metar_30d.csv"

python -m src.models.delay_gnn.train \
  --data data/raw/iem_metar_30d.csv \
  --epochs 200 \
  --batch-size 32 \
  --lr 1e-3 \
  --patience 25 \
  --hidden 32 \
  --seed 42 \
  --output-dir model_weights \
  --device cuda

echo "Training complete. Weights saved to model_weights/stgnn_best.pt"
