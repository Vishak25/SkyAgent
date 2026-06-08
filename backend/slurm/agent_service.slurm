#!/bin/bash
#SBATCH --job-name=skyagent-service
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/%u/skyagent/logs/agents_%j.log
#SBATCH --error=/scratch/%u/skyagent/logs/agents_%j.err

# ─── This runs on a CPU node while vLLM runs on a GPU node ─────────────────
# ─── Set LLM_BASE_URL to the GPU node hostname printed by vllm_qwen9b.slurm ─

# Read the vLLM node from the job log or pass it via environment variable
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
export LLM_API_KEY="${LLM_API_KEY:-skyagent-dev}"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.5-9B-Instruct}"

# API keys for data sources (set these in ~/.bashrc or pass via --export)
export FLIGHTAWARE_API_KEY="${FLIGHTAWARE_API_KEY:-}"
export CHECKWX_API_KEY="${CHECKWX_API_KEY:-}"
export OPENSKY_USER="${OPENSKY_USER:-}"
export OPENSKY_PASSWORD="${OPENSKY_PASSWORD:-}"

cd /scratch/$USER/skyagent

module load python/3.10
source /scratch/$USER/skyagent/venv/bin/activate

echo "Starting SkyAgent service..."
echo "LLM_BASE_URL: $LLM_BASE_URL"

# Start the FastAPI agent service
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
