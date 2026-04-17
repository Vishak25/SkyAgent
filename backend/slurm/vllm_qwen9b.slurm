#!/bin/bash
#SBATCH --job-name=vllm-skyagent
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/%u/skyagent/logs/vllm_%j.log
#SBATCH --error=/scratch/%u/skyagent/logs/vllm_%j.err

# ─── Environment ───────────────────────────────────────────────────────────
export HF_HOME=/scratch/$USER/hf_cache
# Export HF_TOKEN before submitting this job (e.g. `export HF_TOKEN=hf_xxx`
# in ~/.bashrc) — never hardcode it here.
: "${HF_TOKEN:?HF_TOKEN must be set in the submitting shell}"
export VLLM_IMAGE=/scratch/$USER/singularity_images/vllm-openai_latest.sif

# ─── Load Singularity ───────────────────────────────────────────────────────
module load singularity

# ─── Print connection info ───────────────────────────────────────────────────
NODE_HOSTNAME=$(hostname)
echo "=============================================="
echo "vLLM server starting on: $NODE_HOSTNAME"
echo "To connect via SSH tunnel from your laptop:"
echo "  ssh -L 8000:${NODE_HOSTNAME}:8000 ${USER}@hopper.orc.gmu.edu"
echo "Then access the API at: http://localhost:8000"
echo "=============================================="

# ─── Start vLLM Server ──────────────────────────────────────────────────────
# Override MIG UUID that SLURM sets — vLLM requires an integer device index
export CUDA_VISIBLE_DEVICES=0

singularity exec --nv \
  -B /scratch/$USER:/scratch/$USER \
  -B $HF_HOME:/hf_cache \
  --env HF_HOME=/hf_cache \
  --env HF_TOKEN=$HF_TOKEN \
  $VLLM_IMAGE \
  vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --api-key skyagent-dev
