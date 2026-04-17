#!/bin/bash
#SBATCH --job-name=vllm-mistral-demo
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --mem=256G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/%u/skyagent/logs/vllm_mistral_%j.log
#SBATCH --error=/scratch/%u/skyagent/logs/vllm_mistral_%j.err

export HF_HOME=/scratch/$USER/hf_cache
export HF_TOKEN="your_token_here"
export VLLM_IMAGE=/scratch/$USER/singularity_images/vllm-openai_latest.sif

module load singularity

NODE_HOSTNAME=$(hostname)
echo "=============================================="
echo "Mistral Small 4 (119B NVFP4) on 4x A100 80GB"
echo "vLLM server on: $NODE_HOSTNAME"
echo "SSH tunnel: ssh -L 8000:${NODE_HOSTNAME}:8000 ${USER}@hopper.orc.gmu.edu"
echo "=============================================="

singularity exec --nv \
  -B /scratch/$USER:/scratch/$USER \
  -B $HF_HOME:/hf_cache \
  --env HF_HOME=/hf_cache \
  --env HF_TOKEN=$HF_TOKEN \
  $VLLM_IMAGE \
  vllm serve mistralai/Mistral-Small-4-119B-2603-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --attention-backend FLASH_ATTN_MLA \
    --tool-call-parser mistral \
    --enable-auto-tool-choice \
    --reasoning-parser mistral \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 \
    --api-key skyagent-demo
