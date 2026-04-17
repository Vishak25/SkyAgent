# SkyAgent: Agentic Aviation Delay Propagation System
## Updated for GMU Hopper Cluster + Free-Tier Cloud Options

---

## Hopper Cluster — What You Have

Based on the GMU ORC documentation, here's what's available to you:

| Resource | Specs |
|----------|-------|
| **GPU Nodes (gpuq)** | 31 nodes × 4× A100 80GB each |
| **H100 Node** | 1 node × 4× H100 80GB |
| **DGX Nodes** | 2 nodes × 8× A100 40GB, 1TB RAM |
| **MIG Slices** | 32× 3g.40gb, 32× 2g.20gb, 64× 1g.10gb |
| **Storage** | 50GB home, 4PB scratch (90-day purge) |
| **Container Runtime** | Singularity (no Docker directly) |
| **Scheduler** | SLURM |
| **Network** | VPN required for OnDemand access |

**Key constraint:** Interactive GPU sessions (`salloc` / Open OnDemand) are limited to MIG slices only. Full A100/H100 GPUs require `sbatch` batch jobs.

---

## Model Selection for Hopper

### Recommended: Qwen 3.5 9B (Development & Primary)
- **VRAM needed:** ~18GB (FP16) or ~10GB (AWQ/GPTQ 4-bit)
- **Fits on:** MIG 2g.20gb (interactive!) or MIG 3g.40gb
- **Why:** You can run this interactively via `salloc`, fast iteration, great tool-calling support
- **HuggingFace:** `Qwen/Qwen3.5-9B-Instruct`

### Stretch: Mistral Small 4 (119B) — Demo/Showcase
- **VRAM needed:** ~240GB FP16, ~60GB with NVFP4 quantization
- **Fits on:** 1 full A100 80GB node (4× A100) with tensor parallel = 4
- **Requires:** `sbatch` job (no interactive)
- **HuggingFace:** `mistralai/Mistral-Small-4-119B-2603-NVFP4`

### Budget Middle Ground: Qwen 3.5 32B or Mistral-7B
- **VRAM needed:** ~64GB (FP16) or ~20GB (4-bit)
- **Fits on:** 1× A100 80GB (full) or MIG 3g.40gb (quantized)

**Strategy:** Develop everything with Qwen 3.5 9B on MIG slices interactively → demo with Mistral Small 4 on full A100 nodes via batch job. The OpenAI-compatible API means zero code changes.

---

## Hopper Setup — Step by Step

### Step 1: SSH into Hopper
```bash
# From your machine (must be on GMU VPN)
ssh <your-netid>@hopper.orc.gmu.edu
```

### Step 2: Set Up Your Working Directory
```bash
# Use scratch for large model files (4PB available, 90-day purge)
mkdir -p /scratch/$USER/skyagent
mkdir -p /scratch/$USER/hf_cache
mkdir -p /scratch/$USER/singularity_images

# Set HuggingFace cache to scratch (models are huge)
echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc
echo 'export HF_TOKEN="your_huggingface_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Pull the vLLM Singularity Container
```bash
# Hopper uses Singularity, not Docker
# Pull from the official vLLM Docker image → auto-converts to .sif
cd /scratch/$USER/singularity_images
module load singularity

# Pull the latest vLLM OpenAI-compatible server image
singularity pull vllm-openai_latest.sif docker://vllm/vllm-openai:latest

# This will take 10-15 minutes and produce a ~8GB .sif file
```

### Step 4: Download the Model (Do This Once)
```bash
# Interactive session on a login node (has internet access)
# Models download to your HF cache on scratch

pip install --user huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-9B-Instruct', cache_dir='/scratch/$USER/hf_cache')
"

# For Mistral Small 4 (optional, 119B model is ~60GB quantized):
# python -c "
# from huggingface_hub import snapshot_download
# snapshot_download('mistralai/Mistral-Small-4-119B-2603-NVFP4', cache_dir='/scratch/$USER/hf_cache')
# "
```

---

## SLURM Scripts for vLLM

### Script A: vLLM Server on MIG Slice (Interactive Dev — Qwen 9B)

```bash
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

# ─── Environment ───
export HF_HOME=/scratch/$USER/hf_cache
export HF_TOKEN="your_token_here"
export VLLM_IMAGE=/scratch/$USER/singularity_images/vllm-openai_latest.sif

# ─── Load Singularity ───
module load singularity

# ─── Get the node hostname and print connection info ───
NODE_HOSTNAME=$(hostname)
echo "=============================================="
echo "vLLM server starting on: $NODE_HOSTNAME"
echo "To connect via SSH tunnel from your laptop:"
echo "  ssh -L 8000:${NODE_HOSTNAME}:8000 ${USER}@hopper.orc.gmu.edu"
echo "Then access the API at: http://localhost:8000"
echo "=============================================="

# ─── Start vLLM Server ───
singularity exec --nv \
  -B /scratch/$USER:/scratch/$USER \
  -B $HF_HOME:/hf_cache \
  --env HF_HOME=/hf_cache \
  --env HF_TOKEN=$HF_TOKEN \
  $VLLM_IMAGE \
  vllm serve Qwen/Qwen3.5-9B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --api-key skyagent-dev
```

**Save as:** `/scratch/$USER/skyagent/slurm/vllm_qwen9b.slurm`

**Submit:**
```bash
mkdir -p /scratch/$USER/skyagent/logs
sbatch /scratch/$USER/skyagent/slurm/vllm_qwen9b.slurm
```

### Script B: vLLM Server on Full A100 Node (Mistral Small 4 — Demo)

```bash
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
echo "Mistral Small 4 (119B NVFP4) on 4× A100 80GB"
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
```

### Script C: Agent Service (Runs Alongside vLLM)

```bash
#!/bin/bash
#SBATCH --job-name=skyagent-agents
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/%u/skyagent/logs/agents_%j.log
#SBATCH --error=/scratch/%u/skyagent/logs/agents_%j.err

# ─── This runs on a CPU node while vLLM runs on a GPU node ───
# ─── Set VLLM_BASE_URL to point to the GPU node ───

# Read the vLLM node hostname from the vLLM job's log
# (You'll set this manually or via a coordination file)
export VLLM_BASE_URL="http://<vllm-gpu-node>:8000/v1"
export VLLM_API_KEY="skyagent-dev"

cd /scratch/$USER/skyagent

# Load Python environment
module load python/3.10
source /scratch/$USER/skyagent/venv/bin/activate

# Start the FastAPI agent service
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

---

## SSH Tunneling — Accessing vLLM from Your Laptop

Since Hopper compute nodes aren't directly accessible from outside, you'll use SSH tunneling:

```bash
# Terminal 1: Tunnel to vLLM server
# (Replace <gpu-node> with the actual hostname from your SLURM job log)
ssh -L 8000:<gpu-node>:8000 <your-netid>@hopper.orc.gmu.edu

# Terminal 2: Tunnel to agent service (if running)
ssh -L 8080:<cpu-node>:8080 <your-netid>@hopper.orc.gmu.edu

# Now from your laptop:
curl http://localhost:8000/v1/models
# Should return the model info
```

**Testing vLLM tool-calling from your laptop:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="skyagent-dev"
)

# Test basic completion
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B-Instruct",
    messages=[{"role": "user", "content": "What causes flight delays?"}],
    max_tokens=200
)
print(response.choices[0].message.content)

# Test tool-calling
tools = [{
    "type": "function",
    "function": {
        "name": "get_airport_weather",
        "description": "Get current METAR weather for an airport",
        "parameters": {
            "type": "object",
            "properties": {
                "icao_code": {
                    "type": "string",
                    "description": "ICAO airport code, e.g. KJFK"
                }
            },
            "required": ["icao_code"]
        }
    }
}]

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B-Instruct",
    messages=[{"role": "user", "content": "What's the weather at JFK airport right now?"}],
    tools=tools,
    tool_choice="auto"
)
print(response.choices[0].message.tool_calls)
```

---

## Python Environment Setup on Hopper

```bash
# On a login node (has internet)
module load python/3.10

# Create a virtual environment on scratch
python -m venv /scratch/$USER/skyagent/venv
source /scratch/$USER/skyagent/venv/bin/activate

# Install project dependencies
pip install --upgrade pip
pip install \
    langchain langgraph langchain-openai \
    torch torchvision torch-geometric \
    fastapi uvicorn websockets \
    chromadb \
    redis \
    httpx aiohttp \
    pandas numpy scikit-learn \
    openai \
    pyopensky \
    pydantic \
    python-dotenv

# For the frontend (on your laptop, not Hopper)
# cd frontend && npm install
```

---

## Architecture — Adapted for Hopper

```
YOUR LAPTOP (via SSH tunnels)
    │
    ├── Browser → React Frontend (localhost:3000)
    │                  │
    │                  ▼
    │         FastAPI Gateway (localhost:8080)
    │         ┌────── on Hopper CPU node ──────┐
    │         │                                 │
    │         │   Orchestrator (LangGraph)       │
    │         │   ├─► Flight Monitor Agent       │
    │         │   ├─► Weather Agent              │
    │         │   ├─► Delay Risk Agent           │
    │         │   └─► Rerouting Agent            │
    │         │         │                        │
    │         └─────────┼────────────────────────┘
    │                   │ HTTP (internal Hopper network)
    │                   ▼
    │         ┌── Hopper GPU Node (A100/H100) ──┐
    │         │                                  │
    │         │   vLLM Server (Singularity)       │
    │         │   Qwen 3.5 9B / Mistral Small 4   │
    │         │   OpenAI-compatible API (:8000)    │
    │         │                                  │
    │         └──────────────────────────────────┘
    │
    └── SSH Tunnel: laptop:8000 → gpu-node:8000
                    laptop:8080 → cpu-node:8080
```

**Key insight:** On Hopper, vLLM runs on a GPU node via SLURM, and your agent code runs on a CPU node. They communicate over Hopper's internal network. You SSH-tunnel both to your laptop for development and the frontend.

---

## Free Cloud Options (After Hopper — No Money Spent)

### Option 1: AWS Free Tier + Credits
- **AWS Educate / AWS Academy:** GMU likely has an institutional agreement — check with your department or ORC. Many CS programs give students $100-$300 in credits.
- **AWS Free Tier:** t2.micro (free for 12 months) can host your FastAPI + frontend. No free GPU though.

### Option 2: Google Cloud Free Tier
- **$300 free credits** for new accounts (valid 90 days)
- Can spin up an `a2-highgpu-1g` (1× A100 40GB) for ~$3.67/hr — your $300 gets you ~80 hours of GPU time
- Enough to do a polished demo deployment

### Option 3: Oracle Cloud Free Tier
- **Always Free:** 1 VM with 1/8 GPU (A10G) — limited but free forever
- ARM instances (4 CPUs, 24GB RAM) are always free — great for hosting the API/frontend

### Option 4: Hugging Face Spaces (Free for CPU)
- Host your frontend + API for free on HF Spaces (Gradio/Streamlit)
- Use HF Inference Endpoints for the model ($0.60/hr for a small model — not free, but cheap)

### Option 5: Modal / RunPod / Lambda Labs (Pay-per-second)
- **Modal:** $30 free monthly credits, GPUs billed per second
- **RunPod:** ~$0.44/hr for RTX A4000, ~$1.99/hr for A100
- **Lambda Labs:** ~$1.10/hr for A10

### Recommended Free Path:
1. **Develop & test** entirely on Hopper (free, you already have access)
2. **Deploy frontend** on Oracle Cloud Always Free or HF Spaces
3. **Demo the GPU inference** live on Hopper via SSH tunnel + screen recording
4. **If you get AWS credits** from GMU, do a proper ECS deployment

---

## Updated Project Structure

```
skyagent/
├── README.md
├── slurm/                            # SLURM job scripts (Hopper-specific)
│   ├── vllm_qwen9b.slurm            # Dev: Qwen 9B on MIG slice
│   ├── vllm_mistral.slurm           # Demo: Mistral Small 4 on 4× A100
│   ├── agent_service.slurm          # Agent orchestrator on CPU node
│   └── gnn_training.slurm           # ST-GNN training job
│
├── containers/                       # Singularity container definitions
│   └── skyagent.def                  # Custom container def (if needed)
│
├── docker/                           # For cloud deployment later
│   ├── docker-compose.yml
│   ├── Dockerfile.vllm
│   └── Dockerfile.agent
│
├── terraform/                        # AWS IaC (for when you get credits)
│   ├── main.tf
│   └── variables.tf
│
├── src/
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── flight_monitor.py
│   │   ├── weather_agent.py
│   │   ├── delay_risk_agent.py
│   │   └── rerouting_agent.py
│   │
│   ├── tools/
│   │   ├── opensky_tools.py
│   │   ├── weather_tools.py
│   │   ├── bts_tools.py
│   │   ├── gnn_tools.py
│   │   └── routing_tools.py
│   │
│   ├── models/
│   │   ├── delay_gnn/
│   │   │   ├── model.py
│   │   │   ├── graph_builder.py
│   │   │   └── train.py
│   │   └── embeddings.py
│   │
│   ├── data/
│   │   ├── ingest/
│   │   │   ├── opensky_ingester.py
│   │   │   ├── bts_loader.py
│   │   │   └── weather_fetcher.py
│   │   ├── schemas.py
│   │   └── db.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── websocket.py
│   │   └── routes.py
│   │
│   └── config/
│       ├── settings.py
│       └── airports.json
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FlightMap.tsx
│   │   │   ├── DelayGraph.tsx
│   │   │   ├── RiskDashboard.tsx
│   │   │   ├── AgentActivityFeed.tsx
│   │   │   └── RouteComparison.tsx
│   │   └── hooks/
│   │       └── useWebSocket.ts
│   └── package.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_gnn_training.ipynb
│   ├── 03_vllm_tool_calling_test.ipynb   # Test vLLM on Hopper
│   ├── 04_agent_testing.ipynb
│   └── 05_vllm_benchmarks.ipynb          # Qwen vs Mistral benchmark
│
├── scripts/
│   ├── download_bts_data.sh
│   ├── download_model.py
│   ├── setup_hopper_env.sh               # One-shot Hopper setup
│   └── tunnel.sh                         # SSH tunnel helper
│
├── tests/
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_gnn.py
│
└── requirements.txt
```

---

## Implementation Phases (Revised for Hopper)

### Phase 1: Hopper Environment + vLLM (Week 1)
- [ ] Set up scratch directories, HF cache, Python venv
- [ ] Pull vLLM Singularity container
- [ ] Download Qwen 3.5 9B model weights
- [ ] Write and test SLURM script for vLLM server
- [ ] Verify tool-calling works via SSH tunnel from laptop
- [ ] **Deliverable:** vLLM running on Hopper, tool-calling working

### Phase 2: Data Pipeline + GNN Training (Week 2-3)
- [ ] Download BTS data, load into scratch
- [ ] Build airport-flight graph (PyTorch Geometric)
- [ ] Write GNN training SLURM script, train on A100
- [ ] Set up OpenSky API wrapper with caching
- [ ] Fetch and parse METAR/TAF data
- [ ] **Deliverable:** Trained ST-GNN model, data pipeline working

### Phase 3: Agent System (Week 4-5)
- [ ] Build LangGraph orchestrator
- [ ] Implement 4 agents + their tools
- [ ] Wire GNN model as a callable tool
- [ ] Parallel agent execution (asyncio)
- [ ] Test full pipeline: event → agents → recommendation
- [ ] **Deliverable:** Multi-agent system running on Hopper

### Phase 4: Frontend + Demo (Week 6-7)
- [ ] FastAPI backend with WebSocket
- [ ] React dashboard (run locally, point to Hopper via tunnel)
- [ ] Live flight map, delay graph, agent activity feed
- [ ] **Deliverable:** Working demo via SSH tunnel

### Phase 5: Cloud Deployment (Week 8 — if credits available)
- [ ] Docker Compose for local reproducibility
- [ ] Terraform for AWS (if credits obtained)
- [ ] OR deploy frontend to HF Spaces / Oracle Free Tier
- [ ] Demo video with screen recording
- [ ] **Deliverable:** Portfolio-ready project with README + video

---

## Quick Reference: Common Hopper Commands

```bash
# Check GPU availability
sinfo -p gpuq -o "%N %G %C %m %T"

# Check your running jobs
squeue -u $USER

# Submit vLLM server job
sbatch slurm/vllm_qwen9b.slurm

# Check job output (get job ID from squeue)
tail -f /scratch/$USER/skyagent/logs/vllm_<jobid>.log

# Cancel a job
scancel <jobid>

# Interactive GPU session (MIG slices only)
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:3g.40gb:1 --cpus-per-task=8 --mem=64G -t 0-02:00:00

# SSH tunnel from your laptop
ssh -L 8000:<gpu-node>:8000 <netid>@hopper.orc.gmu.edu
```

---

## Resources

- **Hopper Wiki:** https://wiki.orc.gmu.edu/mkdocs/
- **GPU Jobs Guide:** https://wiki.orc.gmu.edu/mkdocs/Running_GPU_Jobs/
- **Singularity on Hopper:** https://wiki.orc.gmu.edu/mkdocs/Containerized_jobs_on_Hopper/
- **vLLM Docs:** https://docs.vllm.ai/
- **vLLM on HPC (reference):** https://github.com/hwang2006/gpt-oss-with-vllm-on-supercomputer
- **LangGraph:** https://langchain-ai.github.io/langgraph/
- **OpenSky API:** https://openskynetwork.github.io/opensky-api/
- **BTS Data:** https://www.transtats.bts.gov/
