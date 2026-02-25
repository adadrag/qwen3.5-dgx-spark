# Qwen3.5-35B-A3B on NVIDIA DGX Spark

A complete guide to running [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) on the NVIDIA DGX Spark (GB10) using vLLM. Includes installation instructions, benchmark results, and configuration tips.

## Table of Contents

- [Overview](#overview)
- [Hardware](#hardware)
- [Why This Model on DGX Spark](#why-this-model-on-dgx-spark)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Usage](#api-usage)
- [Benchmark Results](#benchmark-results)
- [Vision / Multimodal Features](#vision--multimodal-features)
- [Stress Testing / Context Limits](#stress-testing--context-limits)
- [Comparison with Other Models](#comparison-with-other-models)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

Qwen3.5-35B-A3B is a Mixture-of-Experts (MoE) multimodal model with:

- **35B total parameters**, but only **3B active** at inference
- **262K native context** (extendable to 1M+ with YaRN)
- **Multimodal**: text, images, and video input
- **Thinking mode**: built-in chain-of-thought reasoning
- **Tool calling**: function calling and MCP support
- **201 languages** supported
- **Apache 2.0 license**

## Hardware

| Component | Specification |
|-----------|---------------|
| Device | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (Blackwell) |
| Memory | 128 GB unified (shared CPU/GPU) |
| CUDA Capability | 12.1 |
| Storage | 3.7 TB NVMe SSD |
| OS | DGX OS (Ubuntu 24.04 based) |

## Why This Model on DGX Spark

The MoE architecture makes this model uniquely suited for the DGX Spark:

- **Only 3B active parameters** means fast inference (~31 tok/s) despite 35B total
- **~70 GB model weights** in BF16 fits comfortably in 128 GB unified memory
- **28.6 GB remaining for KV cache** after loading, supporting 374K tokens
- Benchmarks competitive with models 10-40x the inference cost

## Installation

### Prerequisites

- NVIDIA DGX Spark with DGX OS
- Docker installed and configured
- SSH access to the DGX Spark

### Step 1: Configure Docker Permissions

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2: Pull the vLLM Docker Image

> **Important**: The standard NVIDIA vLLM container (`nvcr.io/nvidia/vllm:26.01-py3`) ships with vLLM 0.13.0, which does **not** support Qwen3.5. You need the nightly build with Qwen3.5 support.

```bash
docker pull vllm/vllm-openai:cu130-nightly
```

This image contains:
- vLLM v0.16.0+ (with `Qwen3_5MoeForConditionalGeneration` support)
- CUDA 13.1
- PyTorch with Blackwell support
- FlashAttention backend

### Step 3: Launch the Model

```bash
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  Qwen/Qwen3.5-35B-A3B \
    --served-model-name qwen3.5-35b \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.80 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enable-prefix-caching
```

> **Note**: The `vllm/vllm-openai:cu130-nightly` image has `vllm serve` as its entrypoint, so the model name is passed directly as the first argument (not `vllm serve Qwen/...`).

First run will download ~70 GB of model weights. Subsequent starts use the cached weights.

### Step 4: Verify

Wait for the server to fully initialize (model download + CUDA graph capture takes ~15 minutes on first run), then:

```bash
curl http://localhost:8000/v1/models
```

Expected output:
```json
{
  "data": [{"id": "qwen3.5-35b", "object": "model", "max_model_len": 262144}]
}
```

## Configuration

### Memory Allocation

| `--gpu-memory-utilization` | Model Weights | KV Cache | Notes |
|---------------------------|---------------|----------|-------|
| 0.80 (recommended) | ~70 GB | 28.6 GB (374K tokens) | Stable, no OOM risk |
| 0.85 | ~70 GB | ~35 GB (~460K tokens) | More headroom for long context |
| 0.90 | ~70 GB | ~42 GB (~550K tokens) | Risk of OOM after extended use |

> Community reports suggest `0.90` can cause OOM after ~1 hour. Stick with `0.80` for stability.

### Context Length Options

**Default (262K native):**
```
--max-model-len 262144
```

**Extended (1M with YaRN scaling):**
```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3.5-35B-A3B \
  --max-model-len 1010000 \
  --hf-overrides '{"text_config": {"rope_parameters": {
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 262144
  }}}'
```

### Text-Only Mode (Disable Vision Encoder)

If you only need text inference, disable the vision encoder to save memory:

```
--language-model-only
```

### Speculative Decoding (Multi-Token Prediction)

For potentially faster inference:

```
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

## API Usage

The server exposes an **OpenAI-compatible API** at `http://<dgx-spark-ip>:8000/v1`.

### Text Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 1024,
    "temperature": 1.0,
    "top_p": 0.95
  }'
```

### Image Analysis

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        {"type": "text", "text": "Describe this image in detail."}
      ]
    }],
    "max_tokens": 1024
  }'
```

### Disable Thinking Mode

By default, the model uses chain-of-thought reasoning. To disable it for faster responses:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1024,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.42.2:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3.5-35b",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    max_tokens=1024,
    temperature=1.0,
    top_p=0.95,
)

print(response.choices[0].message.content)
```

### Recommended Sampling Parameters

| Mode | Temperature | top_p | top_k | presence_penalty |
|------|------------|-------|-------|-----------------|
| Thinking - General | 1.0 | 0.95 | 20 | 1.5 |
| Thinking - Coding | 0.6 | 0.95 | 20 | 0.0 |
| Instruct - General | 0.7 | 0.8 | 20 | 1.5 |
| Instruct - Reasoning | 1.0 | 1.0 | 40 | 2.0 |

## Benchmark Results

### Token Speed on DGX Spark

Measured on February 25, 2026 with BF16 precision:

| Test | Prompt Tokens | Output Tokens | Time | Speed |
|------|--------------|---------------|------|-------|
| Short response | 18 | 128 | 4.1s | **31.1 tok/s** |
| Medium response | 35 | 1,024 | 32.2s | **31.8 tok/s** |
| Long response | 32 | 3,831 | 121.0s | **31.6 tok/s** |

- **Consistent ~31-32 tokens/sec** regardless of output length
- **Time to first token: ~0.1s**
- Approximately **24 words per second**

### Reasoning Tests

All classic reasoning benchmarks passed correctly:

| Test | Question | Model Answer | Correct |
|------|----------|-------------|---------|
| Trick question | "A farmer has 17 sheep. All but 9 die. How many left?" | 9 sheep | Yes |
| Widget problem | "5 machines, 5 minutes, 5 widgets. 100 machines, 100 widgets?" | 5 minutes | Yes |
| Sibling puzzle | "Sally has 3 brothers. Each brother has 2 sisters. How many sisters?" | 1 sister | Yes |
| Bat & ball (CRT) | "Bat costs $1 more than ball. Total $1.10. Ball cost?" | $0.05 | Yes |
| Box labeling | Mislabeled boxes logic puzzle | Perfect reasoning | Yes |
| LIS algorithm | Code + O(n log n) complexity analysis | Both approaches correct | Yes |

### Academic Benchmarks vs. Comparable Models

| Benchmark | Qwen3.5-35B-A3B (3B active) | Qwen3.5-27B (27B dense) | GPT-OSS-120B | Qwen3-235B | GPT-5-mini |
|-----------|------------------------------|-------------------------|--------------|------------|------------|
| MMLU-Pro | 85.3 | 86.1 | 80.8 | 84.4 | 83.7 |
| GPQA Diamond | 84.2 | 85.5 | 80.1 | 81.1 | 82.8 |
| MMLU-Redux | 93.3 | 93.2 | 91.0 | 93.8 | 93.7 |
| IFEval | 91.9 | 95.0 | 88.9 | 87.8 | 93.9 |
| SWE-bench Verified | 69.2 | 72.4 | 62.0 | -- | 72.0 |
| LiveCodeBench v6 | 74.6 | 80.7 | 82.7 | 75.1 | 80.5 |
| CodeForces | 2028 | 1899 | 2157 | 2146 | 2160 |
| HMMT Feb 25 | 89.0 | 92.0 | 90.0 | 85.1 | 89.2 |
| HLE w/ CoT | 22.4 | 24.3 | 14.9 | 18.2 | 19.4 |

**Key takeaway**: With only 3B active parameters, this model beats GPT-OSS-120B and Qwen3-235B on most knowledge and reasoning benchmarks.

## Stress Testing / Context Limits

We ran extensive stress tests to find the breaking point of vLLM + Qwen3.5-35B-A3B on DGX Spark. Spoiler: **vLLM is extremely resilient** — it never OOM'd or crashed.

### Single Request Context Tests

| Test | Prompt Tokens | Max Tokens | Result | Time |
|------|--------------|------------|--------|------|
| 50K moderate | 62,501 | 10 | OK | 18.8s |
| 130K half capacity | 162,501 | 10 | OK | 48.3s |
| 250K near max | ~262K | 10 | Rejected (over limit by 1 token) | instant |
| 500K double limit | ~500K text sent | 100 | Rejected (tokenizer capped at 262K) | instant |
| 1M quadruple limit | ~1M text sent | 100 | Rejected (tokenizer capped at 262K) | instant |

**Key finding**: vLLM **truncates tokenization** at `max_model_len`. Even sending 1 million tokens of text, the tokenizer stops at 262,144 and returns a clean error. No OOM possible through the API.

### Concurrent Request Tests (262K context config)

| Test | Requests | Tokens Each | Total Demand | Result |
|------|----------|-------------|-------------|--------|
| 4x prefill only | 4 | 250K prompt + 10 output | 1M tokens | All OK (serialized, 72s each) |
| 4x with generation | 4 | 192K prompt + 2000 output | ~776K tokens | 1 completed (595s), 3 timed out waiting |

**Key finding**: vLLM's scheduler **serializes** requests when KV cache is full. It processes one, frees KV, then processes the next. Requests queue rather than crash.

### Forced 1M Context Window (Override)

We restarted vLLM with `--max-model-len 1048576` and `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` to force 1M context (4x the model's trained 262K):

```bash
docker run -d \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:cu130-nightly \
  Qwen/Qwen3.5-35B-A3B \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.95
```

**Startup results:**
- Model weights: 65.53 GiB
- Available KV cache: 44.47 GiB
- KV cache capacity: **581,856 tokens**
- Max concurrency for 1M requests: **2.21x**

| Test | Tokens | Result | Time |
|------|--------|--------|------|
| 300K prompt (over 262K native) | 300,001 | OK | 179.7s |
| 3x concurrent 300K | ~450K each | All completed (serialized) | ~380s |

**Key finding**: The model processes 300K+ tokens beyond its trained 262K context via RoPE extrapolation. vLLM allocated 581K tokens of KV cache from the available 44.47 GiB. No OOM — the scheduler queues requests that don't fit.

### Conclusions

1. **You cannot OOM vLLM through the API** — it validates input length before allocating KV cache
2. **Concurrent large requests queue**, they don't crash — vLLM serializes when KV cache is full
3. **The 262K context limit is soft** — with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, the model processes 300K+ tokens (quality may degrade beyond trained range)
4. **At 0.95 GPU memory utilization**, the system handles 1M context config with 581K token KV cache capacity
5. **vLLM's scheduler is the real safety net** — it never allocates more than available, just queues

## Vision / Multimodal Features

The model includes a vision encoder supporting:

| Capability | Status | Notes |
|-----------|--------|-------|
| Image description | Works | Detailed, nuanced descriptions |
| Object detection | Works | Provides bounding box coordinates |
| Object counting | Works | Accurate counting of people, objects |
| OCR / Text recognition | Works | Reads text from images (signs, documents) |
| Spatial reasoning | Works | Left/center/right positioning |
| Video understanding | Supported | Pass video URLs in messages |
| Chart / diagram analysis | Works | Requires raster images (not SVG) |

### Supported Image Formats

- JPEG, PNG, WebP (via URL or base64)
- **Not supported**: SVG, vector graphics

### Video Input

```json
{
  "role": "user",
  "content": [
    {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
    {"type": "text", "text": "What happens in this video?"}
  ]
}
```

## Troubleshooting

### "Model type qwen3_5_moe not recognized"

Your vLLM version is too old. The `Qwen3_5MoeForConditionalGeneration` architecture requires vLLM v0.16.0+. Use `vllm/vllm-openai:cu130-nightly` instead of the NVIDIA container.

### "Architectures not supported"

Same issue as above. The NVIDIA `nvcr.io/nvidia/vllm:26.01-py3` container ships with vLLM 0.13.0 which doesn't support Qwen3.5. Upgrading just `transformers` inside the container is not enough -- vLLM itself needs the model implementation.

### OOM after extended use

Reduce `--gpu-memory-utilization` from `0.90` to `0.80`. Community reports confirm `0.80` is stable for long-running sessions.

### "MoE config file not found" warning

```
WARNING: Using default MoE config. Performance might be sub-optimal!
Config file not found at .../E=256,N=512,device_name=NVIDIA_GB10.json
```

This is expected -- there is no optimized MoE kernel config for the GB10 GPU yet. The model still runs correctly with default settings.

### PyTorch CUDA capability warning

```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

This is a harmless warning. The GB10 works fine despite the version mismatch message.

### Prefix caching warning

```
Prefix caching in Mamba cache 'align' mode is currently enabled. Its support for Mamba layers is experimental.
```

This is informational. Prefix caching works and improves performance for repeated prompts. It can be disabled with `--no-enable-prefix-caching` if issues arise.

## Useful Commands

```bash
# Check if server is running
curl http://localhost:8000/v1/models

# View logs
docker logs qwen35 -f

# Stop the server
docker stop qwen35

# Start the server
docker start qwen35

# Remove and recreate
docker rm -f qwen35
# Then re-run the docker run command above

# Check GPU memory usage
nvidia-smi
```

## References

- [Qwen3.5-35B-A3B Model Card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.5 vLLM Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/index.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [DGX Spark vLLM Community Docker](https://github.com/eugr/spark-vllm-docker)
- [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)

## License

This guide is provided as-is under the [MIT License](LICENSE). The Qwen3.5-35B-A3B model itself is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
