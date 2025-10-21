# OpenAI-Compatible Translation Backends

## Overview

The LiveTranslate Translation Service now supports **any OpenAI-compatible API endpoint** for translation! This gives you maximum flexibility to choose between:

- **Local services** (Ollama, vLLM, LM Studio) - 100% private, no API costs
- **Cloud services** (Groq, Together AI, OpenAI) - High performance, pay-per-use
- **Self-hosted** (your own vLLM server) - Full control, enterprise-grade

All these services use the same OpenAI API format, so you can easily switch between them or use multiple backends simultaneously.

## Supported Services

### 🏠 Local Services (Free & Private)

#### Ollama
**Recommendation**: ⭐⭐⭐⭐⭐ Best for local deployment

- ✅ **Free** and **100% private**
- ✅ Easy to install and use
- ✅ Supports many models (Llama, Mistral, Phi, etc.)
- ✅ Automatic model management
- ✅ GPU acceleration with CPU fallback

**Setup**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download a model
ollama pull llama3.1:8b

# Start the server (runs automatically)
ollama serve

# Enable in translation service
export OLLAMA_ENABLE=true
export OLLAMA_MODEL=llama3.1:8b
```

**Models**:
- `llama3.1:8b` - Good balance (4.7GB)
- `llama3.1:70b` - Highest quality (40GB)
- `mistral:7b` - Fast and efficient (4.1GB)
- `phi3:latest` - Small and fast (2.3GB)

---

#### vLLM Server
**Recommendation**: ⭐⭐⭐⭐ Best for high-performance self-hosting

- ✅ **Extremely fast** inference
- ✅ Continuous batching for high throughput
- ✅ Multi-GPU support
- ❌ Requires more setup

**Setup**:
```bash
# Install vLLM
pip install vllm

# Start OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --dtype auto

# Enable in translation service
export VLLM_SERVER_ENABLE=true
export VLLM_SERVER_BASE_URL=http://localhost:8000/v1
export VLLM_SERVER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
```

---

#### LM Studio
**Recommendation**: ⭐⭐⭐⭐ Best for desktop users

- ✅ Beautiful GUI interface
- ✅ Easy model management
- ✅ Built-in OpenAI-compatible server
- ❌ Desktop-only (not for servers)

**Setup**:
1. Download from https://lmstudio.ai/
2. Download a model in the GUI
3. Start the local server (Settings → Local Server)
4. Configure translation service:

```bash
export CUSTOM_OPENAI_ENABLE=true
export CUSTOM_OPENAI_NAME=lmstudio
export CUSTOM_OPENAI_BASE_URL=http://localhost:1234/v1
export CUSTOM_OPENAI_MODEL=llama-3.1-8b-instruct
```

---

### ☁️ Cloud Services (Fast & Scalable)

#### Groq
**Recommendation**: ⭐⭐⭐⭐⭐ Best for cloud deployment

- ✅ **Ultra-fast** inference (500+ tokens/sec)
- ✅ **Generous free tier** (14,400 requests/day)
- ✅ Simple API key authentication
- ✅ Multiple models available

**Setup**:
```bash
# Get API key from https://console.groq.com/

export GROQ_ENABLE=true
export GROQ_API_KEY=gsk_your_api_key_here
export GROQ_MODEL=llama-3.1-8b-instant
```

**Models**:
- `llama-3.1-8b-instant` - Ultra-fast, free tier
- `llama-3.1-70b-versatile` - Higher quality
- `mixtral-8x7b-32768` - Long context (32K tokens)
- `gemma2-9b-it` - Google's Gemma model

**Free Tier**: 14,400 requests/day with `llama-3.1-8b-instant`

---

#### Together AI
**Recommendation**: ⭐⭐⭐⭐ Best model variety

- ✅ **Wide selection** of models
- ✅ Competitive pricing
- ✅ Good performance
- ❌ No free tier

**Setup**:
```bash
# Get API key from https://api.together.xyz/

export TOGETHER_ENABLE=true
export TOGETHER_API_KEY=your_api_key_here
export TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
```

**Models**:
- `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` - Fast
- `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` - High quality
- `mistralai/Mixtral-8x7B-Instruct-v0.1` - Balanced
- `Qwen/Qwen2-72B-Instruct` - Chinese-optimized

**Pricing**: ~$0.0002/1K tokens (very cheap)

---

#### OpenAI
**Recommendation**: ⭐⭐⭐ Best quality, highest cost

- ✅ **Highest quality** translations
- ✅ Most reliable service
- ✅ Best multilingual support
- ❌ Most expensive
- ❌ Data sent to OpenAI

**Setup**:
```bash
# Get API key from https://platform.openai.com/

export OPENAI_ENABLE=true
export OPENAI_API_KEY=sk-your_api_key_here
export OPENAI_MODEL=gpt-4o-mini
```

**Models**:
- `gpt-4o-mini` - Good balance ($0.15/1M input tokens)
- `gpt-4o` - Best quality ($2.50/1M input tokens)
- `gpt-3.5-turbo` - Cheapest ($0.50/1M input tokens)

---

### 🌐 Other OpenAI-Compatible Services

All of these work with the **Custom OpenAI** configuration:

#### OpenRouter
Aggregates 100+ models from different providers.
```bash
export CUSTOM_OPENAI_ENABLE=true
export CUSTOM_OPENAI_NAME=openrouter
export CUSTOM_OPENAI_BASE_URL=https://openrouter.ai/api/v1
export CUSTOM_OPENAI_API_KEY=your_openrouter_key
export CUSTOM_OPENAI_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

#### Replicate
Run models on-demand with pay-per-use.
```bash
export CUSTOM_OPENAI_ENABLE=true
export CUSTOM_OPENAI_NAME=replicate
export CUSTOM_OPENAI_BASE_URL=https://api.replicate.com/v1
export CUSTOM_OPENAI_API_KEY=your_replicate_token
export CUSTOM_OPENAI_MODEL=meta/meta-llama-3.1-8b-instruct
```

#### LocalAI
OpenAI alternative that runs locally.
```bash
# Install LocalAI and start server
export CUSTOM_OPENAI_ENABLE=true
export CUSTOM_OPENAI_BASE_URL=http://localhost:8080/v1
export CUSTOM_OPENAI_MODEL=llama-3.1-8b
```

---

## Configuration Guide

### Single Backend (Simple)

Choose ONE backend that fits your needs:

```bash
# Local Ollama (free, private)
OLLAMA_ENABLE=true
OLLAMA_MODEL=llama3.1:8b
```

OR

```bash
# Groq Cloud (fast, free tier)
GROQ_ENABLE=true
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
```

### Multiple Backends (Recommended)

Enable multiple backends for **redundancy** and **automatic fallback**:

```bash
# Primary: Local Ollama (free, private)
OLLAMA_ENABLE=true
OLLAMA_MODEL=llama3.1:8b

# Fallback: Groq Cloud (when Ollama unavailable)
GROQ_ENABLE=true
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
```

**Priority order** (auto-selected when `model=auto`):
1. Ollama (local)
2. Groq (cloud)
3. Together AI (cloud)
4. vLLM Server (self-hosted)
5. OpenAI (cloud)
6. Custom (any)

### Model-Specific Translation

Users can specify which backend to use:

```bash
curl -X POST http://localhost:5003/api/translate/multi \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "target_languages": ["es", "fr"],
    "model": "groq"
  }'
```

Available model values:
- `"auto"` - Auto-select best available backend (default)
- `"ollama"` - Use Ollama
- `"groq"` - Use Groq
- `"together"` - Use Together AI
- `"openai"` - Use OpenAI
- `"vllm_server"` - Use vLLM server
- `"custom"` - Use custom endpoint
- `"llama"` - Use local Llama transformers (legacy)
- `"nllb"` - Use NLLB transformers (legacy)

---

## API Examples

### Check Available Models

```bash
curl http://localhost:5003/api/models/available
```

Response:
```json
{
  "models": [
    {
      "name": "ollama",
      "display_name": "Ollama (Local)",
      "available": true,
      "description": "Local Ollama server - private and free",
      "backend": "openai_compatible",
      "endpoint": "http://localhost:11434/v1",
      "model": "llama3.1:8b",
      "supported_languages": ["en", "es", "fr", "de", ...]
    },
    {
      "name": "groq",
      "display_name": "Groq (Cloud - Fast)",
      "available": true,
      "description": "Groq cloud inference - ultra-fast with free tier",
      "backend": "openai_compatible",
      "endpoint": "https://api.groq.com/openai/v1",
      "model": "llama-3.1-8b-instant"
    }
  ],
  "default": "auto"
}
```

### Translate with Auto-Selection

```bash
curl -X POST http://localhost:5003/api/translate/multi \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning, how are you?",
    "target_languages": ["es", "fr", "de"],
    "model": "auto"
  }'
```

### Translate with Specific Backend

```bash
curl -X POST http://localhost:5003/api/translate/multi \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning",
    "target_languages": ["es"],
    "model": "ollama"
  }'
```

---

## Recommended Configurations

### Local Development
**Goal**: Free, private, no internet required

```bash
OLLAMA_ENABLE=true
OLLAMA_MODEL=llama3.1:8b
```

**Why**: 100% local, no API costs, good quality.

---

### Production (Cost-Optimized)
**Goal**: Minimize costs while maintaining reliability

```bash
# Primary: Ollama (local, free)
OLLAMA_ENABLE=true
OLLAMA_MODEL=llama3.1:8b

# Fallback: Groq (cloud, free tier)
GROQ_ENABLE=true
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
```

**Why**: Use local when possible, fallback to Groq's free tier when needed.

---

### Production (Performance)
**Goal**: Maximum speed and reliability

```bash
# Primary: Groq (ultra-fast)
GROQ_ENABLE=true
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-70b-versatile

# Fallback: Together AI
TOGETHER_ENABLE=true
TOGETHER_API_KEY=your_together_key
TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
```

**Why**: Cloud services for reliability, Groq for speed, Together AI for redundancy.

---

### Production (Quality)
**Goal**: Highest quality translations

```bash
# Primary: OpenAI (best quality)
OPENAI_ENABLE=true
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o

# Fallback: Groq (fast and good)
GROQ_ENABLE=true
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-70b-versatile
```

**Why**: OpenAI for best quality, Groq for cost-effective fallback.

---

### Enterprise (Self-Hosted)
**Goal**: Full control, data privacy, high performance

```bash
# Primary: vLLM Server (self-hosted)
VLLM_SERVER_ENABLE=true
VLLM_SERVER_BASE_URL=http://vllm-cluster.internal:8000/v1
VLLM_SERVER_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct

# Fallback: Local Ollama
OLLAMA_ENABLE=true
OLLAMA_BASE_URL=http://ollama.internal:11434/v1
OLLAMA_MODEL=llama3.1:8b
```

**Why**: Keep all data in-house, high performance with vLLM, local fallback.

---

## Performance Comparison

| Service | Speed | Quality | Cost | Privacy | Free Tier |
|---------|-------|---------|------|---------|-----------|
| Ollama | ⚡⚡⚡ | ⭐⭐⭐⭐ | FREE | ✅ 100% | ✅ Unlimited |
| Groq | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | FREE* | ❌ Cloud | ✅ 14.4K/day |
| Together AI | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | $ | ❌ Cloud | ❌ No |
| OpenAI | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | $$$ | ❌ Cloud | ❌ No |
| vLLM Server | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | FREE | ✅ 100% | ✅ Unlimited |

*Groq free tier limited to 14,400 requests/day

---

## Troubleshooting

### Ollama not connecting

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Test with a simple request
ollama run llama3.1:8b "Translate 'hello' to Spanish"
```

### Groq API errors

```bash
# Verify API key
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"

# Check rate limits
# Free tier: 14,400 requests/day, 30 requests/minute
```

### No backends available

```bash
# Check service logs
tail -f /var/log/translation-service.log

# Verify environment variables
env | grep -E "OLLAMA|GROQ|TOGETHER|OPENAI"

# Check available models
curl http://localhost:5003/api/models/available
```

---

## Migration from Legacy Backends

If you're using the old local transformers backends (vLLM internal, NLLB, Llama transformers), you can easily migrate:

### Old Configuration (Legacy)
```bash
TRANSLATION_MODEL=./models/Llama-3.1-8B-Instruct
GPU_ENABLE=true
```

### New Configuration (OpenAI-Compatible)
```bash
# Use Ollama instead
OLLAMA_ENABLE=true
OLLAMA_MODEL=llama3.1:8b

# Or use vLLM server
VLLM_SERVER_ENABLE=true
VLLM_SERVER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Benefits**:
- ✅ Easier to set up
- ✅ Better performance
- ✅ Automatic model management
- ✅ Same API across all providers
- ✅ Easy to switch between providers

---

## Next Steps

1. **Choose your backend** based on your needs (local vs cloud, cost vs quality)
2. **Copy `.env.example`** to `.env` and configure
3. **Test with curl** to verify it's working
4. **Configure caching** for even better performance (see main README)

For questions or issues, see the main [Translation Service Documentation](./CLAUDE.md).
