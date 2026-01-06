# Environment Setup Guide

## OpenRouter API Configuration

### Step 1: Get Your API Key

1. Go to [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up or log in
3. Create a new API key
4. Copy the key (starts with `sk-or-v1-...`)

### Step 2: Configure Environment Variables

The API key is already set in `.env` file:

```env
OPENROUTER_API_KEY=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
```

**For production deployment**, update this key in `.env` file.

### Step 3: Choose Model

**Testing (Current - FREE):**
```env
USE_PRODUCTION_LLM=False
OPENROUTER_MODEL_TEST=mistralai/mistral-7b-instruct:free
```

**Production (Paid - Better Quality):**
```env
USE_PRODUCTION_LLM=True
OPENROUTER_MODEL_PROD=anthropic/claude-sonnet-4.5
```

### Step 4: Verify Installation

```bash
# Activate virtual environment
cd C:\Users\USER-PC\fluxpointai-backend\fluxpoint
..\venv\Scripts\Activate.ps1

# Test LLM integration
python -c "from strategy_builder.nlp.llm_parser import StrategyLLMParser; parser = StrategyLLMParser(); print('✅ API Key loaded successfully')"
```

### Step 5: Test Strategy Parsing

```python
from strategy_builder.nlp.llm_parser import parse_with_llm

description = "Buy when price crosses above 20 MA and RSI below 30"
result = parse_with_llm(description)
print(result)
```

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | *(required)* |
| `USE_PRODUCTION_LLM` | Use production model (True/False) | `False` |
| `OPENROUTER_MODEL_TEST` | Testing model (free) | `mistralai/mistral-7b-instruct:free` |
| `OPENROUTER_MODEL_PROD` | Production model (paid) | `anthropic/claude-sonnet-4.5` |
| `SITE_URL` | Your site URL for OpenRouter | `https://fluxpointai.com` |
| `SITE_NAME` | Your site name for OpenRouter | `FluxPoint AI Trading Platform` |

## Security Best Practices

### 1. Never Commit .env File

`.env` file is already in `.gitignore` to prevent accidental commits.

### 2. Use Different Keys for Development/Production

```bash
# Development
OPENROUTER_API_KEY=sk-or-v1-dev-key-here

# Production (on server)
OPENROUTER_API_KEY=sk-or-v1-prod-key-here
```

### 3. Set Environment Variables on Server

**For production servers**, set environment variables directly:

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
export USE_PRODUCTION_LLM="True"
```

**Windows:**
```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-your-key-here"
$env:USE_PRODUCTION_LLM="True"
```

**Docker:**
```yaml
environment:
  - OPENROUTER_API_KEY=sk-or-v1-your-key-here
  - USE_PRODUCTION_LLM=True
```

### 4. Rotate Keys Regularly

- Generate new keys every 3-6 months
- Immediately rotate if key is exposed
- Use separate keys for each environment

## Django Settings Integration

Add to `settings.py` for Django-wide access:

```python
from dotenv import load_dotenv
import os

load_dotenv()

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
USE_PRODUCTION_LLM = os.getenv('USE_PRODUCTION_LLM', 'False').lower() == 'true'
```

## Troubleshooting

### Error: "OPENROUTER_API_KEY not found"

**Solution:** Ensure `.env` file exists and contains the API key:
```bash
# Check if .env exists
ls .env

# View contents (be careful not to share this!)
cat .env
```

### Error: "Invalid API key"

**Solution:** 
1. Verify key on [OpenRouter dashboard](https://openrouter.ai/keys)
2. Check for extra spaces or quotes in `.env` file
3. Ensure key starts with `sk-or-v1-`

### Error: Rate limit exceeded

**Solution:**
1. Check usage on OpenRouter dashboard
2. Add credits if using paid models
3. Implement request caching

### LLM returns error or empty response

**Solution:**
1. Check OpenRouter status: [https://status.openrouter.ai](https://status.openrouter.ai)
2. Verify model name is correct
3. Check internet connection
4. Review request logs

## Cost Management

### Free Tier (Current)
- Model: `mistralai/mistral-7b-instruct:free`
- Cost: $0
- Rate limits: Generous for testing
- Quality: Good for simple strategies

### Paid Models (Production)
- Claude Sonnet 4.5: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- Average strategy parse: 500 tokens input, 300 tokens output
- Cost per parse: ~$0.001-0.005
- 10,000 parses: ~$10-50

### Optimization Tips

1. **Cache parsed strategies** - Don't re-parse identical descriptions
2. **Use free model first** - Only use Claude for complex strategies
3. **Batch requests** - Process multiple strategies in one request
4. **Set timeouts** - Prevent hanging requests
5. **Monitor usage** - Track costs on OpenRouter dashboard

## Testing

Run full test suite:

```bash
python strategy_builder/nlp/test_llm.py
```

Quick test:

```bash
python -c "from strategy_builder.nlp.llm_parser import parse_with_llm; print(parse_with_llm('Buy when RSI below 30', use_production=False))"
```

## Files Created

- [`.env`](.env) - Environment variables (DO NOT COMMIT)
- [`.env.example`](.env.example) - Template for other developers
- [`.gitignore`](.gitignore) - Prevents committing sensitive files
- [`SETUP_ENV.md`](SETUP_ENV.md) - This guide

## Next Steps

1. ✅ Environment configured
2. ✅ API key set
3. ✅ python-dotenv installed
4. ⏳ Test with `python strategy_builder/nlp/test_llm.py`
5. ⏳ Integrate with Django views
6. ⏳ Deploy to production with production API key
