# PR Code Review Assistant - Quick Start Guide

## Installation & Testing

### 1. Install Dependencies

```bash
# Option 1: Using pip
pip install openenv-core fastapi uvicorn pydantic requests openai

# Option 2: Using the package
pip install -e .

# Option 3: From requirements.txt
pip install -r requirements.txt
```

### 2. Run Basic Test

```bash
python3 test_basic.py
```

Expected output:
```
Testing PR Review Environment...
✓ Environment created
✓ Environment reset (task: task1_security_basic)
  PR ID: PR-001
  Files changed: 2
  Done: False
✓ Action created
✓ Step executed
  Score: 0.53
  Done: True
  Passed: False

Feedback:
  True Positives: 1
  False Positives: 0
  False Negatives: 2
  Precision: 1.00
  Recall: 0.33

✅ All tests passed!
```

### 3. Run the Server

```bash
# Start the FastAPI server
python3 -m uvicorn pr_review_env.server.app:app --host 0.0.0.0 --port 8000

# Or using the entry point
python3 -m pr_review_env.server.app
```

Test endpoints:
```bash
# Health check
curl http://localhost:8000/health

curl -sS "https://Saptak225-pr-review-openenv.hf.space/health"

# List tasks
curl http://localhost:8000/tasks

curl -sS "https://Saptak225-pr-review-openenv.hf.space/tasks"


# Get live baseline scores (runs inference; requires HF_TOKEN, MODEL_NAME)
curl http://localhost:8000/baseline
curl -sS "https://Saptak225-pr-review-openenv.hf.space/baseline"

# /grader endpoint that returns the grader score for the last completed episode
curl -sS "http://localhost:8000/grader"
# or on Space:
curl -sS "https://Saptak225-pr-review-openenv.hf.space/grader"

# Use your own AI/model
curl -sS "https://Saptak225-pr-review-openenv.hf.space/baseline?model=<THEIR_MODEL>&api_base_url=<THEIR_API_BASE_URL>&hf_token=<THEIR_HF_TOKEN>"

# (Optional) Pass secrets/config explicitly if your server/container doesn't already have them
curl "http://localhost:8000/baseline?hf_token=$HF_TOKEN&model=$MODEL_NAME&api_base_url=$API_BASE_URL"

# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_security_basic"}'
```

### 4. Run Inference Baseline (Optional)

```bash
# Set required variables (OpenAI-compatible client via HF router)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b"
export HF_TOKEN="your-hf-token"

# Run inference evaluation against local env server
python3 inference.py --env-url http://localhost:8000 --output inference_results.json
```

### 5. Build and Test Docker

```bash
# Build image
docker build -t pr-review-env .

# Run container
docker run -p 8000:8000 pr-review-env

# Test
curl http://localhost:8000/health
```

### 6. Validate OpenEnv Compliance

```bash
# Install openenv CLI
pip install openenv-core

# Validate local structure
openenv validate

# Validate running server
openenv validate --url http://localhost:8000
```

## Pre-Submission Checklist

- [ ] All dependencies installed
- [ ] `python3 test_basic.py` passes
- [ ] Server starts without errors
- [ ] All endpoints respond (health, tasks, baseline, reset, step)
- [ ] Docker builds successfully
- [ ] Docker container runs and responds
- [ ] openenv validate passes (local)
- [ ] openenv validate --url passes (runtime)

## Deployment to Hugging Face Spaces

### Quick Deploy

1. Create a new Space on Hugging Face (Docker SDK)
2. Clone this repository
3. Push to HF Space repository

```bash
git remote add hf https://huggingface.co/spaces/USERNAME/pr-review-env
git push hf main
```

### Verify Deployment

```bash
# Check if Space is running
curl https://USERNAME-pr-review-env.hf.space/health

# Validate remotely
openenv validate --url https://USERNAME-pr-review-env.hf.space
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, install dependencies:
```bash
pip install openenv-core pydantic fastapi uvicorn
```

### Port Already in Use

Change the port:
```bash
uvicorn pr_review_env.server.app:app --port 8001
```

### Docker Build Fails

Ensure you have Docker installed and running:
```bash
docker --version
docker info
```

## Next Steps

1. **Test locally** - Verify everything works on your machine
2. **Run baseline** - Get GPT-4 baseline scores
3. **Deploy to HF Spaces** - Make it publicly available
4. **Submit** - Provide the Space URL for judging

## Questions?

- Check README.md for full documentation
- Review OpenEnv docs: https://github.com/meta-pytorch/OpenEnv
- Open an issue if you find bugs

Good luck! 🚀
