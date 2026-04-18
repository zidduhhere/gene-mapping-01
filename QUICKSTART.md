# PharmaAI Predictor - Quick Start Guide

Get up and running with PharmaAI Predictor in minutes. This guide covers essential setup and execution steps.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation (5 minutes)](#installation-5-minutes)
3. [Running the Model](#running-the-model)
4. [Running the Backend API](#running-the-backend-api)
5. [Running the Frontend](#running-the-frontend)
6. [Making Predictions](#making-predictions)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.8+** (check: `python --version`)
- **Node.js 18+** (check: `node --version`)
- **npm 9+** (check: `npm --version`)
- **4GB+ RAM** (8GB+ recommended)
- **GPU optional** (NVIDIA CUDA for faster training)

---

## Installation (5 minutes)

### Step 1: Clone Repository

```bash
git clone https://github.com/zidduhhere/gene-mapping-01.git
cd PharmaAI-Predictor
```

### Step 2: Setup Python Backend

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; import streamlit; print('✓ Backend dependencies installed')"
```

### Step 3: Setup Node.js Frontend

```bash
cd frontend

# Install dependencies
npm install

# Verify installation
npm --version && node --version
echo "✓ Frontend dependencies installed"

cd ..
```

---

## Running the Model

### Option A: Quick Test (TOY Dataset - 5 minutes)

Perfect for first-time users to test everything works:

```bash
# Train on small toy dataset
python train_pharmaai.py --toy

# Output:
# ✓ Training complete!
# ✓ Model saved to: results/PharmaAI_TOY_test/TabTransformer/
# ✓ Ready for inference
```

**What this does:**
- Downloads small toy dataset (~100 samples)
- Trains TabTransformer for 2 CV folds
- Takes ~5-10 minutes on CPU, <5 min on GPU
- Creates model checkpoint: `results/PharmaAI_TOY_test/TabTransformer/model.pt`

### Option B: Full Training (GDSC2 - several hours)

For production-grade models:

```bash
# Train on GDSC2 (large dataset)
python train_pharmaai.py --dataset GDSC2 --n-cv-splits 5

# Output:
# ✓ Training complete!
# ✓ Model saved to: results/PharmaAI_Transformer_2025/TabTransformer/
# ✓ Metrics: R² = 0.78, RMSE = 0.62
```

**What this does:**
- Downloads GDSC2 pharmacogenomics dataset
- Trains with Leave-Pair-Out cross-validation
- 5 folds × 2-4 hours per fold = 10-20 hours total
- Generates comprehensive metrics and checkpoints
- Produces baselines (ElasticNet, RandomForest, NeuralNet)

### Option C: Training with Specific Options

```bash
# Skip baseline models (faster)
python train_pharmaai.py --dataset GDSC2 --no-baselines

# Cross-study validation (GDSC2 train, CCLE test)
python train_pharmaai.py --dataset GDSC2 --cross-study CCLE

# Different test mode (Leave-Cell-Line-Out)
python train_pharmaai.py --dataset GDSC2 --test-mode LCO

# Skip hyperparameter tuning (faster)
python train_pharmaai.py --dataset GDSC2 --no-hpam-tuning

# Combine options
python train_pharmaai.py --toy --no-baselines --no-hpam-tuning
```

### Generate Model Explanations (SHAP)

After training, generate feature importance visualizations:

```bash
# Generate SHAP explanations
python explain.py --model-dir results/PharmaAI_TOY_test/TabTransformer

# Output:
# ✓ Computed SHAP values
# ✓ Generated shap_summary_bar.png
# ✓ Generated shap_beeswarm.png
# ✓ Saved top_features.json

# Update model directory for GDSC2 if trained
python explain.py --model-dir results/PharmaAI_Transformer_2025/TabTransformer
```

---

## Running the Backend API

The FastAPI backend provides REST endpoints for predictions, explanations, and drug information. Run it before using the frontend.

### Start API Server (Uvicorn)

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
# Uvicorn: Started server process [PID]
```

**Server is ready when you see:** ✓ "Application startup complete"

### Verify API is Running

```bash
# In a new terminal, test the API
curl http://localhost:8000/docs

# Or open in browser:
# http://localhost:8000/docs        (Interactive Swagger UI)
# http://localhost:8000/redoc       (ReDoc documentation)
# http://localhost:8000/openapi.json (OpenAPI schema)
```

### API Endpoints

**Main Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | `POST` | Predict drug response (IC50) |
| `/explain` | `POST` | Get SHAP explanations for prediction |
| `/drugs` | `GET` | List available drugs |
| `/health` | `GET` | Check API health status |

**Example: Make a Prediction**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cell_line": "A375",
    "drug": "Erlotinib",
    "gene_expression": [0.5, 0.3, ...],
    "drug_fingerprint": [1, 0, 1, ...]
  }'

# Response:
# {
#   "prediction": 4.2,
#   "confidence": 0.87,
#   "model": "TabTransformer"
# }
```

### Production Deployment

```bash
# Start with Gunicorn (production-grade WSGI server)
gunicorn api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or use uvicorn with multiple workers
uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000

# Access at http://localhost:8000
```

### Custom Configuration

**Change API Port:**

```bash
# Run on different port (if 8000 is busy)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

# Update frontend .env.local:
# NEXT_PUBLIC_API_URL=http://localhost:8001
```

**Disable Auto-Reload (Production):**

```bash
# Remove --reload flag for production
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Set Log Level:**

```bash
# Reduce verbosity
uvicorn api.main:app --reload --log-level warning

# Options: critical, error, warning, info, debug
```

---

## Running the Frontend

### Start Development Server

```bash
# From project root
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Output:
# ▲ Next.js 16.2.1
# - Local:        http://localhost:3000
# - Environments: .env.local
#
# ✓ Ready in 2.5s
```

**Access the app:**
- Open browser to: `http://localhost:3000`
- App automatically reloads on code changes
- Open DevTools: `F12` or `Cmd+Option+I`

### Build for Production

```bash
cd frontend

# Build optimized production bundle
npm run build

# Start production server
npm start

# Server runs on http://localhost:3000
# Production-optimized (faster)
```

### Important Configuration

The frontend connects to the backend API. Update API endpoint if needed:

```bash
# Edit frontend/.env.local (create if doesn't exist)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Default assumes API is on localhost:8000
# Change if API runs on different host/port
```

---

## Complete Workflow: From Training to Frontend

### Scenario 1: Start Fresh (5-10 minutes)

```bash
# TERMINAL 1: Train the model
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train toy model (5 min)
python train_pharmaai.py --toy

# Generate explanations (5 min)
python explain.py --model-dir results/PharmaAI_TOY_test/TabTransformer

# TERMINAL 2: Start backend API
source venv/bin/activate  # Activate same venv
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Wait for: "Application startup complete"

# TERMINAL 3: Start frontend
cd frontend
npm install  # first time only
npm run dev

# 5. Open browser
# http://localhost:3000

# 6. Use the app to make predictions!
```

### Scenario 2: Use Pre-Trained Model (2 minutes)

If you already have a trained model:

```bash
# TERMINAL 1: Start backend API
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# TERMINAL 2: Start frontend
cd frontend
npm run dev

# Open browser: http://localhost:3000
# Frontend will load your trained model automatically
```

### Scenario 3: Full Pipeline with API (15 minutes)

```bash
# TERMINAL 1: Start training (if needed)
source venv/bin/activate
python train_pharmaai.py --toy

# TERMINAL 2: Start API server (after training completes)
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Runs on http://localhost:8000
# API docs: http://localhost:8000/docs

# TERMINAL 3: Start frontend
cd frontend
npm run dev
# Runs on http://localhost:3000

# Now you have:
# - Frontend: http://localhost:3000
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

---

## Making Predictions

### Via Web Interface (Recommended)

1. Open `http://localhost:3000`
2. Navigate to "Predict" tab
3. Choose input method:
   - **Upload CSV**: Select a `.csv` file with features (3026 columns)
   - **Manual Entry**: Paste comma-separated numbers
4. Click "Predict"
5. View results and download as CSV

### Via API (Command Line)

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, -0.3, 1.2, -0.1, ...(3026 total)]
  }'

# Response:
# {
#   "prediction": {
#     "ln_ic50": -0.4523,
#     "classification": "Sensitive",
#     "confidence": 0.87
#   }
# }
```

### Via Python

```python
from models.TabTransformer.tab_transformer import TabTransformer
import numpy as np

# Load model
model = TabTransformer.load("results/PharmaAI_TOY_test/TabTransformer")

# Create features (3026-dimensional)
features = np.random.randn(1, 3026).astype(np.float32)

# Predict
pred = model.predict(features)
print(f"LN_IC50: {pred[0]:.4f}")
```

---

## Quick Command Reference

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Training
python train_pharmaai.py --toy                    # Quick test
python train_pharmaai.py --dataset GDSC2          # Full training
python explain.py --model-dir <path>              # Generate explanations

# Backend API (Uvicorn)
uvicorn api.main:app --reload --port 8000        # Development
uvicorn api.main:app --host 0.0.0.0 --port 8000  # Production

# Frontend
cd frontend && npm install && npm run dev         # Start dev server
cd frontend && npm run build && npm start         # Production build

# Testing
curl http://localhost:3000                       # Test frontend
curl http://localhost:8000/docs                  # Test API (Swagger)
curl http://localhost:8000/health                # Test API health
```

---

## Directory Structure After Setup

```
PharmaAI-Predictor/
├── README.md                          # Full documentation
├── QUICKSTART.md                      # This file
├── ARCHITECTURE.md                    # Architecture diagrams
├── FLOW_DIAGRAM.md                   # Workflow diagrams
├── requirements.txt                   # Python dependencies
│
├── app.py                            # Legacy Streamlit app (optional)
├── train_pharmaai.py                 # Training script
├── explain.py                        # SHAP explanation script
│
├── frontend/                         # Next.js web interface
│   ├── package.json
│   ├── next.config.ts
│   ├── src/
│   ├── public/
│   └── .env.local                    # (create: API endpoint)
│
├── api/                              # FastAPI backend
│   ├── main.py                       # FastAPI app entry point
│   ├── routes/                       # API endpoints (predict, explain, drugs, auth)
│   └── services/                     # Business logic (drug, pdf generation)
│
├── models/                           # Model implementations
│   └── TabTransformer/
│
├── data/                             # Datasets (auto-created)
│   └── [GDSC, CCLE, etc]
│
└── results/                          # Trained models (auto-created)
    ├── PharmaAI_TOY_test/
    └── PharmaAI_Transformer_2025/
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Frontend won't load (Port 3000)

**Solution:**
```bash
# Port already in use? Use different port:
cd frontend
npm run dev -- --port 3001
# Open http://localhost:3001
```

### Issue: Model file not found during inference

**Solution:**
```bash
# Train first:
python train_pharmaai.py --toy

# Then verify model exists:
ls -la results/PharmaAI_TOY_test/TabTransformer/model.pt
```

### Issue: "CUDA out of memory" during training

**Solution:**
```bash
# Use CPU only:
# Edit line in train_pharmaai.py or use toy mode
python train_pharmaai.py --toy  # Smaller dataset

# Or reduce batch size (edit in code)
# Or skip baselines: --no-baselines
```

### Issue: API not responding from frontend

**Solution:**
```bash
# 1. Check .env.local in frontend/
cat frontend/.env.local
# Should have: NEXT_PUBLIC_API_URL=http://localhost:8000

# 2. Start API server with uvicorn:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Wait for: "Application startup complete"

# 3. Check API health:
curl http://localhost:8000/docs
# Should open interactive Swagger UI in browser
```

### Issue: "Uvicorn port already in use" (Port 8000)

**Solution:**
```bash
# Use different port:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

# Update frontend .env.local:
# NEXT_PUBLIC_API_URL=http://localhost:8001
```

### Issue: API crashes on startup

**Solution:**
```bash
# Check for missing dependencies
pip install --upgrade uvicorn fastapi

# Run with debug output
uvicorn api.main:app --reload --log-level debug

# Check specific error in output and install missing packages
```

### Issue: Frontend can't reach backend API

**Solution:**
```bash
# Make sure API is running in a separate terminal:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Verify API is listening:
curl -v http://localhost:8000/docs

# Check frontend .env.local has correct API URL:
cat frontend/.env.local

# If URL is wrong, update it and restart frontend:
# NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Issue: "npm: command not found"

**Solution:**
```bash
# Install Node.js from https://nodejs.org/
# Download LTS version (18+)
# Then verify:
node --version
npm --version
```

---

## Next Steps

After getting everything running:

1. **Explore the Frontend**: Make predictions, view explanations
2. **Check Performance**: Review metrics and evaluation results
3. **Customize Model**: Adjust hyperparameters in training scripts
4. **Deploy**: See ARCHITECTURE.md for production deployment options
5. **Contribute**: See Contributing section in README.md

---

## Support & Documentation

- **Full Documentation**: See `README.md`
- **Architecture Details**: See `ARCHITECTURE.md`
- **Flow Diagrams**: See `FLOW_DIAGRAM.md`
- **API Documentation**: Run backend and visit `http://localhost:8000/docs` (Swagger UI)
  ```bash
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
  ```
- **Issues**: Create GitHub issue with error message and reproduction steps

---

## What's Next?

✅ **Basic Setup Complete!**

Ready to explore? Here are some fun experiments:

1. **Train on Different Dataset**
   ```bash
   python train_pharmaai.py --dataset CCLE
   ```

2. **Compare Models**
   ```bash
   python train_pharmaai.py --toy --no-baselines
   # vs
   python train_pharmaai.py --toy  # with baselines
   ```

3. **Analyze Features**
   ```bash
   python explain.py --model-dir results/...
   # View shap_results/shap_summary_bar.png
   ```

4. **Use API for Automation**
   - Write Python/JavaScript scripts using `/predict` endpoint
   - Batch process multiple samples
   - Integrate with other systems

---

**Happy predicting! 🚀**

Questions? See `README.md` or create an issue on GitHub.

---

**Last Updated**: April 2025  
**Version**: 1.0.0
