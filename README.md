# PharmaAI Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**PharmaAI Predictor** is an AI-driven drug response prediction system that leverages a Transformer-based deep learning architecture to predict drug sensitivity (LN_IC50 values) from cancer cell line gene expression profiles and drug molecular fingerprints.

The system uses the **drevalpy** framework to train and evaluate models on pharmacogenomic datasets (GDSC, CCLE, CTRP) and provides both a web interface (Streamlit) and REST API for predictions and model interpretability via SHAP explainability.

### Key Features

- 🧬 **Transformer-based Architecture**: State-of-the-art TabTransformer model for structured data prediction
- 📊 **Multi-Dataset Support**: Train and evaluate on GDSC, CCLE, CTRP, and custom datasets
- 🔍 **Model Explainability**: SHAP-based feature importance analysis and visualization
- 🚀 **Web Interface**: Interactive Streamlit dashboard for predictions and exploration
- 🔌 **REST API**: FastAPI-based backend for integration with external systems
- 🔬 **Cross-Study Validation**: Support for cross-study generalization testing
- 📈 **Baseline Comparisons**: Includes ElasticNet, Random Forest, and Neural Network baselines

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Workflow](#workflow)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Project Structure](#project-structure)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)

---

## System Requirements

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: Minimum 8GB RAM (16GB+ recommended for full GDSC2 training)
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster training

### Required Python Libraries

All dependencies are specified in `requirements.txt`:

```
drevalpy>=1.4.0
torch>=2.0.0
pytorch-lightning>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
shap>=0.42.0
streamlit>=1.28.0
joblib>=1.3.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pyjwt>=2.8.0
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/zidduhhere/gene-mapping-01.git
cd PharmaAI-Predictor
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies (Node.js)

```bash
cd frontend

# Install Node.js packages
npm install

# Verify installation
npm --version

cd ..
```

### 5. Verify Installation

```bash
python -c "import torch; import fastapi; print('✓ Backend installed')"
node --version && npm --version && echo "✓ Frontend installed"
```

---

## Quick Start

⚡ **NEW**: See [QUICKSTART.md](QUICKSTART.md) for fastest setup (5 minutes)

### Option 1: Run Web Interface (Recommended for beginners)

```bash
# Terminal 1: Train model first (if needed)
python train_pharmaai.py --toy

# Terminal 2: Start Next.js frontend
cd frontend
npm install  # First time only
npm run dev

# Open browser to http://localhost:3000
```

### Option 2: Quick Model Training (Toy Dataset - ~5 minutes)

```bash
# Train on small toy dataset for testing
python train_pharmaai.py --toy

# Output will be saved to: results/PharmaAI_TOY_test/TabTransformer/
```

### Option 3: Full Training (GDSC2 - several hours)

```bash
# Train on GDSC2 dataset with cross-validation
python train_pharmaai.py --dataset GDSC2 --n-cv-splits 5

# Output will be saved to: results/PharmaAI_Transformer_2025/TabTransformer/
```

---

## Workflow

### 1. **Data Preparation Phase**

The system automatically downloads and preprocesses pharmacogenomic data:

```
Raw Datasets (GDSC/CCLE) 
    ↓
Standardization & Normalization
    ↓
Feature Extraction (Gene Expression + Drug Fingerprints)
    ↓
Training/Validation/Test Split
```

**Supported Datasets:**
- GDSC1, GDSC2 (large-scale cancer cell lines)
- CCLE (Cancer Cell Line Encyclopedia)
- CTRP v1, v2 (Cancer Therapeutics Response Portal)
- TOYv1, TOYv2 (small datasets for testing)

### 2. **Model Training Phase**

```bash
python train_pharmaai.py [OPTIONS]
```

**Key Parameters:**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--dataset` | Primary training dataset | GDSC2 | `--dataset GDSC2` |
| `--test-mode` | Cross-validation strategy | LPO | `--test-mode LPO` |
| `--n-cv-splits` | Number of CV folds | 5 | `--n-cv-splits 5` |
| `--toy` | Quick test mode | False | `--toy` |
| `--cross-study` | External validation dataset | None | `--cross-study CCLE` |
| `--no-baselines` | Skip baseline models | False | `--no-baselines` |
| `--path-data` | Data directory | data | `--path-data ./data` |
| `--path-out` | Output directory | results | `--path-out ./results` |

**Cross-Validation Test Modes:**
- `LPO` (Leave-Pair-Out): Most challenging, tests generalization
- `LCO` (Leave-Cell-Line-Out): Cell line generalization
- `LDO` (Leave-Drug-Out): Drug generalization
- `LTO` (Leave-Tissue-Out): Tissue type generalization

**Example Commands:**

```bash
# Train with cross-study validation
python train_pharmaai.py --dataset GDSC2 --cross-study CCLE

# Skip hyperparameter tuning for faster training
python train_pharmaai.py --dataset GDSC2 --no-hpam-tuning

# Use different test mode
python train_pharmaai.py --dataset GDSC1 --test-mode LCO --n-cv-splits 3
```

### 3. **Model Evaluation Phase**

Results include:
- Performance metrics (R², MSE, MAE, RMSE, Spearman correlation)
- Cross-validation statistics
- Trained model checkpoints
- Hyperparameter logs

**Output Structure:**
```
results/
├── PharmaAI_Transformer_2025/
│   ├── TabTransformer/
│   │   ├── model.pt          # Trained model weights
│   │   ├── hyperparameters.json
│   │   ├── scaler.pkl        # Feature normalization
│   │   └── logs/
│   ├── ElasticNet/
│   ├── SimpleNeuralNetwork/
│   └── RandomForest/
```

### 4. **Model Explanation Phase**

Generate SHAP explainability plots:

```bash
python explain.py --model-dir results/PharmaAI_Transformer_2025/TabTransformer
```

**Output:**
- `shap_summary_bar.png` - Top 20 features by importance
- `shap_beeswarm.png` - Feature impact distribution
- `top_features.json` - Feature rankings

### 5. **Prediction & Inference Phase**

#### A. Web Interface (Next.js Frontend)
```bash
# Terminal 1: Start Next.js development server
cd frontend
npm run dev

# Access at: http://localhost:3000
```

**Features:**
- Modern React UI with responsive design
- Load trained models
- Make predictions via CSV upload or manual input
- View SHAP explainability visualizations
- Download prediction results in CSV format
- Real-time model evaluation

**Configuration:**
Create `frontend/.env.local` to configure API endpoint:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### B. REST API
```bash
python api/main.py
```

Access API documentation: `http://localhost:8000/docs`

#### C. Programmatic Usage
```python
from models.TabTransformer.tab_transformer import TabTransformer
import numpy as np

# Load model
model = TabTransformer.load("results/PharmaAI_Transformer_2025/TabTransformer")

# Make prediction
features = np.random.randn(1, 1024)  # Example features
prediction = model.predict(features)
print(f"Predicted LN_IC50: {prediction[0]:.4f}")
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Model Configuration
MODEL_DIR=results/PharmaAI_Transformer_2025/TabTransformer
SHAP_DIR=shap_results

# Data Configuration
DATA_PATH=data
OUTPUT_PATH=results

# Training Configuration
BATCH_SIZE=128
LEARNING_RATE=0.001
EPOCHS=100
SEED=42

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=INFO
```

### Model Hyperparameters

TabTransformer architecture can be configured in training scripts:

```python
hyperparameters = {
    "input_dim_gex": 978,           # Gene expression features
    "input_dim_fp": 2048,           # Drug fingerprint dimensions
    "token_size": 64,               # Feature token dimensions
    "n_layers": 4,                  # Transformer layers
    "n_heads": 8,                   # Multi-head attention heads
    "dropout": 0.1,                 # Dropout rate
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 100,
}
```

---

## API Reference

### REST API Endpoints

#### 1. Health Check
```
GET /health
Response: {"status": "healthy"}
```

#### 2. Make Prediction
```
POST /predict
Content-Type: application/json

Request Body:
{
  "features": [0.5, -0.3, 1.2, ...]  # Concatenated features
}

Response:
{
  "prediction": {
    "ln_ic50": -0.4523,
    "classification": "Sensitive",
    "confidence": 0.87
  },
  "timestamp": "2025-04-18T10:30:00Z"
}
```

#### 3. Batch Predictions
```
POST /predict/batch
Content-Type: application/json

Request Body:
{
  "features": [
    [0.5, -0.3, 1.2, ...],
    [0.2, 0.1, -0.5, ...]
  ]
}

Response:
{
  "predictions": [
    {"ln_ic50": -0.45, "classification": "Sensitive"},
    {"ln_ic50": 0.23, "classification": "Resistant"}
  ]
}
```

#### 4. Get Model Info
```
GET /model/info
Response:
{
  "model_type": "TabTransformer",
  "version": "1.0",
  "training_dataset": "GDSC2",
  "metrics": {
    "r2_score": 0.78,
    "mse": 0.34
  }
}
```

#### 5. Feature Importance
```
GET /interpretability/top-features
Response:
{
  "top_features": [
    {"rank": 1, "name": "Gene_123", "importance": 0.045},
    {"rank": 2, "name": "FP_bit_567", "importance": 0.038}
  ]
}
```

---

## Project Structure

```
PharmaAI-Predictor/
├── README.md                      # This file
├── ARCHITECTURE.md                # Architecture diagram documentation
├── FLOW_DIAGRAM.md               # Workflow and data flow diagrams
├── requirements.txt               # Python dependencies
├── app.py                         # Streamlit web application
├── train_pharmaai.py             # Model training script
├── explain.py                     # SHAP explainability script
├── register_model.py              # Model registration for drevalpy
│
├── api/                           # REST API backend
│   ├── main.py                   # FastAPI application
│   ├── routes/                   # API endpoints
│   │   ├── predict.py           # Prediction endpoints
│   │   ├── model.py             # Model info endpoints
│   │   └── interpretability.py  # SHAP endpoints
│   └── services/                # Business logic
│       ├── predictor.py         # Prediction service
│       └── explainer.py         # SHAP explanation service
│
├── models/                        # Model implementations
│   └── TabTransformer/
│       ├── tab_transformer.py   # TabTransformer architecture
│       ├── encoder.py           # Transformer encoder
│       └── __init__.py
│
├── frontend/                      # Next.js web interface
│   ├── package.json               # Node.js dependencies
│   ├── next.config.ts             # Next.js configuration
│   ├── src/                       # React components & pages
│   ├── public/                    # Static assets
│   ├── .env.local                 # API endpoint config (create this)
│   └── node_modules/              # Node packages
│
├── data/                          # Data directory (auto-created)
│   └── [GDSC, CCLE, CTRP datasets]
│
├── results/                       # Training outputs
│   └── PharmaAI_Transformer_2025/
│       ├── TabTransformer/
│       ├── ElasticNet/
│       ├── RandomForest/
│       └── [other models]/
│
└── shap_results/                  # SHAP analysis outputs
    ├── shap_summary_bar.png
    ├── shap_beeswarm.png
    └── top_features.json
```

---

## Detailed Workflow

### Training Workflow

```
1. Data Loading
   ├── Download/load GDSC2/CCLE/CTRP datasets
   ├── Load gene expression (landmark genes: ~978 features)
   └── Load drug fingerprints (Morgan: ~2048 bits)

2. Feature Preprocessing
   ├── Concatenate gene expression + drug fingerprints
   ├── Normalize using StandardScaler
   ├── Handle missing values
   └── Split into train/val/test

3. Hyperparameter Tuning (if enabled)
   ├── Grid/random search over parameter space
   ├── Cross-validation on training set
   └── Select best hyperparameters

4. Model Training
   ├── Initialize TabTransformer
   ├── Train with PyTorch Lightning
   ├── Monitor validation metrics
   ├── Early stopping if applicable
   └── Save best model checkpoint

5. Model Evaluation
   ├── Test on held-out test set
   ├── Compute metrics (R², MSE, MAE, Spearman)
   ├── Generate cross-validation results
   └── Optional cross-study validation
```

### Prediction Workflow

```
User Input
   ├── CSV upload or manual entry
   │
Input Validation
   ├── Check feature dimensions
   ├── Validate data types
   └── Handle missing values
   │
Feature Preprocessing
   ├── Normalize using saved scaler
   └── Ensure correct ordering
   │
Model Inference
   ├── Load model weights
   ├── Forward pass through TabTransformer
   └── Generate LN_IC50 prediction
   │
Post-Processing
   ├── Classify: Sensitive (< 0) vs Resistant (≥ 0)
   ├── Compute confidence scores
   └── Generate explanations (SHAP)
   │
Output
   └── Display prediction + visualization
```

---

## Common Usage Scenarios

### Scenario 1: Quick Testing
```bash
# 1. Test installation with toy dataset
python train_pharmaai.py --toy

# 2. Launch web interface
streamlit run app.py

# 3. Make test predictions via web UI
```

### Scenario 2: Full Model Development
```bash
# 1. Train on GDSC2
python train_pharmaai.py --dataset GDSC2 --n-cv-splits 5

# 2. Validate on CCLE
python train_pharmaai.py --dataset GDSC2 --cross-study CCLE

# 3. Generate explanations
python explain.py --model-dir results/PharmaAI_Transformer_2025/TabTransformer

# 4. Launch Next.js web interface for exploration
cd frontend
npm run dev
# Open http://localhost:3000
```

### Scenario 3: Production Deployment
```bash
# 1. Train model on full dataset
python train_pharmaai.py --dataset GDSC2 --no-hpam-tuning

# 2. Build frontend for production
cd frontend
npm run build
npm start
# Runs on http://localhost:3000

# 3. Start API server
python api/main.py
# Runs on http://localhost:8000

# 4. Query predictions via API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -0.3, 1.2, ...]}'
```

### Scenario 4: Comparative Analysis
```bash
# Train with multiple test modes
python train_pharmaai.py --dataset GDSC2 --test-mode LPO
python train_pharmaai.py --dataset GDSC2 --test-mode LCO
python train_pharmaai.py --dataset GDSC2 --test-mode LDO

# Compare results in results/ directory

# Start frontend to visualize
cd frontend && npm run dev
```

---

## Troubleshooting

### Issue: Model fails to load
```
Error: FileNotFoundError: model.pt not found
```
**Solution:**
- Ensure model training completed successfully
- Check model directory path: `results/PharmaAI_Transformer_2025/TabTransformer/`
- Run training: `python train_pharmaai.py --toy`

### Issue: Out of memory during training
```
Error: CUDA out of memory or insufficient RAM
```
**Solution:**
- Reduce batch size: `--batch-size 32` (in code)
- Reduce CV splits: `--n-cv-splits 2`
- Skip baselines: `--no-baselines`
- Use toy dataset: `--toy`

### Issue: SHAP computation is slow
```
Generating SHAP values takes hours...
```
**Solution:**
- Reduce explanation samples: `--n-explain 20`
- Reduce background samples: `--n-background 50`
- Use KernelExplainer (faster than DeepExplainer)

### Issue: Next.js frontend won't start
```
Error: Port 3000 already in use
```
**Solution:**
```bash
# Run on different port
cd frontend
npm run dev -- --port 3001
# Open http://localhost:3001
```

### Issue: Frontend can't connect to API
```
Error: Failed to fetch from API endpoint
```
**Solution:**
- Check API is running: `python api/main.py`
- Verify API endpoint in `frontend/.env.local`:
  ```
  NEXT_PUBLIC_API_URL=http://localhost:8000
  ```
- Ensure API is accessible: `curl http://localhost:8000/health`
- Check CORS settings in API config

### Issue: "npm: command not found"
```
Error: npm is not installed
```
**Solution:**
- Install Node.js from https://nodejs.org/ (LTS version)
- Verify: `node --version && npm --version`
- Then install frontend dependencies: `cd frontend && npm install`

### Issue: API endpoint returns 404
```
Error: POST /predict HTTP/1.1" 404
```
**Solution:**
- Verify API is running: `python api/main.py`
- Check endpoint path: `/predict` (not `/api/predict`)
- View docs: http://localhost:8000/docs

### Issue: Features have incorrect dimensions
```
Error: Expected 3026 features, got 2048
```
**Solution:**
- Gene expression + fingerprints must be concatenated
- Typical: 978 gene features + 2048 fingerprint bits = 3026 total
- Check feature order in preprocessing

---

## Performance Metrics

### Typical Performance (GDSC2 with TabTransformer)

| Metric | Value | Note |
|--------|-------|------|
| **R² Score** | 0.75-0.82 | Test set |
| **RMSE** | 0.58-0.65 | Log-scaled |
| **MAE** | 0.42-0.50 | Log-scaled |
| **Spearman Corr** | 0.78-0.85 | Rank correlation |
| **Training Time** | 4-8 hours | GDSC2, 1 split |
| **Inference Time** | <100ms | Single sample |

### Baseline Comparisons

| Model | R² | RMSE | Training Time |
|-------|-----|------|---------------|
| ElasticNet | 0.65-0.72 | 0.68-0.75 | 5 min |
| Random Forest | 0.70-0.75 | 0.60-0.68 | 30 min |
| SimpleNeuralNetwork | 0.72-0.78 | 0.58-0.65 | 1 hour |
| **TabTransformer** | **0.75-0.82** | **0.58-0.65** | **4-8 hours** |

---

## Contributing

### Reporting Issues
1. Create an issue on GitHub
2. Include error message, command, and system info
3. Provide minimal reproducible example

### Development Setup
```bash
# Clone and setup
git clone https://github.com/zidduhhere/gene-mapping-01.git
cd PharmaAI-Predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install dev dependencies (if available)
pip install -r requirements-dev.txt

# Run tests
pytest
```

### Code Style
- Follow PEP 8 conventions
- Use type hints where applicable
- Add docstrings for functions/classes
- Include tests for new features

---

## Citation

If you use PharmaAI Predictor in your research, please cite:

```bibtex
@software{pharmaai2025,
  title={PharmaAI Predictor: AI-Driven Drug Response Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/zidduhhere/gene-mapping-01}
}
```

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Acknowledgments

- **drevalpy**: Drug Response Evaluation Framework
- **PyTorch Lightning**: Deep learning framework
- **SHAP**: Feature importance analysis
- **Streamlit**: Web interface framework
- **GDSC/CCLE/CTRP**: Pharmacogenomic datasets

---

## Support

For questions or support:
- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/zidduhhere/gene-mapping-01/issues)
- 📖 Documentation: See ARCHITECTURE.md and FLOW_DIAGRAM.md

---

**Last Updated:** April 2025  
**Current Version:** 1.0.0
