# PharmaAI Predictor - Architecture Documentation

## System Architecture Overview

This document provides comprehensive architecture diagrams and documentation for the PharmaAI Predictor system.

---

## 1. High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         PharmaAI PREDICTOR SYSTEM                              │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────┐    ┌──────────────────────────────────┐ │
│  │      USER INTERFACES             │    │   DATA & MODEL SOURCES           │ │
│  ├──────────────────────────────────┤    ├──────────────────────────────────┤ │
│  │  • Streamlit Web UI (Port 8501)  │    │  • GDSC Datasets                 │ │
│  │  • REST API (FastAPI, Port 8000) │    │  • CCLE Datasets                 │ │
│  │  • Jupyter Notebooks             │    │  • CTRP Datasets                 │ │
│  │  • CLI Scripts                   │    │  • Toy Datasets (Testing)        │ │
│  └──────────────────────────────────┘    └──────────────────────────────────┘ │
│           │                                      │                              │
│           └──────────────┬───────────────────────┘                              │
│                          │                                                       │
│                    ┌─────▼──────────┐                                           │
│                    │  Data Layer    │                                           │
│                    │  - drevalpy    │                                           │
│                    │  - Loaders     │                                           │
│                    │  - Normalizers │                                           │
│                    └─────┬──────────┘                                           │
│                          │                                                       │
│        ┌─────────────────┼─────────────────┐                                    │
│        │                 │                 │                                    │
│   ┌────▼────┐      ┌────▼─────┐     ┌────▼──────┐                             │
│   │Training │      │Inference │     │Explanation│                             │
│   │Pipeline │      │Engine    │     │Engine     │                             │
│   └────┬────┘      └────┬─────┘     └────┬──────┘                             │
│        │                │                 │                                    │
│   ┌────▼────────────────▼─────────────────▼──────┐                            │
│   │        Core Model Architecture                │                            │
│   │        TabTransformer Network                 │                            │
│   │  ┌────────────────────────────────────────┐  │                            │
│   │  │ Concatenated Input Features            │  │                            │
│   │  │ (Gene Expression + Drug Fingerprints)  │  │                            │
│   │  └──────────────┬─────────────────────────┘  │                            │
│   │               │                               │                            │
│   │  ┌────────────▼──────────────────────────┐   │                            │
│   │  │  Feature Tokenization                 │   │                            │
│   │  │  (Chunking into 64-dim tokens)        │   │                            │
│   │  └──────────────┬───────────────────────┘   │                            │
│   │               │                               │                            │
│   │  ┌────────────▼──────────────────────────┐   │                            │
│   │  │  Transformer Encoder                  │   │                            │
│   │  │  (Multi-head Attention × 4 layers)    │   │                            │
│   │  └──────────────┬───────────────────────┘   │                            │
│   │               │                               │                            │
│   │  ┌────────────▼──────────────────────────┐   │                            │
│   │  │  [CLS] Token Aggregation              │   │                            │
│   │  └──────────────┬───────────────────────┘   │                            │
│   │               │                               │                            │
│   │  ┌────────────▼──────────────────────────┐   │                            │
│   │  │  Regression Head                      │   │                            │
│   │  │  (Linear layer → LN_IC50 prediction)  │   │                            │
│   │  └──────────────┬───────────────────────┘   │                            │
│   │               │                               │                            │
│   │  ┌────────────▼──────────────────────────┐   │                            │
│   │  │  Output: LN_IC50 Value                │   │                            │
│   │  │  Classification: Sensitive/Resistant   │   │                            │
│   │  └────────────────────────────────────────┘   │                            │
│   └────────────────────────────────────────────────┘                           │
│        │                │                 │                                    │
│   ┌────▼────┐      ┌────▼─────┐     ┌────▼──────────────┐                    │
│   │Training │      │Prediction │     │SHAP               │                    │
│   │Results  │      │Output     │     │Explanation       │                    │
│   └─────────┘      └───────────┘     └───────────────────┘                    │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                    PHARMAAI COMPONENT HIERARCHY                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ PRESENTATION LAYER                                               │ │
│  │ ┌────────────────────┐  ┌────────────────────┐  ┌────────────┐ │ │
│  │ │  Streamlit App     │  │  REST API          │  │  Notebooks │ │ │
│  │ │  (app.py)          │  │  (FastAPI)         │  │  (Scripts) │ │ │
│  │ │  - Predict Tab     │  │  - /predict        │  │  - Training│ │ │
│  │ │  - Explainability  │  │  - /batch_predict  │  │  - Analysis│ │ │
│  │ │  - About Tab       │  │  - /model/info     │  │            │ │ │
│  │ │  - Model loading   │  │  - /explanations   │  │            │ │ │
│  │ └────────────────────┘  └────────────────────┘  └────────────┘ │ │
│  └────────────┬────────────────────┬──────────────────┬─────────────┘ │
│               │                    │                  │                │
│  ┌────────────▼────────────────────▼──────────────────▼─────────────┐ │
│  │ BUSINESS LOGIC LAYER                                             │ │
│  │ ┌────────────────────────────────────────────────────────────┐  │ │
│  │ │ Services                                                   │  │ │
│  │ │  - Predictor Service (predictor.py)                       │  │ │
│  │ │    • Load models                                           │  │ │
│  │ │    • Make predictions                                     │  │ │
│  │ │    • Batch processing                                    │  │ │
│  │ │                                                            │  │ │
│  │ │  - Explainer Service (explainer.py)                       │  │ │
│  │ │    • Generate SHAP values                                 │  │ │
│  │ │    • Create visualizations                                │  │ │
│  │ │    • Extract feature importance                           │  │ │
│  │ │                                                            │  │ │
│  │ │  - Training Service (train_pharmaai.py)                   │  │ │
│  │ │    • Orchestrate training pipeline                        │  │ │
│  │ │    • Handle hyperparameter tuning                         │  │ │
│  │ │    • Manage cross-validation                              │  │ │
│  │ └────────────────────────────────────────────────────────────┘  │ │
│  └────────────┬────────────────────────────────────────────────────┘ │
│               │                                                       │
│  ┌────────────▼────────────────────────────────────────────────────┐ │
│  │ MODEL LAYER                                                      │ │
│  │ ┌──────────────────────────────────────────────────────────┐   │ │
│  │ │ TabTransformer Model                                     │   │ │
│  │ │ (models/TabTransformer/tab_transformer.py)              │   │ │
│  │ │                                                          │   │ │
│  │ │  Architecture Components:                               │   │ │
│  │ │  • TokenizationLayer (chunk features → tokens)         │   │ │
│  │ │  • TransformerEncoder (multi-head attention)           │   │ │
│  │ │  • RegressionHead (token aggregation → output)         │   │ │
│  │ │  • Hyperparameter store                                │   │ │
│  │ │  • Model I/O (save/load)                              │   │ │
│  │ └──────────────────────────────────────────────────────────┘   │ │
│  │                                                                  │ │
│  │ ┌──────────────────────────────────────────────────────────┐   │ │
│  │ │ Baseline Models (via MODEL_FACTORY)                      │   │ │
│  │ │  • ElasticNet (sklearn)                                  │   │ │
│  │ │  • RandomForest (sklearn)                                │   │ │
│  │ │  • SimpleNeuralNetwork (PyTorch)                        │   │ │
│  │ └──────────────────────────────────────────────────────────┘   │ │
│  └────────────┬─────────────────────────────────────────────────┘ │
│               │                                                   │
│  ┌────────────▼─────────────────────────────────────────────────┐ │
│  │ DATA LAYER                                                    │ │
│  │ ┌──────────────────────────────────────────────────────────┐ │ │
│  │ │ Data Loaders (drevalpy)                                  │ │ │
│  │ │  • Dataset loading (GDSC, CCLE, CTRP, Toy)             │ │ │
│  │ │  • Feature extraction                                   │ │ │
│  │ │    - Gene expression (landmark genes: ~978)            │ │ │
│  │ │    - Drug fingerprints (Morgan: ~2048)                │ │ │
│  │ │  • Train/Val/Test splitting                            │ │ │
│  │ │  • Cross-validation scheduling                         │ │ │
│  │ └──────────────────────────────────────────────────────────┘ │ │
│  │                                                                │ │
│  │ ┌──────────────────────────────────────────────────────────┐ │ │
│  │ │ Data Preprocessing                                        │ │ │
│  │ │  • Normalization (StandardScaler)                        │ │ │
│  │ │  • Missing value handling                                │ │ │
│  │ │  • Feature concatenation                                 │ │ │
│  │ │  • Batch creation                                        │ │ │
│  │ └──────────────────────────────────────────────────────────┘ │ │
│  └────────────┬────────────────────────────────────────────────┘ │
│               │                                                   │
│  ┌────────────▼─────────────────────────────────────────────────┐ │
│  │ STORAGE LAYER                                                 │ │
│  │  • Local filesystem                                           │ │
│  │    - Model checkpoints (model.pt)                            │ │
│  │    - Hyperparameters (hyperparameters.json)                  │ │
│  │    - Scalers (scaler.pkl)                                   │ │
│  │    - Logs & metrics                                          │ │
│  │  • Cache directory                                            │ │
│  │    - Downloaded datasets                                     │ │
│  │    - SHAP results                                            │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Architecture

```
INPUT DATA
    │
    ├─────────────────┬──────────────────┬─────────────────┐
    │                 │                  │                 │
┌───▼──────┐   ┌─────▼──────┐    ┌─────▼──────┐    ┌────▼────────┐
│ GDSC      │   │ CCLE       │    │ CTRP       │    │ Toy/Custom  │
│ Datasets  │   │ Datasets   │    │ Datasets   │    │ Datasets    │
└───┬──────┘   └─────┬──────┘    └─────┬──────┘    └────┬────────┘
    │                │                 │                │
    └────────────────┼─────────────────┴────────────────┘
                     │
         ┌───────────▼───────────┐
         │  FEATURE EXTRACTION   │
         │  ┌─────────────────┐  │
         │  │ Gene Expression │  │
         │  │ Landmark Genes  │  │
         │  │ (~978 features) │  │
         │  └─────────────────┘  │
         │  ┌─────────────────┐  │
         │  │ Drug Properties │  │
         │  │ Morgan Finger   │  │
         │  │ (~2048 features)│  │
         │  └─────────────────┘  │
         └───────────┬───────────┘
                     │
         ┌───────────▼─────────────┐
         │ FEATURE CONCATENATION   │
         │ (978 + 2048 = 3026)     │
         └───────────┬─────────────┘
                     │
         ┌───────────▼──────────────┐
         │  NORMALIZATION           │
         │  StandardScaler          │
         │  Mean=0, Std=1           │
         └───────────┬──────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼─────┐   ┌────▼─────┐   ┌────▼─────┐
│ TRAINING │   │ VALIDATION│   │   TEST   │
│ SET (70%)│   │ SET (15%) │   │Set (15%) │
└────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │
     └──────────────┼──────────────┘
                    │
        ┌───────────▼──────────┐
        │  CROSS-VALIDATION    │
        │  ┌────────────────┐  │
        │  │ LPO: Pair-Out  │  │
        │  │ LCO: Cell-Out  │  │
        │  │ LDO: Drug-Out  │  │
        │  │ LTO: Tissue-Out│  │
        │  └────────────────┘  │
        └───────────┬──────────┘
                    │
        ┌───────────▼──────────────┐
        │  MODEL TRAINING          │
        │                          │
        │  TabTransformer          │
        │  + Baselines             │
        │  (ElasticNet, RF, etc)   │
        └───────────┬──────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
┌───▼────┐   ┌─────▼──────┐   ┌────▼──────┐
│ Metrics │   │ Checkpoints│   │ Logs      │
└───┬────┘   └─────┬──────┘   └────┬──────┘
    │              │               │
    └──────────────┴───────────────┘
                   │
        ┌──────────▼──────────┐
        │ INFERENCE/PREDICT   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  EXPLAINABILITY     │
        │  SHAP Analysis      │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  VISUALIZATION      │
        │  & REPORTING        │
        └─────────────────────┘
```

---

## 4. TabTransformer Architecture Details

```
╔════════════════════════════════════════════════════════════════════╗
║               TABTRANSFORMER NEURAL NETWORK                        ║
╚════════════════════════════════════════════════════════════════════╝

Input Features (3026-dimensional)
    │
    ├─ Gene Expression: 978 features
    └─ Drug Fingerprints: 2048 features
    │
    ▼
┌────────────────────────────────────┐
│ TOKENIZATION LAYER                 │
│                                    │
│  Input Dimension: 3026             │
│  Token Dimension: 64               │
│  Number of Tokens: 3026/64 ≈ 48   │
│                                    │
│  Process:                          │
│  1. Chunk features into 64-dim     │
│     subsequences                   │
│  2. Add position embeddings        │
│  3. Add [CLS] token (special)      │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ TRANSFORMER ENCODER BLOCK × 4      │
│                                    │
│ For each block:                    │
│ ┌──────────────────────────────┐  │
│ │ Multi-Head Attention         │  │
│ │  • Heads: 8                  │  │
│ │  • Head Dim: 64/8 = 8        │  │
│ │  • Query, Key, Value proj    │  │
│ │  • Self-attention across     │  │
│ │    all token positions       │  │
│ └──────────────────────────────┘  │
│    │                               │
│    ▼                               │
│ ┌──────────────────────────────┐  │
│ │ Layer Normalization           │  │
│ └──────────────────────────────┘  │
│    │                               │
│    ▼                               │
│ ┌──────────────────────────────┐  │
│ │ Feed-Forward Network          │  │
│ │  • Linear(64 → 256)          │  │
│ │  • GELU Activation            │  │
│ │  • Dropout(0.1)               │  │
│ │  • Linear(256 → 64)          │  │
│ └──────────────────────────────┘  │
│    │                               │
│    ▼                               │
│ ┌──────────────────────────────┐  │
│ │ Layer Normalization           │  │
│ │ Residual Connection           │  │
│ └──────────────────────────────┘  │
│                                    │
└────────────────────────────────────┘
    │
    ▼ (Repeat 4 times)
    │
    ▼
┌────────────────────────────────────┐
│ [CLS] TOKEN POOLING                │
│                                    │
│ Extract [CLS] token representation │
│ (Learns global context)            │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ REGRESSION HEAD                    │
│                                    │
│ ┌──────────────────────────────┐  │
│ │ Linear(64 → 1)               │  │
│ │ (Projects to scalar output)   │  │
│ └──────────────────────────────┘  │
└────────────────────────────────────┘
    │
    ▼
OUTPUT: LN_IC50 (scalar value)
    │
    ├─ Value < 0: SENSITIVE (drug effective)
    └─ Value ≥ 0: RESISTANT (drug ineffective)

Hyperparameters:
─────────────────
• Input Dimension: 3026
• Token Dimension: 64
• Number of Layers: 4
• Number of Heads: 8
• Attention Head Dimension: 8
• FF Hidden Dimension: 256
• Dropout Rate: 0.1
• Learning Rate: 0.001 (adaptive)
• Batch Size: 128
• Number of Epochs: 100
• Optimizer: Adam
• Loss Function: Mean Squared Error (MSE)
```

---

## 5. Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          TRAINING PIPELINE (Drevalpy Integration)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 1. DATASET SELECTION & LOADING                     │   │
│  │                                                    │   │
│  │  Options:                                          │   │
│  │  ├─ GDSC1 / GDSC2 (primary)                       │   │
│  │  ├─ CCLE (external validation)                     │   │
│  │  ├─ CTRP v1 / v2                                  │   │
│  │  └─ TOYv1 / TOYv2 (testing)                       │   │
│  │                                                    │   │
│  │  drevalpy.load_dataset(dataset_name, measure)     │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 2. RESPONSE DATA PREPARATION                      │   │
│  │                                                  │   │
│  │  • Load response values (LN_IC50)                │   │
│  │  • Remove missing values                         │   │
│  │  • Statistics: mean, std, range                  │   │
│  │  • Sample counts                                 │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 3. FEATURE LOADING & SELECTION                    │   │
│  │                                                  │   │
│  │  Gene Expression:                                │   │
│  │  • Feature: "landmark_genes_reduced"             │   │
│  │  • Count: ~978 genes                             │   │
│  │  • Normalization: StandardScaler                │   │
│  │                                                  │   │
│  │  Drug Properties:                                │   │
│  │  • Feature: "fingerprints" (Morgan, ECFP)       │   │
│  │  • Bits: 2048 (standard)                        │   │
│  │  • Type: Binary (0/1)                           │   │
│  │                                                  │   │
│  │  load_and_select_gene_features()                │   │
│  │  load_drug_fingerprint_features()               │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 4. FEATURE MATRIX CONSTRUCTION                    │   │
│  │                                                  │   │
│  │  For each (cell_line, drug) pair:               │   │
│  │  ├─ Fetch gene expression vector (978)          │   │
│  │  ├─ Fetch drug fingerprint vector (2048)        │   │
│  │  └─ Concatenate → Feature vector (3026)         │   │
│  │                                                  │   │
│  │  Output: Dense matrix (n_samples × 3026)        │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 5. DATA SPLITTING & CV SCHEDULING                │   │
│  │                                                  │   │
│  │  Cross-Validation Strategies:                    │   │
│  │  ┌──────────────────────────────────────────┐   │   │
│  │  │ LPO (Leave-Pair-Out)                     │   │   │
│  │  │ • Split by (cell_line, drug) pairs       │   │   │
│  │  │ • Most challenging, generalizes best     │   │   │
│  │  └──────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────┐   │   │
│  │  │ LCO (Leave-Cell-Line-Out)               │   │   │
│  │  │ • Test on unseen cell lines             │   │   │
│  │  │ • Measures cell line generalization     │   │   │
│  │  └──────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────┐   │   │
│  │  │ LDO (Leave-Drug-Out)                    │   │   │
│  │  │ • Test on unseen drugs                  │   │   │
│  │  │ • Measures drug generalization          │   │   │
│  │  └──────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────┐   │   │
│  │  │ LTO (Leave-Tissue-Out)                  │   │   │
│  │  │ • Test on unseen tissue types           │   │   │
│  │  │ • Measures tissue generalization        │   │   │
│  │  └──────────────────────────────────────────┘   │   │
│  │                                                  │   │
│  │  n_cv_splits = 5 (default)                     │   │
│  │  10-fold or k-fold cross-validation            │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 6. HYPERPARAMETER TUNING (Optional)              │   │
│  │                                                  │   │
│  │  • Grid search or random search                 │   │
│  │  • Validate on fold-specific validation set     │   │
│  │  • Select best hyperparameters                  │   │
│  │                                                  │   │
│  │  HPAMTuner.tune() or skip with --no-hpam        │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 7. MODEL TRAINING (per fold)                     │   │
│  │                                                  │   │
│  │  For each CV fold:                              │   │
│  │  ├─ Create fresh model instance                 │   │
│  │  ├─ Train on training samples                   │   │
│  │  │  • PyTorch Lightning trainer                 │   │
│  │  │  • Batch gradient descent                    │   │
│  │  │  • Early stopping on validation              │   │
│  │  ├─ Validate on validation samples              │   │
│  │  │  • Monitor metrics in real-time              │   │
│  │  └─ Evaluate on test samples                    │   │
│  │                                                  │   │
│  │  Duration: ~4-8 hours for GDSC2, 1 fold         │   │
│  │            ~30-60 hours for GDSC2, 5 folds      │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 8. BASELINE TRAINING (Parallel)                  │   │
│  │                                                  │   │
│  │  • ElasticNet (sklearn)                         │   │
│  │  • RandomForest (sklearn)                       │   │
│  │  • SimpleNeuralNetwork (PyTorch)                │   │
│  │                                                  │   │
│  │  Use skip: --no-baselines                       │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 9. METRICS COMPUTATION & AGGREGATION             │   │
│  │                                                  │   │
│  │  Per-fold metrics:                              │   │
│  │  • R² Score                                     │   │
│  │  • Mean Squared Error (MSE)                     │   │
│  │  • Mean Absolute Error (MAE)                    │   │
│  │  • Root Mean Squared Error (RMSE)               │   │
│  │  • Spearman Correlation                         │   │
│  │  • Pearson Correlation                          │   │
│  │                                                  │   │
│  │  Aggregate across folds:                        │   │
│  │  • Mean and standard deviation                  │   │
│  │  • 95% confidence intervals                     │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 10. CROSS-STUDY VALIDATION (Optional)            │   │
│  │                                                  │   │
│  │  Train on primary dataset (e.g., GDSC2)        │   │
│  │  Test on external dataset (e.g., CCLE)         │   │
│  │                                                  │   │
│  │  Measures:                                      │   │
│  │  • Cross-dataset generalization ability        │   │
│  │  • Transfer learning performance                │   │
│  │                                                  │   │
│  │  --cross-study CCLE                             │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │ 11. RESULTS AGGREGATION & REPORTING              │   │
│  │                                                  │   │
│  │  Output directory structure:                    │   │
│  │  results/                                       │   │
│  │  └─ PharmaAI_Transformer_2025/                  │   │
│  │     ├─ TabTransformer/                         │   │
│  │     │  ├─ model.pt (weights)                   │   │
│  │     │  ├─ hyperparameters.json                 │   │
│  │     │  ├─ scaler.pkl                           │   │
│  │     │  └─ logs/                                │   │
│  │     ├─ ElasticNet/                             │   │
│  │     ├─ RandomForest/                           │   │
│  │     └─ experiment_results.csv                  │   │
│  │                                                  │   │
│  │  Log files:                                     │   │
│  │  • Training logs (metrics per epoch)            │   │
│  │  • Final metrics report                         │   │
│  │  • Model configuration                         │   │
│  │  • Hardware & environment info                 │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ TRAINING COMPLETE                              │   │
│  │ Models ready for inference & explanation       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 6. REST API Architecture

```
┌───────────────────────────────────────┐
│  FASTAPI SERVER (Port 8000)           │
├───────────────────────────────────────┤
│                                       │
│  ┌─────────────────────────────────┐ │
│  │  ROUTING LAYER                  │ │
│  │  ┌───────────────────────────┐  │ │
│  │  │ /health (GET)             │  │ │
│  │  │ ├─ Status check           │  │ │
│  │  │ └─ Readiness probe        │  │ │
│  │  └───────────────────────────┘  │ │
│  │  ┌───────────────────────────┐  │ │
│  │  │ /predict (POST)           │  │ │
│  │  │ ├─ Single prediction      │  │ │
│  │  │ ├─ Input validation       │  │ │
│  │  │ └─ Response formatting    │  │ │
│  │  └───────────────────────────┘  │ │
│  │  ┌───────────────────────────┐  │ │
│  │  │ /predict/batch (POST)     │  │ │
│  │  │ ├─ Batch predictions      │  │ │
│  │  │ ├─ Parallel processing    │  │ │
│  │  │ └─ Progress tracking      │  │ │
│  │  └───────────────────────────┘  │ │
│  │  ┌───────────────────────────┐  │ │
│  │  │ /model/info (GET)         │  │ │
│  │  │ ├─ Model metadata         │  │ │
│  │  │ ├─ Training info          │  │ │
│  │  │ └─ Performance metrics    │  │ │
│  │  └───────────────────────────┘  │ │
│  │  ┌───────────────────────────┐  │ │
│  │  │ /interpretability/        │  │ │
│  │  │ top-features (GET)        │  │ │
│  │  │ ├─ Feature importance     │  │ │
│  │  │ ├─ SHAP values            │  │ │
│  │  │ └─ Feature rankings       │  │ │
│  │  └───────────────────────────┘  │ │
│  │  ┌───────────────────────────┐  │ │
│  │  │ /docs (GET)               │  │ │
│  │  │ ├─ Swagger UI             │  │ │
│  │  │ └─ Interactive testing    │  │ │
│  │  └───────────────────────────┘  │ │
│  └─────────────────────────────────┘ │
│           │                           │
│  ┌────────▼──────────────────────────┐ │
│  │ SERVICE LAYER                     │ │
│  │ ┌────────────────────────────────┐ │ │
│  │ │ PredictorService               │ │ │
│  │ │ • load_model()                 │ │ │
│  │ │ • predict()                    │ │ │
│  │ │ • batch_predict()              │ │ │
│  │ │ • validate_input()             │ │ │
│  │ └────────────────────────────────┘ │ │
│  │ ┌────────────────────────────────┐ │ │
│  │ │ ExplainerService               │ │ │
│  │ │ • get_top_features()           │ │ │
│  │ │ • compute_shap()               │ │ │
│  │ │ • get_feature_importance()     │ │ │
│  │ └────────────────────────────────┘ │ │
│  │ ┌────────────────────────────────┐ │ │
│  │ │ ModelInfoService               │ │ │
│  │ │ • get_metadata()               │ │ │
│  │ │ • get_metrics()                │ │ │
│  │ │ • get_config()                 │ │ │
│  │ └────────────────────────────────┘ │ │
│  └─────────────────────────────────────┘ │
│           │                              │
│  ┌────────▼──────────────────────────┐  │
│  │ MIDDLEWARE LAYER                  │  │
│  │ • CORS handling                   │  │
│  │ • Request logging                 │  │
│  │ • Error handling                  │  │
│  │ • Rate limiting (optional)        │  │
│  │ • Authentication (optional)       │  │
│  └─────────────────────────────────────┘ │
│                                       │
└───────────────────────────────────────┘
```

---

## 7. Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│              DEPLOYMENT TOPOLOGY                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Development Environment:                              │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Local Machine                                     │ │
│  │  ├─ Python 3.8+                                  │ │
│  │  ├─ Virtual Environment (venv)                   │ │
│  │  ├─ Dependencies (pip)                           │ │
│  │  └─ GPU Support (CUDA/cuDNN optional)           │ │
│  └──────────────────────────────────────────────────┘ │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Application Layer                              │   │
│  │  ┌────────────────────────────────────────┐   │   │
│  │  │ Streamlit (Port 8501)                 │   │   │
│  │  │ └─ Web UI                             │   │   │
│  │  └────────────────────────────────────────┘   │   │
│  │  ┌────────────────────────────────────────┐   │   │
│  │  │ FastAPI (Port 8000)                   │   │   │
│  │  │ └─ REST API                           │   │   │
│  │  └────────────────────────────────────────┘   │   │
│  │  ┌────────────────────────────────────────┐   │   │
│  │  │ Jupyter Notebooks                     │   │   │
│  │  │ └─ Analysis & Training                │   │   │
│  │  └────────────────────────────────────────┘   │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Storage Layer                                  │   │
│  │  ├─ ./data/              (Datasets)           │   │
│  │  ├─ ./results/           (Models)             │   │
│  │  ├─ ./shap_results/      (Analysis)           │   │
│  │  └─ ./.env               (Config)             │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  Production Environment (Scalable):                   │
│  ┌────────────────────────────────────────────────┐   │
│  │ Containerization (Docker)                      │   │
│  │  ├─ Dockerfile (multi-stage build)             │   │
│  │  ├─ Separate API image                        │   │
│  │  ├─ Separate UI image                         │   │
│  │  └─ Volume management                         │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Orchestration (Kubernetes/Docker Compose)      │   │
│  │  ├─ Service discovery                         │   │
│  │  ├─ Load balancing                            │   │
│  │  ├─ Auto-scaling (if K8s)                    │   │
│  │  └─ Health checks                             │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ External Services                              │   │
│  │  ├─ Cloud Storage (S3, GCS)                   │   │
│  │  ├─ Database (PostgreSQL for logging)         │   │
│  │  ├─ Cache Layer (Redis)                       │   │
│  │  └─ Monitoring (Prometheus, ELK)              │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Error Handling & Monitoring

```
┌─────────────────────────────────────────────────┐
│     ERROR HANDLING & MONITORING ARCHITECTURE    │
├─────────────────────────────────────────────────┤
│                                                 │
│  INPUT VALIDATION                              │
│  ├─ Feature dimension check                    │
│  ├─ Data type validation                       │
│  ├─ Missing value detection                    │
│  └─ Outlier detection                          │
│       │                                         │
│       ▼                                         │
│  ERROR RESPONSES                               │
│  ├─ 400: Bad Request (invalid input)          │
│  ├─ 422: Unprocessable Entity (validation)    │
│  ├─ 500: Internal Server Error                │
│  └─ 503: Service Unavailable                  │
│       │                                         │
│       ▼                                         │
│  LOGGING & MONITORING                          │
│  ├─ Request/Response logging                   │
│  ├─ Model inference timing                     │
│  ├─ Error stack traces                         │
│  ├─ Performance metrics                        │
│  └─ System resource usage                      │
│       │                                         │
│       ▼                                         │
│  ALERTING (Production)                         │
│  ├─ High latency alerts                        │
│  ├─ Error rate thresholds                      │
│  ├─ Resource exhaustion                        │
│  └─ Service health checks                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 9. Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive web UI |
| **API** | FastAPI, Uvicorn | RESTful service |
| **ML Framework** | PyTorch, PyTorch Lightning | Deep learning |
| **Interpretability** | SHAP | Feature importance |
| **Data Pipeline** | drevalpy | ML experiment framework |
| **ML Ops** | scikit-learn | Baseline models |
| **Data Processing** | NumPy, Pandas | Data manipulation |
| **Visualization** | Matplotlib | Plotting |
| **Environment** | Python 3.8+ | Runtime |
| **Package Management** | pip, requirements.txt | Dependencies |
| **Version Control** | Git | Code management |

---

## 10. Performance Characteristics

### Computational Requirements

**Training (GDSC2 with TabTransformer)**
- **Time**: 4-8 hours per CV split (single GPU)
- **Memory**: 16GB RAM, 8GB VRAM (GPU recommended)
- **CPU Cores**: 8+ recommended
- **Total Training (5-fold CV)**: ~40-60 hours

**Inference**
- **Per Sample**: <100ms (CPU), <10ms (GPU)
- **Throughput**: 10-100 samples/second
- **Latency**: 50-200ms API response time
- **Memory**: <2GB for model + batch processing

**SHAP Explanation**
- **Time**: 2-4 hours per 100 background samples
- **Memory**: 8GB+ (KernelExplainer)
- **Parallelization**: Limited (mostly sequential)

---

## Architecture Decisions

### Why TabTransformer?

1. **Structured Data**: Effective for concatenated features (not pure sequences)
2. **Attention Mechanism**: Learns feature interactions
3. **Interpretability**: Attention weights reveal important features
4. **Performance**: Outperforms baselines on drug response prediction
5. **Scalability**: Can handle 3000+ features efficiently

### Why Tokenization?

- Reduces feature space for attention computation
- Allows flexible feature grouping
- Reduces quadratic attention complexity
- Improves numerical stability

### Why drevalpy Framework?

- **Standardized Benchmarking**: Consistent evaluation across models
- **Feature Management**: Automatic gene selection and drug fingerprints
- **CV Scheduling**: Multiple test modes (LPO, LCO, LDO, LTO)
- **Cross-Study Validation**: Easy external validation setup
- **Community Standards**: Aligns with drug response literature

---

## Future Architecture Enhancements

1. **Multi-Task Learning**: Predict multiple response measures simultaneously
2. **Transfer Learning**: Pre-train on large datasets, fine-tune on specific cell types
3. **Uncertainty Quantification**: Bayesian deep learning for confidence intervals
4. **Federated Learning**: Train on distributed hospital data
5. **AutoML**: Automated hyperparameter and architecture search
6. **Graph Neural Networks**: Model drug-target interactions explicitly
7. **Temporal Dynamics**: Predict resistance evolution over treatment

---

## References

- PyTorch Documentation: https://pytorch.org/docs/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- SHAP Documentation: https://shap.readthedocs.io/
- drevalpy Repository: https://github.com/daisybio/drevalpy
- Streamlit Documentation: https://docs.streamlit.io/

---

**Document Version**: 1.0  
**Last Updated**: April 2025
