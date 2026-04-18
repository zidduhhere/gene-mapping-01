# PharmaAI Predictor - Flow Diagrams & Workflows

This document contains detailed flow diagrams showing how data moves through the system, complete workflows, and process maps for different use cases.

---

## 1. Complete End-to-End System Flow

```
                    ┌─────────────────────────────────────┐
                    │     USER / EXTERNAL SYSTEM          │
                    └──────────────┬──────────────────────┘
                                   │
                  ┌────────────────┼────────────────┐
                  │                │                │
         ┌────────▼──────┐  ┌─────▼────────┐ ┌─────▼─────────┐
         │ Web Interface │  │  REST API    │ │ Python Script │
         │  (Streamlit)  │  │  (FastAPI)   │ │ / Notebook    │
         └────────┬──────┘  └─────┬────────┘ └─────┬─────────┘
                  │                │                │
                  └────────────────┼────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │  INPUT HANDLER   │
                          │                  │
                          │ • CSV upload     │
                          │ • JSON request   │
                          │ • Manual entry   │
                          │ • Batch data     │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────────────┐
                          │  INPUT VALIDATION       │
                          │                         │
                          │ ✓ Feature count check   │
                          │ ✓ Data type validation  │
                          │ ✓ Range checking        │
                          │ ✓ Missing value check   │
                          └────────┬────────────────┘
                                   │
                         ┌─────────▼──────────┐
                         │ LOAD SCALER FILE   │
                         │ (StandardScaler)   │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  FEATURE NORMALIZATION     │
                    │                            │
                    │ X_norm = (X - mean) / std │
                    └──────────────┬─────────────┘
                                   │
                ┌──────────────────▼──────────────────┐
                │   LOAD TRAINED MODEL WEIGHTS        │
                │                                     │
                │  • model.pt (TabTransformer)       │
                │  • hyperparameters.json            │
                │  • Scaler.pkl                      │
                └──────────────────┬──────────────────┘
                                   │
            ┌──────────────────────▼──────────────────────┐
            │     MODEL INFERENCE (Forward Pass)         │
            │                                            │
            │  Input: Normalized Features (3026-dim)    │
            │    ↓                                        │
            │  [Tokenization] → 48 tokens                │
            │    ↓                                        │
            │  [Transformer Encoder × 4 layers]         │
            │    ↓                                        │
            │  [Multi-head Attention × 8 heads]         │
            │    ↓                                        │
            │  [[CLS] Token Pooling]                    │
            │    ↓                                        │
            │  [Regression Head (Linear)]               │
            │    ↓                                        │
            │  Output: LN_IC50 (scalar)                 │
            └──────────────────┬───────────────────────┘
                               │
                    ┌──────────▼─────────┐
                    │  CLASSIFICATION    │
                    │                    │
                    │ IF LN_IC50 < 0:   │
                    │   SENSITIVE       │
                    │ ELSE:             │
                    │   RESISTANT       │
                    └──────────────┬────┘
                                  │
              ┌───────────────────▼────────────────┐
              │   GENERATE EXPLANATION (SHAP)     │
              │                                   │
              │  • Compute SHAP values            │
              │  • Feature importance scores      │
              │  • Top 20 features                │
              │  • Feature impact direction       │
              └───────────────────┬────────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │  FORMAT RESPONSE       │
                      │                        │
                      │  {                     │
                      │   "prediction": ...,   │
                      │   "classification": ..,│
                      │   "confidence": ...,   │
                      │   "top_features": ..,  │
                      │   "timestamp": ...     │
                      │  }                     │
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │  SEND RESPONSE         │
                      │                        │
                      │  • JSON (API)          │
                      │  • HTML (Web)          │
                      │  • CSV Download        │
                      │  • Visualization       │
                      └───────────┬────────────┘
                                  │
                          ┌───────▼────────┐
                          │  USER OUTPUT   │
                          └────────────────┘
```

---

## 2. Training Pipeline Flow

```
                         ┌──────────────────┐
                         │ START TRAINING   │
                         │ train_pharmaai.py│
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │ PARSE ARGUMENTS          │
                    │                          │
                    │ --dataset GDSC2          │
                    │ --test-mode LPO          │
                    │ --n-cv-splits 5          │
                    │ --toy (optional)         │
                    │ --cross-study CCLE       │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────────┐
                    │ LOAD PRIMARY DATASET         │
                    │                              │
                    │ Options:                     │
                    │ • GDSC1 / GDSC2             │
                    │ • CCLE                      │
                    │ • CTRP v1 / v2              │
                    │ • TOYv1 / TOYv2             │
                    │                              │
                    │ drevalpy.load_dataset()     │
                    └─────────────┬────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        │          ┌──────────────▼──────────────┐          │
        │          │ LOAD GENE EXPRESSION        │          │
        │          │                             │          │
        │          │ • Feature: landmark_genes   │          │
        │          │ • Count: ~978 genes         │          │
        │          │ • Normalization: Z-score   │          │
        │          │                             │          │
        │          │ load_and_select_gene_...() │          │
        │          └──────────────┬──────────────┘          │
        │                         │                         │
        │                         ▼                         │
        │          ┌─────────────────────────────────────┐  │
        │          │ Feature Matrix (N × 978)           │  │
        │          └─────────────┬─────────────────────┘  │
        │                        │                        │
        │          ┌─────────────▼──────────────┐         │
        │          │ LOAD DRUG FINGERPRINTS      │         │
        │          │                             │         │
        │          │ • Type: Morgan (ECFP)       │         │
        │          │ • Bits: 2048                │         │
        │          │ • Format: Binary            │         │
        │          │                             │         │
        │          │ load_drug_fingerprint_...() │         │
        │          └──────────────┬──────────────┘         │
        │                         │                        │
        │                         ▼                        │
        │          ┌─────────────────────────────────────┐ │
        │          │ Fingerprint Matrix (M × 2048)      │ │
        │          └─────────────┬─────────────────────┘ │
        │                        │                       │
        └─────────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ FEATURE CONCATENATION     │
                    │                           │
                    │ For each (cell, drug):    │
                    │ feature = [gex, fp]      │
                    │ Dimensions: 978 + 2048   │
                    │         = 3026 total     │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ BUILD RESPONSE MATRIX      │
                    │                            │
                    │ X: (N × 3026) features    │
                    │ y: (N,) LN_IC50 values    │
                    │ Sample indices             │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │ SELECT TEST MODE               │
                    │                                │
                    │ LPO: Leave-Pair-Out (default)  │
                    │  → Split by (cell, drug)       │
                    │  → Most challenging            │
                    │  → Best generalization         │
                    │                                │
                    │ LCO: Leave-Cell-Out            │
                    │  → Split by cell line          │
                    │  → Cell generalization         │
                    │                                │
                    │ LDO: Leave-Drug-Out            │
                    │  → Split by drug               │
                    │  → Drug generalization         │
                    │                                │
                    │ LTO: Leave-Tissue-Out          │
                    │  → Split by tissue type        │
                    │  → Tissue generalization       │
                    └─────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ CV SPLIT SCHEDULING        │
                    │                            │
                    │ Create 5 folds:            │
                    │ • Fold 1: Train on 4, ...  │
                    │ • Fold 2: Test on 1 fold   │
                    │ • ...                      │
                    │ • Fold 5                   │
                    │                            │
                    │ drevalpy.split_cv()        │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────────────┐
                    │ LOAD CROSS-STUDY DATA (Optional)   │
                    │                                    │
                    │ IF --cross-study CCLE:             │
                    │ • Load CCLE dataset                │
                    │ • Same feature extraction          │
                    │ • For external validation          │
                    │ • After training on primary        │
                    └─────────────┬──────────────────────┘
                                  │
    ┌─────────────────────────────▼─────────────────────────────┐
    │                                                           │
    │  ╔═════════════════════════════════════════════════════╗  │
    │  ║  FOR EACH CV FOLD (Fold Index = 1 to 5):          ║  │
    │  ╚═════════════════════════════════════════════════════╝  │
    │                     │                                    │
    │        ┌────────────▼─────────────┐                    │
    │        │ EXTRACT FOLD SPLITS      │                    │
    │        │                          │                    │
    │        │ Train set: Indices set   │                    │
    │        │ Val set: Indices set     │                    │
    │        │ Test set: Indices set    │                    │
    │        └────────────┬─────────────┘                    │
    │                     │                                  │
    │        ┌────────────▼──────────────────┐              │
    │        │ OPTIONAL: HP TUNING           │              │
    │        │                               │              │
    │        │ IF not --no-hpam-tuning:      │              │
    │        │ • Grid search over params     │              │
    │        │ • Validate on fold's val set  │              │
    │        │ • Select best hyperparams     │              │
    │        │                               │              │
    │        │ Duration: 1-2 hours           │              │
    │        └────────────┬──────────────────┘              │
    │                     │                                  │
    │        ┌────────────▼──────────────────────┐          │
    │        │ INITIALIZE MODEL INSTANCES        │          │
    │        │                                   │          │
    │        │ TabTransformer:                   │          │
    │        │ • Input dim: 3026                │          │
    │        │ • Token size: 64                 │          │
    │        │ • Encoder layers: 4              │          │
    │        │ • Attention heads: 8             │          │
    │        │ • Dropout: 0.1                   │          │
    │        │                                   │          │
    │        │ Baselines (ElasticNet, RF, etc) │          │
    │        └────────────┬──────────────────────┘          │
    │                     │                                  │
    │        ┌────────────▼──────────────────────────┐      │
    │        │ TRAIN TABTRANSFORMER                 │      │
    │        │                                      │      │
    │        │ PyTorch Lightning Trainer:           │      │
    │        │ • Optimizer: Adam                    │      │
    │        │ • LR: 0.001 (adaptive)               │      │
    │        │ • Batch size: 128                    │      │
    │        │ • Epochs: 100 max                    │      │
    │        │                                      │      │
    │        │ Per Epoch:                           │      │
    │        │ 1. Forward pass on train batches     │      │
    │        │ 2. Compute MSE loss                  │      │
    │        │ 3. Backward pass (gradients)         │      │
    │        │ 4. Optimizer step                    │      │
    │        │ 5. Validate on val set               │      │
    │        │ 6. Log metrics (loss, R², etc)      │      │
    │        │                                      │      │
    │        │ Early Stopping:                      │      │
    │        │ • Monitor val loss                   │      │
    │        │ • Stop if no improvement for N epochs│      │
    │        │                                      │      │
    │        │ Duration: 2-4 hours per fold        │      │
    │        └────────────┬──────────────────────────┘      │
    │                     │                                  │
    │        ┌────────────▼──────────────────────────┐      │
    │        │ TRAIN BASELINE MODELS (Parallel)     │      │
    │        │                                      │      │
    │        │ ElasticNet:                          │      │
    │        │ • sklearn.linear_model               │      │
    │        │ • Duration: 5-10 min                │      │
    │        │                                      │      │
    │        │ RandomForest:                        │      │
    │        │ • n_estimators: 100                 │      │
    │        │ • max_depth: 20                     │      │
    │        │ • Duration: 20-30 min              │      │
    │        │                                      │      │
    │        │ SimpleNeuralNetwork:                 │      │
    │        │ • PyTorch 2-layer MLP               │      │
    │        │ • Duration: 30-60 min              │      │
    │        │                                      │      │
    │        │ Can skip: --no-baselines             │      │
    │        └────────────┬──────────────────────────┘      │
    │                     │                                  │
    │        ┌────────────▼──────────────────────────────┐  │
    │        │ SAVE CHECKPOINT (TabTransformer only)    │  │
    │        │                                          │  │
    │        │ Save to:                                 │  │
    │        │ results/PharmaAI_Transformer_2025/      │  │
    │        │         TabTransformer/                 │  │
    │        │                                          │  │
    │        │ Files:                                   │  │
    │        │ • model.pt (weights)                     │  │
    │        │ • hyperparameters.json                   │  │
    │        │ • scaler.pkl                             │  │
    │        │ • training_log.txt                       │  │
    │        └────────────┬──────────────────────────────┘  │
    │                     │                                  │
    │        ┌────────────▼──────────────────────────────┐  │
    │        │ EVALUATE ON TEST SET                     │  │
    │        │                                          │  │
    │        │ For each model:                          │  │
    │        │ 1. Make predictions on test set         │  │
    │        │ 2. Compute metrics:                     │  │
    │        │    • R² score                           │  │
    │        │    • Mean Squared Error (MSE)          │  │
    │        │    • Mean Absolute Error (MAE)         │  │
    │        │    • Root MSE (RMSE)                   │  │
    │        │    • Spearman correlation               │  │
    │        │    • Pearson correlation                │  │
    │        │ 3. Store results for aggregation       │  │
    │        │                                          │  │
    │        │ Results shape:                           │  │
    │        │ {fold_id: {model: {metric: value}}}    │  │
    │        └────────────┬──────────────────────────────┘  │
    │                     │                                  │
    │        ┌────────────▼──────────────────────────────┐  │
    │        │ OPTIONAL: CROSS-STUDY EVAL              │  │
    │        │                                          │  │
    │        │ IF --cross-study CCLE:                   │  │
    │        │ 1. Test TabTransformer on CCLE test    │  │
    │        │ 2. Compute same metrics                 │  │
    │        │ 3. Compare with primary results        │  │
    │        │ 4. Assess generalization               │  │
    │        └────────────┬──────────────────────────────┘  │
    │                     │                                  │
    │        ┌────────────▼──────────────────────────────┐  │
    │        │ END OF FOLD                              │  │
    │        └────────────┬──────────────────────────────┘  │
    │                     │                                  │
    └─────────────────────┼──────────────────────────────────┘
                          │
                ┌─────────▼──────────┐
                │ AGGREGATE METRICS  │
                │                    │
                │ For each model:    │
                │ • Mean across folds│
                │ • Std dev          │
                │ • 95% CI           │
                │ • Best/worst fold  │
                └─────────┬──────────┘
                          │
                ┌─────────▼─────────────────────┐
                │ GENERATE RESULTS REPORT       │
                │                               │
                │ Create:                       │
                │ • experiment_results.csv      │
                │ • metrics_summary.txt         │
                │ • config.json                 │
                │ • training_log.txt            │
                └─────────┬─────────────────────┘
                          │
                ┌─────────▼────────────────────────────┐
                │ TRAINING COMPLETE                    │
                │                                      │
                │ Best model ready for:                │
                │ • Inference (predictions)            │
                │ • Explanation (SHAP)                 │
                │ • Web deployment (Streamlit)         │
                │ • API deployment (FastAPI)           │
                └──────────────────────────────────────┘
```

---

## 3. Prediction Workflow (Single Sample)

```
USER INPUT
    │
    ├─────────────────────┬──────────────────────┐
    │                     │                      │
    ▼                     ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ CSV Upload  │    │ REST API     │    │ Manual Text     │
│ (.csv file) │    │ (JSON POST)  │    │ (Comma-separated│
└──────┬──────┘    └───────┬──────┘    └────────┬────────┘
       │                   │                    │
       └───────────────────┼────────────────────┘
                           │
                ┌──────────▼─────────┐
                │ PARSE INPUT DATA   │
                │                    │
                │ Extract features   │
                │ (float array)      │
                └──────────┬─────────┘
                           │
         ┌─────────────────▼──────────────┐
         │ INPUT VALIDATION              │
         │                               │
         │ Check:                        │
         │ ✓ Feature count = 3026        │
         │ ✓ All values are floats       │
         │ ✓ No NaN or Inf              │
         │ ✓ Values in reasonable range  │
         │                               │
         │ IF validation fails:          │
         │  → Return 400 error message   │
         └─────────────┬──────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │ LOAD SCALER FILE (.pkl)    │
         │                            │
         │ StandardScaler object      │
         │ Contains:                  │
         │ • Mean per feature         │
         │ • Std per feature          │
         │ • n_features_in_           │
         │ • feature_names_in_        │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────────────┐
         │ NORMALIZE FEATURES                 │
         │                                    │
         │ For each feature:                  │
         │ X_norm = (X - mean) / std          │
         │                                    │
         │ Formula:                           │
         │ X_normalized[i] =                  │
         │    (X[i] - scaler.mean_[i]) /     │
         │    scaler.scale_[i]                │
         │                                    │
         │ Result: Zero-centered, unit        │
         │ variance features                  │
         └─────────────┬──────────────────────┘
                       │
         ┌─────────────▼──────────────────────┐
         │ LOAD MODEL WEIGHTS                 │
         │                                    │
         │ Load: model.pt                     │
         │ Type: PyTorch state_dict           │
         │ Size: ~5-10 MB                     │
         │ Location: results/...              │
         │           /TabTransformer/         │
         │                                    │
         │ Also load:                         │
         │ • hyperparameters.json             │
         │ • Architecture config              │
         └─────────────┬──────────────────────┘
                       │
         ┌─────────────▼──────────────────────────────┐
         │ PREPARE TENSOR INPUT                      │
         │                                           │
         │ Convert numpy array to PyTorch Tensor:    │
         │                                           │
         │ X_tensor = torch.from_numpy(X)            │
         │           .float()                        │
         │           .unsqueeze(0)  # Add batch dim  │
         │                                           │
         │ Shape: (1, 3026)                          │
         │ Device: CPU (or GPU if available)         │
         └─────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼───────────────────────────┐
         │ ENABLE INFERENCE MODE                  │
         │                                         │
         │ model.eval()  # Disable dropout, etc    │
         │ torch.no_grad()  # Disable autograd     │
         │                                         │
         │ Performance optimization:               │
         │ • Skip gradient computation             │
         │ • Reduce memory usage                   │
         │ • Faster inference                      │
         └─────────────┬───────────────────────────┘
                       │
         ┌─────────────▼──────────────────────────────────┐
         │ FORWARD PASS (MODEL INFERENCE)                │
         │                                               │
         │ Input: (1, 3026) normalized features          │
         │   │                                            │
         │   ▼                                            │
         │ ┌──────────────────────────────────────┐      │
         │ │ TOKENIZATION LAYER                   │      │
         │ │                                      │      │
         │ │ Chunk features into tokens:          │      │
         │ │ • Token size: 64 dimensions          │      │
         │ │ • N tokens: 3026 / 64 ≈ 48          │      │
         │ │ • Add position embeddings            │      │
         │ │ • Add [CLS] special token            │      │
         │ │                                      │      │
         │ │ Output shape: (1, 49, 64)            │      │
         │ │ (batch, tokens, dimensions)          │      │
         │ └──────────────┬───────────────────────┘      │
         │               │                                │
         │   ┌───────────▼──────────────┐                │
         │   │ TRANSFORMER ENCODER      │                │
         │   │ Layer 1 of 4             │                │
         │   │                          │                │
         │   │ Multi-Head Attention:    │                │
         │   │ • 8 heads                │                │
         │   │ • Self-attention across  │                │
         │   │   all 49 tokens          │                │
         │   │ • Compute Q, K, V        │                │
         │   │ • Scale dot-product attn │                │
         │   │ • Concatenate heads      │                │
         │   │ • Linear projection      │                │
         │   │                          │                │
         │   │ Feed-Forward Network:    │                │
         │   │ • Linear: 64 → 256       │                │
         │   │ • GELU activation        │                │
         │   │ • Dropout: 0.1           │                │
         │   │ • Linear: 256 → 64       │                │
         │   │                          │                │
         │   │ Residual Connections:    │                │
         │   │ • Add input to output    │                │
         │   │ • LayerNorm              │                │
         │   │                          │                │
         │   │ Output: (1, 49, 64)      │                │
         │   └───────────┬──────────────┘                │
         │               │                                │
         │   ┌───────────▼──────────────┐                │
         │   │ TRANSFORMER ENCODER      │                │
         │   │ Layers 2, 3, 4           │                │
         │   │ (Repeat above)           │                │
         │   │                          │                │
         │   │ Each layer:              │                │
         │   │ • Input: (1, 49, 64)     │                │
         │   │ • Output: (1, 49, 64)    │                │
         │   └───────────┬──────────────┘                │
         │               │                                │
         │   ┌───────────▼──────────────┐                │
         │   │ [CLS] TOKEN EXTRACTION   │                │
         │   │                          │                │
         │   │ Extract first token      │                │
         │   │ (special [CLS] token)    │                │
         │   │                          │                │
         │   │ Shape: (1, 64)           │                │
         │   │ Learned global context   │                │
         │   └───────────┬──────────────┘                │
         │               │                                │
         │   ┌───────────▼──────────────┐                │
         │   │ REGRESSION HEAD          │                │
         │   │                          │                │
         │   │ Linear transformation:   │                │
         │   │ output = W × [CLS] + b   │                │
         │   │ W: (1, 64)               │                │
         │   │ b: (1,)                  │                │
         │   │                          │                │
         │   │ Output shape: (1, 1)     │                │
         │   │ → Scalar value           │                │
         │   └───────────┬──────────────┘                │
         │               │                                │
         │               ▼                                │
         │ Output: LN_IC50 value (float)                 │
         │ Example: -0.4523                              │
         └──────────┬────────────────────────────────────┘
                    │
         ┌──────────▼──────────────┐
         │ DETACH FROM GPU         │
         │ (if inference on GPU)   │
         │                         │
         │ .cpu()                  │
         │ .numpy()                │
         └──────────┬──────────────┘
                    │
         ┌──────────▼─────────────────────────┐
         │ CLASSIFY PREDICTION                │
         │                                    │
         │ IF LN_IC50 < 0:                    │
         │   Classification = "Sensitive"     │
         │   Confidence = 0.87 (example)      │
         │   Meaning: Drug is effective       │
         │                                    │
         │ ELSE (LN_IC50 ≥ 0):                │
         │   Classification = "Resistant"     │
         │   Confidence = 0.78 (example)      │
         │   Meaning: Drug is ineffective     │
         │                                    │
         │ Sensitivity threshold: 0 (tunable) │
         └──────────┬─────────────────────────┘
                    │
         ┌──────────▼──────────────────────┐
         │ OPTIONAL: GENERATE EXPLANATION  │
         │ (SHAP Analysis)                 │
         │                                 │
         │ IF user requested explanations: │
         │ • Compute SHAP values           │
         │ • Identify top contributing     │
         │   features                      │
         │ • Generate explanations         │
         │                                 │
         │ Duration: 30-60 seconds         │
         │ (slow, optional)                │
         └──────────┬──────────────────────┘
                    │
         ┌──────────▼──────────────────┐
         │ FORMAT RESPONSE             │
         │                             │
         │ JSON Response:              │
         │ {                           │
         │   "prediction": {           │
         │     "ln_ic50": -0.4523,    │
         │     "classification":       │
         │       "Sensitive",          │
         │     "confidence": 0.87      │
         │   },                        │
         │   "explanation": {          │
         │     "top_features": [       │
         │       {                     │
         │         "rank": 1,          │
         │         "name": "Gene_123", │
         │         "shap": 0.045,      │
         │         "direction": "+"    │
         │       },                    │
         │       ...                   │
         │     ]                       │
         │   },                        │
         │   "metadata": {             │
         │     "model": "TabTransformer│
         │     "timestamp": "...",     │
         │     "version": "1.0"        │
         │   }                         │
         │ }                           │
         └──────────┬──────────────────┘
                    │
         ┌──────────▼──────────────────┐
         │ SEND RESPONSE               │
         │                             │
         │ • API: HTTP 200 + JSON     │
         │ • Web: Display on Streamlit │
         │ • CSV: Download as file     │
         │ • Plot: Visualization       │
         └──────────┬──────────────────┘
                    │
                    ▼
              OUTPUT TO USER
```

---

## 4. Explainability (SHAP) Workflow

```
                    ┌──────────────────────┐
                    │ START EXPLANATION    │
                    │ explain.py           │
                    └──────────┬───────────┘
                               │
                ┌──────────────▼───────────────┐
                │ PARSE ARGUMENTS              │
                │                              │
                │ --model-dir <path>           │
                │ --dataset TOYv2 (default)    │
                │ --n-background 100           │
                │ --n-explain 50               │
                │ --output-dir shap_results    │
                └──────────────┬───────────────┘
                               │
                ┌──────────────▼────────────────────┐
                │ LOAD TRAINED MODEL               │
                │                                  │
                │ Load from directory:             │
                │ • model.pt (weights)             │
                │ • hyperparameters.json           │
                │ • scaler.pkl                     │
                │                                  │
                │ TabTransformer.load(model_dir)   │
                └──────────────┬────────────────────┘
                               │
                ┌──────────────▼────────────────────┐
                │ LOAD DATASET & FEATURES          │
                │                                  │
                │ Dataset:                         │
                │ • Load GDSC/CCLE response data   │
                │ • Cell line IDs                  │
                │ • Drug IDs                       │
                │                                  │
                │ Features:                        │
                │ • Gene expression (978)          │
                │ • Drug fingerprints (2048)       │
                │                                  │
                │ load_dataset()                   │
                │ load_and_select_gene_features()  │
                │ load_drug_fingerprint_features() │
                └──────────────┬────────────────────┘
                               │
                ┌──────────────▼────────────────────┐
                │ EXTRACT FEATURE NAMES            │
                │                                  │
                │ Gene names from meta_info        │
                │ Fingerprint names: FP_bit_0,... │
                │ Total: ~3026 feature names       │
                │                                  │
                │ Output:                          │
                │ feature_names = [...]            │
                └──────────────┬────────────────────┘
                               │
                ┌──────────────▼────────────────────────┐
                │ BUILD FEATURE MATRIX FROM DATASET    │
                │                                      │
                │ For first N samples in dataset:      │
                │ 1. Get cell line ID: cl_id          │
                │ 2. Get drug ID: drug_id             │
                │ 3. Fetch gene expression: gex (978) │
                │ 4. Fetch fingerprint: fp (2048)     │
                │ 5. Concatenate: [gex, fp] (3026)   │
                │ 6. Append to feature matrix         │
                │                                      │
                │ N = n_background + n_explain        │
                │   = 100 + 50 = 150                  │
                │                                      │
                │ Output: X (150, 3026)               │
                └──────────────┬────────────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │ SPLIT INTO BACKGROUND & EXPLAIN │
                │                                 │
                │ X_background = X[:100]          │
                │   Shape: (100, 3026)            │
                │   Used for SHAP baseline        │
                │                                 │
                │ X_explain = X[100:150]          │
                │   Shape: (50, 3026)             │
                │   Samples to explain            │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────────────┐
                │ CREATE MODEL PREDICTION WRAPPER        │
                │                                        │
                │ def model_predict(x):                  │
                │   model.eval()                         │
                │   with torch.no_grad():                │
                │     tensor_x = torch.Tensor(x).float() │
                │     output = model(tensor_x)           │
                │   return output.cpu().numpy()          │
                │                                        │
                │ SHAP needs a function that:            │
                │ • Takes numpy array input              │
                │ • Returns numpy array output           │
                └──────────────┬──────────────────────────┘
                               │
                ┌──────────────▼───────────────────────────────┐
                │ INITIALIZE SHAP EXPLAINER (KernelExplainer)  │
                │                                              │
                │ explainer = shap.KernelExplainer(            │
                │     model_predict,                           │
                │     X_background                            │
                │ )                                            │
                │                                              │
                │ KernelExplainer:                            │
                │ • Model-agnostic                            │
                │ • Works with any model                      │
                │ • Slower than DeepExplainer                 │
                │ • More robust                               │
                │ • Number of background: 100                 │
                │   (larger = more accurate)                  │
                └──────────────┬───────────────────────────────┘
                               │
                ┌──────────────▼─────────────────────────────┐
                │ COMPUTE SHAP VALUES                       │
                │                                           │
                │ shap_values = explainer.shap_values(      │
                │     X_explain,                            │
                │     nsamples=100                          │
                │ )                                          │
                │                                           │
                │ For each sample in X_explain:             │
                │ 1. Generate synthetic samples             │
                │ 2. Get model predictions                  │
                │ 3. Fit weighted linear regression         │
                │ 4. Extract coefficients as SHAP values   │
                │ 5. Compute baseline (background mean)     │
                │                                           │
                │ Duration: 2-4 hours total (50 samples)   │
                │ Per sample: 2-5 min                       │
                │                                           │
                │ Output: shap_values (50, 3026)            │
                │ Each value = feature contribution         │
                │              to prediction                │
                └──────────────┬─────────────────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │ EXTRACT FEATURE IMPORTANCE      │
                │                                 │
                │ For each feature:               │
                │ importance[i] = mean(           │
                │   abs(shap_values[:, i])        │
                │ )                               │
                │                                 │
                │ Ranking:                        │
                │ ranked = argsort(importance)    │
                │ Descending order                │
                │                                 │
                │ Top 20 features selected        │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼────────────────────────┐
                │ GENERATE SUMMARY PLOT (Bar)          │
                │                                      │
                │ matplotlib + shap:                   │
                │ shap.summary_plot(                   │
                │     shap_values,                     │
                │     X_explain,                       │
                │     feature_names=feature_names,     │
                │     plot_type='bar',                 │
                │     max_display=20                   │
                │ )                                    │
                │                                      │
                │ Output:                              │
                │ • Horizontal bar chart               │
                │ • Top 20 features                    │
                │ • Feature importance scores          │
                │ • Saved as PNG (300 dpi)            │
                │                                      │
                │ File: shap_summary_bar.png           │
                └──────────────┬────────────────────────┘
                               │
                ┌──────────────▼────────────────────────┐
                │ GENERATE BEESWARM PLOT               │
                │                                      │
                │ shap.summary_plot(                   │
                │     shap_values,                     │
                │     X_explain,                       │
                │     feature_names=feature_names,     │
                │     max_display=20                   │
                │ )                                    │
                │                                      │
                │ Shows:                               │
                │ • Each sample as point               │
                │ • X-axis: SHAP value (impact)       │
                │ • Y-axis: Feature                    │
                │ • Color: Feature value               │
                │ • Density: Sample distribution       │
                │                                      │
                │ Reveals:                             │
                │ • Feature interactions               │
                │ • Directional effects               │
                │ • Non-linear relationships          │
                │                                      │
                │ File: shap_beeswarm.png              │
                └──────────────┬────────────────────────┘
                               │
                ┌──────────────▼──────────────────────┐
                │ SAVE TOP FEATURES TO JSON           │
                │                                    │
                │ Create list of dicts:              │
                │ [                                  │
                │   {                                │
                │     "rank": 1,                     │
                │     "feature": "Gene_12345",       │
                │     "mean_abs_shap": 0.0456        │
                │   },                               │
                │   {                                │
                │     "rank": 2,                     │
                │     "feature": "FP_bit_789",       │
                │     "mean_abs_shap": 0.0398        │
                │   },                               │
                │   ...                              │
                │ ]                                  │
                │                                    │
                │ Save to JSON file                  │
                │ File: top_features.json             │
                └──────────────┬──────────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │ PRINT SUMMARY TO CONSOLE        │
                │                                 │
                │ Display:                        │
                │ • Top 20 features               │
                │ • SHAP importance scores        │
                │ • Pretty-printed table          │
                │                                 │
                │ Example:                        │
                │ Rank Feature      SHAP          │
                │  1.  Gene_12345   0.0456        │
                │  2.  FP_bit_789   0.0398        │
                │  ...                           │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │ SAVE RESULTS SUMMARY            │
                │                                 │
                │ Directory created:              │
                │ shap_results/                   │
                │  ├─ shap_summary_bar.png        │
                │  ├─ shap_beeswarm.png           │
                │  └─ top_features.json           │
                │                                 │
                │ Output message:                 │
                │ "Results saved to               │
                │  shap_results/"                 │
                └──────────────┬──────────────────┘
                               │
                               ▼
                    EXPLANATION COMPLETE
                    Ready for visualization
                    & model interpretability
```

---

## 5. Web Interface (Streamlit) Workflow

```
┌──────────────────────────────────────────────────────┐
│   streamlit run app.py (Port: 8501)                 │
└──────────────────┬───────────────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │ LOAD PAGE CONFIG  │
         │                   │
         │ st.set_page_config│
         │ - Title           │
         │ - Layout: wide    │
         │ - Icon            │
         └─────────┬─────────┘
                   │
         ┌─────────▼────────────────┐
         │ RENDER TITLE & HEADER    │
         │                          │
         │ st.title()               │
         │ st.caption()             │
         │ Display banner           │
         └─────────┬────────────────┘
                   │
         ┌─────────▼──────────────────────────┐
         │ SIDEBAR CONFIGURATION              │
         │                                    │
         │ st.sidebar.header("Configuration") │
         │ ├─ model_dir input (text)         │
         │ └─ shap_dir input (text)          │
         │                                    │
         │ Allow users to configure:          │
         │ • Model directory path             │
         │ • SHAP results directory           │
         └─────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────────────────────┐
         │ CREATE TABS (3 main sections)      │
         │                                    │
         │ tab1: "Predict"                    │
         │ tab2: "Explainability"             │
         │ tab3: "About"                      │
         └─────────┬──────────────────────────┘
                   │
    ┌──────────────┼──────────────┬─────────────┐
    │              │              │             │
    ▼              ▼              ▼             ▼
┌──────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│TAB 1:    │  │TAB 2:   │  │TAB 3:    │  │...       │
│PREDICT  │  │EXPLAIN  │  │ABOUT     │  │          │
└────┬─────┘  └────┬────┘  └────┬─────┘  └──────────┘
     │             │            │
     │             │            │
     ▼             │            │
┌─────────────────────────────┐ │
│ TAB 1: PREDICTION INTERFACE │ │
├─────────────────────────────┤ │
│                             │ │
│ st.header("Drug Response    │ │
│            Prediction")      │ │
│                             │ │
│ ┌───────────────────────┐   │ │
│ │ LOAD MODEL            │   │ │
│ │                       │   │ │
│ │ @st.cache_resource    │   │ │
│ │ def load_model():     │   │ │
│ │   model = TabTrans... │   │ │
│ │   return model        │   │ │
│ │                       │   │ │
│ │ Display:              │   │ │
│ │ ✓ "Model loaded       │   │ │
│ │    successfully!"      │   │ │
│ │ or                    │   │ │
│ │ ✗ "Error loading      │   │ │
│ │    model"             │   │ │
│ └───────────────────────┘   │ │
│                             │ │
│ ┌───────────────────────┐   │ │
│ │ INPUT MODE SELECTION  │   │ │
│ │                       │   │ │
│ │ st.radio(             │   │ │
│ │   "Input mode",       │   │ │
│ │   ["Upload CSV",      │   │ │
│ │    "Manual Entry"]    │   │ │
│ │ )                     │   │ │
│ └───────────────────────┘   │ │
│          │                  │ │
│ ┌────────┴─────────────────┐ │
│ │                          │ │
│ ▼                          ▼ │
│ ┌──────────────┐   ┌──────────────────┐
│ │UPLOAD CSV    │   │MANUAL ENTRY      │
│ │              │   │                  │
│ │ Input:       │   │ Input:           │
│ │ • CSV file   │   │ • Text area      │
│ │   (features) │   │ • Comma-separated│
│ │              │   │   numbers        │
│ │ Process:     │   │                  │
│ │ • Load CSV   │   │ Process:         │
│ │ • Parse      │   │ • Split by comma │
│ │ • Validate   │   │ • Convert float  │
│ │ • Normalize  │   │ • Validate       │
│ │              │   │ • Normalize      │
│ │ Output:      │   │                  │
│ │ • DataFrame  │   │ Output:          │
│ │ • Display    │   │ • Numpy array    │
│ │   rows/cols  │   │                  │
│ │ • Button:    │   │ Button:          │
│ │   "Predict"  │   │ "Predict"        │
│ │              │   │                  │
│ │ ┌──────────┐ │   │ ┌──────────────┐
│ │ │ PREDICT  │ │   │ │ PREDICT      │
│ │ │ BUTTON   │ │   │ │ BUTTON       │
│ │ └──────┬───┘ │   │ └──────┬───────┘
│ │        │     │   │        │
│ │ ┌──────▼──────────────────────────┐
│ │ │ INFERENCE                        │
│ │ │ model.forward(X_normalized)      │
│ │ │                                  │
│ │ │ with torch.no_grad():           │
│ │ │   pred = model(tensor_x)         │
│ │ │                                  │
│ │ │ Output: LN_IC50 predictions      │
│ │ └──────┬──────────────────────────┘
│ │        │
│ │ ┌──────▼──────────────────────────────┐
│ │ │ DISPLAY RESULTS                      │
│ │ │                                      │
│ │ │ • Results DataFrame:                 │
│ │ │   ├─ Sample ID                       │
│ │ │   ├─ Predicted LN_IC50               │
│ │ │   └─ Classification (Sensitive/      │
│ │ │       Resistant)                     │
│ │ │                                      │
│ │ │ st.dataframe(results_df)             │
│ │ │                                      │
│ │ │ • Distribution Plot:                 │
│ │ │   ├─ Histogram of predictions        │
│ │ │   ├─ Threshold line (LN_IC50=0)      │
│ │ │   └─ matplotlib figure               │
│ │ │                                      │
│ │ │ st.pyplot(fig)                       │
│ │ │                                      │
│ │ │ • Download Button:                   │
│ │ │   ├─ CSV format                      │
│ │ │   └─ "predictions.csv"               │
│ │ │                                      │
│ │ │ st.download_button(...)              │
│ │ └──────────────────────────────────────┘
│ │
│ └──────────────────────────────────────────┘
│                                             │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ TAB 2: EXPLAINABILITY (SHAP)                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│ st.header("Model Explainability (SHAP)")          │
│                                                     │
│ Check if SHAP results exist:                        │
│ if os.path.exists(shap_dir):                       │
│   ├─ Load SHAP visualization files                │
│   │  ├─ shap_summary_bar.png                      │
│   │  └─ shap_beeswarm.png                         │
│   │                                                │
│   ├─ Display in two columns:                      │
│   │  col1: Bar plot                               │
│   │  col2: Beeswarm plot                          │
│   │                                                │
│   │  st.image(bar_path)                           │
│   │  st.image(bee_path)                           │
│   │                                                │
│   ├─ Load and display top features:               │
│   │  ├─ Load top_features.json                    │
│   │  ├─ Parse JSON                                │
│   │  ├─ Create DataFrame                          │
│   │  └─ st.dataframe()                            │
│   │                                                │
│   └─ Show feature rankings:                        │
│      ├─ Rank                                      │
│      ├─ Feature name                              │
│      ├─ SHAP importance                           │
│      └─ Direction of effect                       │
│                                                     │
│ else:                                               │
│   └─ Display message:                              │
│      "Run explain.py first:"                       │
│      "python explain.py --model-dir ..."          │
│                                                     │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ TAB 3: ABOUT                                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│ st.header("About PharmaAI Predictor")              │
│                                                     │
│ Display:                                            │
│ • Project description (markdown)                   │
│ • System architecture                              │
│ • Input format description                         │
│ • Transformer architecture explanation             │
│ • Datasets overview                                │
│ • How to use guide                                 │
│ • Training instructions                            │
│ • Built with (libraries)                           │
│ • Links to documentation                           │
│                                                     │
│ st.markdown("""                                    │
│   **PharmaAI Predictor** uses a Transformer...    │
│   ...                                              │
│ """)                                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 6. REST API Request/Response Flow

```
CLIENT REQUEST (HTTP)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼ (Example 1)                         ▼ (Example 2)
┌──────────────────────┐          ┌──────────────────────────┐
│ PREDICT SINGLE       │          │ BATCH PREDICT            │
│                      │          │                          │
│ POST /predict        │          │ POST /predict/batch      │
│ Content-Type:        │          │ Content-Type:            │
│   application/json   │          │   application/json       │
│                      │          │                          │
│ {                    │          │ {                        │
│   "features": [      │          │   "features": [          │
│     0.5, -0.3,      │          │     [0.5, -0.3, ...],   │
│     1.2, ...         │          │     [0.2, 0.1, ...]     │
│   ]                  │          │   ]                      │
│ }                    │          │ }                        │
└──────────┬───────────┘          └──────────┬───────────────┘
           │                                 │
           │ HTTP POST                       │ HTTP POST
           │                                 │
           ▼                                 ▼
┌──────────────────────────────────────────────────────────┐
│  FASTAPI ROUTER (/predict or /predict/batch)            │
└──────────────┬───────────────────────────────────────────┘
               │
      ┌────────▼───────────┐
      │ ROUTE HANDLER      │
      │                    │
      │ Receives:          │
      │ • request body     │
      │ • query params     │
      │ • path params      │
      └────────┬───────────┘
               │
      ┌────────▼──────────────────────┐
      │ REQUEST VALIDATION            │
      │                               │
      │ Pydantic model validation:    │
      │ • Type checking               │
      │ • Required fields             │
      │ • Constraints                 │
      │                               │
      │ If invalid:                   │
      │ → Return 422 error            │
      └────────┬──────────────────────┘
               │
      ┌────────▼──────────────────────┐
      │ DEPENDENCY INJECTION          │
      │                               │
      │ FastAPI resolves:             │
      │ • PredictorService            │
      │ • ExplainerService            │
      │ • Database connections        │
      │ • Config settings             │
      └────────┬──────────────────────┘
               │
      ┌────────▼──────────────────────────┐
      │ CALL BUSINESS LOGIC SERVICE       │
      │ (PredictorService)                │
      │                                   │
      │ predict_service.predict_single()  │
      │ or                                │
      │ predict_service.predict_batch()   │
      └────────┬──────────────────────────┘
               │
      ┌────────▼─────────────────────────┐
      │ SERVICE PROCESSING               │
      │                                  │
      │ 1. Input validation              │
      │ 2. Load model & scaler           │
      │ 3. Normalize features            │
      │ 4. Run inference                 │
      │ 5. Format results                │
      │ 6. Optional: Get explanations    │
      │ 7. Return response object        │
      └────────┬─────────────────────────┘
               │
      ┌────────▼────────────────────────────────────┐
      │ RESPONSE SERIALIZATION                      │
      │                                             │
      │ FastAPI auto-converts response to JSON:    │
      │                                             │
      │ {                                           │
      │   "prediction": {                           │
      │     "ln_ic50": -0.4523,                    │
      │     "classification": "Sensitive",          │
      │     "confidence": 0.87                      │
      │   },                                        │
      │   "explanation": {                          │
      │     "top_features": [...]                   │
      │   },                                        │
      │   "metadata": {                             │
      │     "timestamp": "2025-04-18T...",         │
      │     "model": "TabTransformer",              │
      │     "version": "1.0"                        │
      │   }                                         │
      │ }                                           │
      └────────┬────────────────────────────────────┘
               │
      ┌────────▼──────────────────┐
      │ SET HTTP RESPONSE HEADERS  │
      │                            │
      │ • Status Code: 200         │
      │ • Content-Type: application│
      │   /json; charset=utf-8    │
      │ • Content-Length           │
      │ • Cache-Control            │
      │ • X-Request-ID             │
      └────────┬──────────────────┘
               │
      ┌────────▼──────────────────────┐
      │ MIDDLEWARE LOGGING            │
      │                               │
      │ Log:                          │
      │ • Request method & path       │
      │ • Request size                │
      │ • Response size               │
      │ • Processing time             │
      │ • Status code                 │
      │ • Timestamp                   │
      └────────┬──────────────────────┘
               │
               ▼
      HTTP 200 OK
      Response JSON
            │
            ▼
      CLIENT RECEIVES RESPONSE
```

---

## 7. Data Transformation Pipeline

```
RAW INPUT DATA
    │
    ├─ CSV Format:
    │  Sample1, 0.5, -0.3, 1.2, ...
    │  Sample2, 0.2, 0.1, -0.5, ...
    │
    ├─ JSON Format:
    │  {"features": [0.5, -0.3, 1.2, ...]}
    │
    └─ Manual Entry:
       "0.5, -0.3, 1.2, ..."
           │
           ▼
    ┌──────────────────────────────────┐
    │ PARSE INPUT                      │
    │                                  │
    │ • Read file or string            │
    │ • Handle different encodings     │
    │ • Check for malformed data       │
    │ • Extract numeric values         │
    └──────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │ VALIDATION LAYER 1: TYPE         │
    │                                  │
    │ Ensure all values are numeric:   │
    │ • float or int                   │
    │ • Not string, None, or object    │
    │                                  │
    │ Convert: str → float             │
    │ Reject: non-numeric strings      │
    └──────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │ VALIDATION LAYER 2: DIMENSION    │
    │                                  │
    │ Check feature count:             │
    │ • Expected: 3026                 │
    │ • Actual: len(features)          │
    │                                  │
    │ If mismatch:                     │
    │ • Pad with zeros (if too small)  │
    │ • Truncate (if too large)        │
    │ • Warn user                      │
    └──────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │ VALIDATION LAYER 3: VALUES       │
    │                                  │
    │ Check for:                       │
    │ • NaN (Not a Number)             │
    │ • Inf (Infinity)                 │
    │ • Extremely large/small values   │
    │ • Outliers                       │
    │                                  │
    │ Handle:                          │
    │ • Replace NaN with 0             │
    │ • Clip extreme values            │
    │ • Log warning if outliers        │
    └──────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │ CONVERSION TO NUMPY ARRAY        │
    │                                  │
    │ X = np.array(features, dtype=    │
    │     np.float32)                  │
    │                                  │
    │ Shape: (1, 3026) for single      │
    │        (n, 3026) for batch       │
    │                                  │
    │ Memory: ~12 KB per sample        │
    └──────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │ LOAD SCALER (StandardScaler)    │
    │                                  │
    │ Load: scaler.pkl                 │
    │                                  │
    │ Contains:                        │
    │ • mean_ (3026,)                  │
    │ • scale_ (3026,)                 │
    │ • n_features_in_: 3026           │
    │ • feature_names_in_: [...]       │
    │                                  │
    │ Fitted on: Training dataset      │
    │ (GDSC2 or other)                 │
    └──────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │ NORMALIZATION (Standardization)  │
    │                                  │
    │ For each feature i:              │
    │ X_norm[i] = (X[i] - mean[i])     │
    │             / scale[i]           │
    │                                  │
    │ Properties:                      │
    │ • Mean becomes 0                 │
    │ • Std becomes 1                  │
    │ • Range: typically [-5, +5]      │
    │ • Numerical stability            │
    │                                  │
    │ Output: X_norm (same shape)      │
    │ dtype: float32                   │
    └──────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │ CONVERSION TO PYTORCH TENSOR     │
    │                                  │
    │ tensor_x = torch.from_numpy(     │
    │   X_norm                         │
    │ ).float()                        │
    │                                  │
    │ Shape: (batch, 3026)             │
    │ Dtype: torch.float32             │
    │ Device: CPU or GPU               │
    │                                  │
    │ Requires no gradients:           │
    │ with torch.no_grad()             │
    └──────────────────────────────────┘
           │
           ▼
    READY FOR MODEL INFERENCE
```

---

## 8. Error Handling Flow

```
REQUEST RECEIVED
    │
    ▼
┌─────────────────────────────────┐
│ TRY: Attempt to process request │
└──────────────┬──────────────────┘
               │
    ┌──────────▼──────────┐
    │ Step 1: Parse input │
    └──────────┬──────────┘
               │
    ┌─ EXCEPT ─▼────────────────────────────────┐
    │ ValueError: Invalid JSON or format        │
    ├─ Action:                                  │
    │ • Catch exception                         │
    │ • Log error details                       │
    │ • Return HTTP 400: Bad Request            │
    │ • Message: "Invalid input format"         │
    └──────────────────────────────────────────┘
               │
    ┌──────────▼──────────────────────┐
    │ Step 2: Validate dimensions      │
    └──────────┬──────────────────────┘
               │
    ┌─ EXCEPT ─▼────────────────────────────────┐
    │ ValueError: Feature count mismatch        │
    ├─ Action:                                  │
    │ • Catch exception                         │
    │ • Log feature count: expected vs actual   │
    │ • Return HTTP 422: Unprocessable Entity   │
    │ • Message: "Expected 3026 features, got   │
    │            {count}"                       │
    └──────────────────────────────────────────┘
               │
    ┌──────────▼──────────────────────┐
    │ Step 3: Load model & scaler      │
    └──────────┬──────────────────────┘
               │
    ┌─ EXCEPT ─▼────────────────────────────────┐
    │ FileNotFoundError: Model not found        │
    ├─ Action:                                  │
    │ • Catch exception                         │
    │ • Log missing file path                   │
    │ • Return HTTP 500: Internal Server Error  │
    │ • Message: "Model not found at {path}"    │
    │ • Include instructions to train model     │
    └──────────────────────────────────────────┘
               │
    ┌─ EXCEPT ─▼────────────────────────────────┐
    │ RuntimeError: CUDA out of memory          │
    ├─ Action:                                  │
    │ • Catch exception                         │
    │ • Log GPU memory error                    │
    │ • Fall back to CPU inference              │
    │ • Return HTTP 200 (slower response)       │
    │ • Include timing info in response         │
    └──────────────────────────────────────────┘
               │
    ┌──────────▼──────────────────────┐
    │ Step 4: Run inference            │
    └──────────┬──────────────────────┘
               │
    ┌─ EXCEPT ─▼────────────────────────────────┐
    │ RuntimeError: Tensor dimension mismatch   │
    ├─ Action:                                  │
    │ • Catch exception                         │
    │ • Log tensor shapes                       │
    │ • Return HTTP 500: Internal Server Error  │
    │ • Message: "Model execution failed"       │
    │ • Include debug info                      │
    └──────────────────────────────────────────┘
               │
    ┌──────────▼──────────────────────┐
    │ Step 5: Format & return response │
    └──────────┬──────────────────────┘
               │
    ┌─ EXCEPT ─▼────────────────────────────────┐
    │ Exception: Unexpected error                │
    ├─ Action:                                  │
    │ • Catch generic exception                 │
    │ • Log full stack trace                    │
    │ • Return HTTP 500: Internal Server Error  │
    │ • Generic error message (no details)      │
    │ • Request user to contact support         │
    └──────────────────────────────────────────┘
               │
    ┌──────────▼─────────────────────────────┐
    │ FINALLY: Always execute                │
    │                                        │
    │ • Log request completion               │
    │ • Record execution time                │
    │ • Update metrics                       │
    │ • Close file handles                   │
    │ • Release GPU memory                   │
    └──────────┬──────────────────────────────┘
               │
    ┌──────────▼──────────────┐
    │ SUCCESS: Return HTTP 200 │
    │ + JSON response          │
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────────┐
    │ ERROR: Return HTTP 4xx/5xx  │
    │ + Error JSON                │
    │                             │
    │ Error Response Format:      │
    │ {                           │
    │   "error": {                │
    │     "code": 422,            │
    │     "message": "...",       │
    │     "details": {...},       │
    │     "timestamp": "..."      │
    │   }                         │
    │ }                           │
    └──────────────────────────────┘
```

---

## 9. Cross-Validation & Evaluation Flow

```
FOR EACH FOLD (1 to 5):
    │
    ├─ Training Set (70%):      Used to train model
    ├─ Validation Set (15%):    Used for early stopping, HP tuning
    └─ Test Set (15%):          Held out for final evaluation
    │
    ▼
┌──────────────────────────────────────────┐
│ EPOCH LOOP (max 100 epochs)              │
├──────────────────────────────────────────┤
│                                          │
│ FOR EACH EPOCH:                          │
│                                          │
│ 1. TRAINING PHASE                        │
│    ├─ Iterate over training batches      │
│    ├─ Forward pass                       │
│    ├─ Compute loss (MSE)                │
│    ├─ Backward pass (compute gradients)  │
│    ├─ Update weights (optimizer step)    │
│    └─ Accumulate training metrics        │
│                                          │
│ 2. VALIDATION PHASE                      │
│    ├─ Iterate over validation batches    │
│    ├─ Forward pass (no_grad)            │
│    ├─ Compute validation loss            │
│    ├─ Compute metrics (R², correlation)  │
│    ├─ Check early stopping criterion     │
│    └─ Log metrics                        │
│                                          │
│ 3. CHECKPOINT MANAGEMENT                 │
│    ├─ Compare val loss to best           │
│    ├─ If improved:                       │
│    │  └─ Save model weights              │
│    ├─ If no improvement for N epochs:    │
│    │  └─ Stop training (early stopping)  │
│    └─ Log improvement                    │
│                                          │
│ 4. LOGGING & MONITORING                  │
│    ├─ Log loss per batch                 │
│    ├─ Log metrics per epoch              │
│    ├─ Monitor GPU memory                 │
│    ├─ Monitor training time              │
│    └─ Print progress bar                 │
│                                          │
│ END EPOCH LOOP                           │
│ (Stop when early stopping triggered)     │
│                                          │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ LOAD BEST MODEL CHECKPOINT               │
│                                          │
│ • Restore weights from best epoch        │
│ • Ensure reproducibility                 │
│ • Discard later epochs                   │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ TEST SET EVALUATION                      │
│                                          │
│ 1. Prepare test data                     │
│ 2. Set model to eval mode                │
│ 3. Make predictions on test set          │
│ 4. Compare to ground truth               │
│ 5. Compute metrics:                      │
│    ├─ R² Score                           │
│    ├─ MSE / RMSE                         │
│    ├─ MAE                                │
│    ├─ Spearman Correlation               │
│    ├─ Pearson Correlation                │
│    └─ Other task-specific metrics        │
│                                          │
│ Output: Fold metrics dict                │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ SAVE FOLD RESULTS                        │
│                                          │
│ Store:                                   │
│ • Metrics for this fold                  │
│ • Predictions (y_pred)                   │
│ • Ground truth (y_true)                  │
│ • Model checkpoint (best weights)        │
│ • Training log (loss per epoch)          │
│                                          │
│ Location: results/{run_id}/{model_name}/ │
└──────────────────────────────────────────┘
    │
    ▼
AGGREGATE ACROSS ALL FOLDS:
    │
    ├─ Collect metrics from all 5 folds
    ├─ Compute mean and std deviation
    ├─ Compute 95% confidence intervals
    ├─ Identify best and worst performing folds
    └─ Generate summary report
    │
    ▼
┌──────────────────────────────────────────┐
│ FINAL METRICS SUMMARY                    │
│                                          │
│ R² Score:                                │
│   Mean: 0.78 ± 0.03 (95% CI: 0.75-0.81) │
│   Best Fold: 0.82, Worst: 0.74           │
│                                          │
│ RMSE:                                    │
│   Mean: 0.62 ± 0.04                      │
│   Best Fold: 0.58, Worst: 0.67           │
│                                          │
│ Spearman Corr:                           │
│   Mean: 0.81 ± 0.02                      │
│   Best Fold: 0.83, Worst: 0.78           │
│                                          │
│ Training Time:                           │
│   Per fold: 2-4 hours                    │
│   Total (5 folds): 30-40 hours           │
│                                          │
│ Save: experiment_results.csv              │
└──────────────────────────────────────────┘
```

---

## 10. System Initialization Sequence

```
APPLICATION START
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 1. ENVIRONMENT SETUP                                │
│                                                     │
│ • Load .env file (if exists)                       │
│ • Set CUDA_VISIBLE_DEVICES (GPU selection)         │
│ • Configure logging level                          │
│ • Set random seeds (reproducibility)               │
│   ├─ Python: random.seed()                         │
│   ├─ NumPy: np.random.seed()                       │
│   └─ PyTorch: torch.manual_seed()                  │
│ • Check GPU availability (CUDA, cuDNN)             │
│ • Display system info                              │
│   ├─ OS & Python version                           │
│   ├─ PyTorch version                               │
│   ├─ CUDA capabilities                             │
│   └─ Available memory (CPU & GPU)                  │
│                                                     │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│ 2. DEPENDENCY VERIFICATION                          │
│                                                     │
│ Check installed packages:                           │
│ ✓ torch >= 2.0.0                                   │
│ ✓ pytorch-lightning >= 2.0.0                       │
│ ✓ drevalpy >= 1.4.0                                │
│ ✓ fastapi >= 0.104.0                               │
│ ✓ streamlit >= 1.28.0                              │
│ ✓ shap >= 0.42.0                                   │
│ ✓ scikit-learn >= 1.3.0                            │
│ ...                                                │
│                                                     │
│ If missing:                                        │
│ → Raise ImportError with install instructions      │
│                                                     │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│ 3. DIRECTORY STRUCTURE VERIFICATION                 │
│                                                     │
│ Check/Create directories:                          │
│ ✓ ./data/              (datasets)                  │
│ ✓ ./results/           (models & logs)             │
│ ✓ ./shap_results/      (explanations)              │
│ ✓ ./models/            (code)                      │
│ ✓ ./api/               (API code)                  │
│                                                     │
│ If missing:                                        │
│ → Create directory with os.makedirs()              │
│                                                     │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│ 4. CONFIGURATION LOADING                            │
│                                                     │
│ Load configuration from:                           │
│ • .env file (environment variables)                │
│ • config.yaml (if exists)                          │
│ • Command-line arguments                           │
│ • Defaults (hardcoded)                             │
│                                                     │
│ Configuration includes:                            │
│ • Dataset paths                                    │
│ • Model directories                                │
│ • Hyperparameters                                  │
│ • GPU settings                                     │
│ • Logging configuration                            │
│                                                     │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│ 5. LOGGING INITIALIZATION                           │
│                                                     │
│ Configure logging system:                          │
│ • Log level: DEBUG, INFO, WARNING, ERROR           │
│ • Log format: timestamp, level, message            │
│ • Log handlers:                                    │
│   ├─ Console (stdout)                              │
│   └─ File (logs/app.log)                           │
│                                                     │
│ First log message:                                 │
│ "Application started successfully"                 │
│                                                     │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│ 6. MODEL REGISTRY INITIALIZATION                    │
│                                                     │
│ Import and register models:                        │
│ • Register TabTransformer into MODEL_FACTORY       │
│ • Import baseline models                           │
│ • Verify all models loaded                         │
│                                                     │
│ Execute: import register_model                     │
│ Output: "Registered TabTransformer into            │
│          MODEL_FACTORY. Available models: X"       │
│                                                     │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│ 7. APPLICATION STARTUP                              │
│                                                     │
│ IF Application Type = Streamlit:                   │
│ ├─ Initialize Streamlit session state              │
│ ├─ Load cached resources                           │
│ ├─ Display welcome page                            │
│ └─ Start interactive server (Port 8501)            │
│                                                     │
│ IF Application Type = FastAPI:                     │
│ ├─ Create FastAPI app instance                     │
│ ├─ Register routes                                 │
│ ├─ Initialize middleware                           │
│ └─ Start Uvicorn server (Port 8000)                │
│                                                     │
│ IF Application Type = Training:                    │
│ ├─ Parse command-line arguments                    │
│ ├─ Load primary dataset                            │
│ ├─ Load features                                   │
│ └─ Start training pipeline                         │
│                                                     │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
        APPLICATION READY FOR USE
```

---

## Summary

This comprehensive flow diagram documentation covers:

1. **End-to-End System Flow**: Complete data journey from input to output
2. **Training Pipeline**: Detailed model training process with cross-validation
3. **Prediction Workflow**: Single sample inference with all preprocessing steps
4. **Explainability Pipeline**: SHAP-based feature importance generation
5. **Web Interface**: Streamlit application interaction flows
6. **REST API**: HTTP request/response patterns
7. **Data Transformation**: Feature preprocessing and normalization
8. **Error Handling**: Comprehensive exception handling strategy
9. **Cross-Validation**: Training loop and evaluation methodology
10. **System Initialization**: Application startup sequence

Each diagram includes detailed step-by-step processes, decision trees, and data transformations that occur within the PharmaAI Predictor system.

---

**Document Version**: 1.0  
**Last Updated**: April 2025
