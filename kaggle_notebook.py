# ============================================================
# PharmaAI Predictor - Kaggle Training Notebook
# ============================================================
# HOW TO USE:
# 1. Go to kaggle.com -> New Notebook
# 2. Settings (right panel) -> Accelerator -> GPU T4 x2
# 3. Copy-paste this entire file into a single code cell
# 4. Click "Run All"
# 5. After training, download results from /kaggle/working/results/
# ============================================================

# --- Step 1: Install drevalpy ---
import subprocess
subprocess.check_call(["pip", "install", "drevalpy[multiprocessing]", "shap", "-q"])

# --- Step 2: Define the Transformer Network ---
import os
import json
import secrets
import warnings
import platform

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from sklearn.preprocessing import StandardScaler
import joblib

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import (
    load_and_select_gene_features,
    load_drug_fingerprint_features,
    scale_gene_expression,
)
from drevalpy.models import MODEL_FACTORY, MULTI_DRUG_MODEL_FACTORY


class RegressionDataset(Dataset):
    """Dataset that concatenates cell line + drug features for regression."""

    def __init__(self, output, cell_line_input, drug_input, cell_line_views, drug_views):
        self.cell_line_views = cell_line_views
        self.drug_views = drug_views
        self.output = output
        self.cell_line_input = cell_line_input
        self.drug_input = drug_input

    def __getitem__(self, idx):
        cell_line_id = self.output.cell_line_ids[idx]
        drug_id = self.output.drug_ids[idx]
        response = self.output.response[idx]

        cell_line_features = None
        for cl_view in self.cell_line_views:
            feat = self.cell_line_input.features[cell_line_id][cl_view]
            cell_line_features = feat if cell_line_features is None else np.concatenate((cell_line_features, feat))

        drug_features = None
        for d_view in self.drug_views:
            feat = self.drug_input.features[drug_id][d_view]
            drug_features = feat if drug_features is None else np.concatenate((drug_features, feat))

        data = np.concatenate((cell_line_features, drug_features)).astype(np.float32)
        return data, np.float32(response)

    def __len__(self):
        return len(self.output.response)


class TransformerDRPNetwork(pl.LightningModule):
    """Transformer-based network for drug response prediction."""

    def __init__(self, hyperparameters, input_dim):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hyperparameters.get("hidden_dim", 128)
        self.num_layers = hyperparameters.get("num_layers", 4)
        self.num_heads = hyperparameters.get("num_heads", 8)
        self.dropout_prob = hyperparameters.get("dropout_prob", 0.1)
        self.token_size = hyperparameters.get("token_size", 64)
        self.input_dim = input_dim

        self.n_tokens = (input_dim + self.token_size - 1) // self.token_size
        self.padded_dim = self.n_tokens * self.token_size

        self.token_embed = nn.Linear(self.token_size, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens + 1, self.hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self.loss = nn.MSELoss()
        self.checkpoint_callback = None

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] < self.padded_dim:
            padding = torch.zeros(batch_size, self.padded_dim - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)

        x = x.view(batch_size, self.n_tokens, self.token_size)
        x = self.token_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        x = self.transformer(x)
        x = self.layer_norm(x)

        cls_output = x[:, 0, :]
        return self.reg_head(cls_output).squeeze(-1)

    def _forward_loss_and_log(self, x, y, log_as):
        y_pred = self.forward(x)
        result = self.loss(y_pred, y)
        self.log(log_as, result, on_step=True, on_epoch=True, prog_bar=True)
        return result

    def training_step(self, batch):
        x, y = batch
        return self._forward_loss_and_log(x, y, "train_loss")

    def validation_step(self, batch):
        x, y = batch
        return self._forward_loss_and_log(x, y, "val_loss")

    def predict_numpy(self, x):
        is_training = self.training
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(torch.from_numpy(x).float().to(self.device))
        self.train(is_training)
        return y_pred.cpu().detach().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 100
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def fit(
        self,
        output_train,
        cell_line_input,
        drug_input,
        cell_line_views,
        drug_views,
        output_earlystopping=None,
        trainer_params=None,
        batch_size=32,
        patience=5,
        num_workers=2,
        model_checkpoint_dir="checkpoints",
    ):
        if trainer_params is None:
            trainer_params = {"max_epochs": 100, "progress_bar_refresh_rate": 500}

        train_dataset = RegressionDataset(
            output=output_train,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_views=cell_line_views,
            drug_views=drug_views,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, persistent_workers=True, drop_last=True,
        )

        val_loader = None
        if output_earlystopping is not None:
            val_dataset = RegressionDataset(
                output=output_earlystopping,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                cell_line_views=cell_line_views,
                drug_views=drug_views,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, persistent_workers=True,
            )

        monitor = "train_loss" if val_loader is None else "val_loss"
        early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=patience)

        unique_subfolder = os.path.join(model_checkpoint_dir, "run_" + secrets.token_hex(8))
        os.makedirs(unique_subfolder, exist_ok=True)

        name = "version-" + "".join([secrets.choice("0123456789abcdef") for _ in range(10)])
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=unique_subfolder, monitor=monitor, mode="min",
            save_top_k=1, filename=name,
        )

        progress_bar = TQDMProgressBar(
            refresh_rate=trainer_params.get("progress_bar_refresh_rate", 500)
        )
        trainer_params_copy = {
            k: v for k, v in trainer_params.items() if k != "progress_bar_refresh_rate"
        }

        trainer = pl.Trainer(
            callbacks=[early_stop_callback, self.checkpoint_callback, progress_bar],
            default_root_dir=model_checkpoint_dir,
            devices=1,
            accelerator="auto",  # auto-detect GPU on Kaggle
            **trainer_params_copy,
        )
        if val_loader is None:
            trainer.fit(self, train_loader)
        else:
            trainer.fit(self, train_loader, val_loader)

        if self.checkpoint_callback.best_model_path:
            checkpoint = torch.load(self.checkpoint_callback.best_model_path, weights_only=True)
            self.load_state_dict(checkpoint["state_dict"])


class TabTransformer(DRPModel):
    """Transformer-based model for drug response prediction."""

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self):
        super().__init__()
        self.model = None
        self.hyperparameters = None
        self.gene_expression_scaler = StandardScaler()

    @classmethod
    def get_model_name(cls):
        return "TabTransformer"

    def build_model(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.hyperparameters.setdefault("input_dim_gex", None)
        self.hyperparameters.setdefault("input_dim_fp", None)

    def train(self, output, cell_line_input, drug_input=None,
              output_earlystopping=None, model_checkpoint_dir="checkpoints"):
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for TabTransformer.")

        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                gene_expression_scaler=self.gene_expression_scaler,
            )

        dim_gex = next(iter(cell_line_input.features.values()))["gene_expression"].shape[0]
        dim_fingerprint = next(iter(drug_input.features.values()))[self.drug_views[0]].shape[0]
        self.hyperparameters["input_dim_gex"] = dim_gex
        self.hyperparameters["input_dim_fp"] = dim_fingerprint

        self.model = TransformerDRPNetwork(
            hyperparameters=self.hyperparameters,
            input_dim=dim_gex + dim_fingerprint,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have many workers.*")
            warnings.filterwarnings("ignore", message="Starting from v1\\.9\\.0.*")

            if output_earlystopping is not None and len(output_earlystopping) == 0:
                output_earlystopping = output

            self.model.fit(
                output_train=output,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                cell_line_views=self.cell_line_views,
                drug_views=self.drug_views,
                output_earlystopping=output_earlystopping,
                trainer_params={
                    "max_epochs": self.hyperparameters.get("max_epochs", 100),
                    "progress_bar_refresh_rate": 500,
                },
                batch_size=self.hyperparameters.get("batch_size", 32),
                patience=self.hyperparameters.get("patience", 5),
                num_workers=4,  # Kaggle has enough CPUs
                model_checkpoint_dir=model_checkpoint_dir,
            )

    def predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input=None):
        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                gene_expression_scaler=self.gene_expression_scaler,
            )

        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view=self.drug_views[0],
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict_numpy(x)

    def load_cell_line_features(self, data_path, dataset_name):
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes_reduced",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path, dataset_name):
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))
        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)
        joblib.dump(self.gene_expression_scaler, os.path.join(directory, "scaler.pkl"))

    @classmethod
    def load(cls, directory):
        instance = cls()
        with open(os.path.join(directory, "hyperparameters.json")) as f:
            instance.hyperparameters = json.load(f)
        instance.gene_expression_scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
        dim_gex = instance.hyperparameters["input_dim_gex"]
        dim_fp = instance.hyperparameters["input_dim_fp"]
        instance.model = TransformerDRPNetwork(instance.hyperparameters, input_dim=dim_gex + dim_fp)
        instance.model.load_state_dict(torch.load(os.path.join(directory, "model.pt"), weights_only=True))
        instance.model.eval()
        return instance

    @classmethod
    def get_hyperparameter_set(cls):
        """Return hyperparameter combinations for grid search."""
        return [
            {"hidden_dim": 128, "num_layers": 2, "num_heads": 4, "dropout_prob": 0.1,
             "token_size": 64, "batch_size": 32, "patience": 5, "max_epochs": 100},
            {"hidden_dim": 128, "num_layers": 4, "num_heads": 8, "dropout_prob": 0.1,
             "token_size": 64, "batch_size": 32, "patience": 5, "max_epochs": 100},
        ]


# --- Step 3: Register the model ---
MODEL_FACTORY["TabTransformer"] = TabTransformer
MULTI_DRUG_MODEL_FACTORY["TabTransformer"] = TabTransformer
print(f"TabTransformer registered. {len(MODEL_FACTORY)} models available.")

# --- Step 4: Check GPU ---
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- Step 5: Run the experiment ---
from drevalpy.experiment import drug_response_experiment
from drevalpy.datasets.loader import load_dataset

# === CONFIGURATION ===
# Change these settings as needed:
DATASET = "GDSC2"          # Options: GDSC1, GDSC2, CCLE, TOYv2 (for testing)
TEST_MODE = "LCO"           # LCO = Leave-Cell-Line-Out (paper's main strategy)
N_CV_SPLITS = 5             # 5-fold cross-validation
MEASURE = "LN_IC50"         # Drug response measure
HPAM_TUNING = True          # Hyperparameter tuning
RUN_BASELINES = True        # Compare against baselines
CROSS_STUDY = None           # Set to "CCLE" for cross-study validation
# =====================

print(f"\nLoading dataset: {DATASET} (measure: {MEASURE})")
response_data = load_dataset(dataset_name=DATASET, path_data="data", measure=MEASURE)

cross_study_datasets = None
if CROSS_STUDY:
    print(f"Loading cross-study dataset: {CROSS_STUDY}")
    cross_study_data = load_dataset(dataset_name=CROSS_STUDY, path_data="data", measure=MEASURE)
    cross_study_datasets = [cross_study_data]

models = [MODEL_FACTORY["TabTransformer"]]

baselines = None
if RUN_BASELINES:
    baselines = [
        MODEL_FACTORY["ElasticNet"],
        MODEL_FACTORY["SimpleNeuralNetwork"],
        MODEL_FACTORY["RandomForest"],
    ]

print(f"\nStarting experiment:")
print(f"  Dataset: {DATASET}")
print(f"  Test mode: {TEST_MODE}")
print(f"  CV splits: {N_CV_SPLITS}")
print(f"  HP tuning: {HPAM_TUNING}")
print(f"  Cross-study: {CROSS_STUDY or 'None'}")
print()

drug_response_experiment(
    models=models,
    baselines=baselines,
    response_data=response_data,
    n_cv_splits=N_CV_SPLITS,
    test_mode=TEST_MODE,
    run_id="PharmaAI_Transformer_2025",
    path_data="data",
    path_out="results",
    hyperparameter_tuning=HPAM_TUNING,
    cross_study_datasets=cross_study_datasets,
)

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE!")
print("=" * 60)

# --- Step 6: Analyze results ---
import pandas as pd
from scipy.stats import pearsonr, spearmanr

print("\n--- RESULTS SUMMARY ---\n")

results_base = f"results/PharmaAI_Transformer_2025/{DATASET}/{TEST_MODE}"
model_names = ["TabTransformer"]
if RUN_BASELINES:
    model_names += ["ElasticNet", "SimpleNeuralNetwork", "RandomForest", "NaiveMeanEffectsPredictor"]

for model_name in model_names:
    pred_dir = os.path.join(results_base, model_name, "predictions")
    if not os.path.exists(pred_dir):
        continue

    all_preds = []
    for split in range(N_CV_SPLITS):
        fpath = os.path.join(pred_dir, f"predictions_split_{split}.csv")
        if os.path.exists(fpath):
            all_preds.append(pd.read_csv(fpath))

    if not all_preds:
        continue

    df = pd.concat(all_preds)
    r_pearson, _ = pearsonr(df["response"], df["predictions"])
    r_spearman, _ = spearmanr(df["response"], df["predictions"])
    rmse = np.sqrt(np.mean((df["response"] - df["predictions"]) ** 2))

    print(f"{model_name}:")
    print(f"  Pearson r  = {r_pearson:.4f}")
    print(f"  Spearman r = {r_spearman:.4f}")
    print(f"  RMSE       = {rmse:.4f}")
    print(f"  N samples  = {len(df)}")
    print()

# --- Step 7: Save model for download ---
print("Saving trained model artifacts...")
# Note: The model is saved by drevalpy in results/
# You can download the entire results/ folder from Kaggle

# Create a zip of results for easy download
import shutil
shutil.make_archive("/kaggle/working/pharmaai_results", "zip", "results")
print("Results zipped to: /kaggle/working/pharmaai_results.zip")
print("Download this file from the Output tab in Kaggle.")
