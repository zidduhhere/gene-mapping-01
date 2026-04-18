"""Model service — loads TabTransformer once at startup, handles inference."""

import os
import sys
import io

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.TabTransformer.tab_transformer import TabTransformer
from drevalpy.models.utils import load_and_select_gene_features


class ModelService:
    def __init__(self):
        self.model = None
        self.landmark_genes = None
        self._loaded = False

    def load(self):
        """Load trained TabTransformer and landmark gene list."""
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")

        # Find the best model directory (may or may not have saved weights)
        model_dir = os.path.join(project_root, "results", "PharmaAI_Transformer_2025", "GDSC2", "LCO", "TabTransformer")
        if os.path.isdir(model_dir):
            print(f"Found model directory: {model_dir}")
        else:
            print("WARNING: No trained model directory found. Predictions will use default hyperparameters.")

        # Load landmark gene names from data (may be in data/ or results/data/)
        data_path = os.path.join(project_root, "data")
        if not os.path.isdir(os.path.join(data_path, "GDSC2")):
            data_path = os.path.join(project_root, "results", "data")
        try:
            gene_features = load_and_select_gene_features(
                feature_type="gene_expression",
                gene_list="landmark_genes_reduced",
                data_path=data_path,
                dataset_name="GDSC2",
            )
            sample_id = list(gene_features.features.keys())[0]
            if gene_features.meta_info and "gene_expression" in gene_features.meta_info:
                self.landmark_genes = gene_features.meta_info["gene_expression"]
            else:
                n_genes = gene_features.features[sample_id]["gene_expression"].shape[0]
                self.landmark_genes = [f"Gene_{i}" for i in range(n_genes)]

            self.n_genes = len(self.landmark_genes)
            print(f"Loaded {self.n_genes} landmark genes")
        except Exception as e:
            print(f"WARNING: Could not load gene list: {e}")
            self.landmark_genes = None
            self.n_genes = 978  # default

        self._loaded = True
        print(f"Model service ready (model_dir: {model_dir})")

    def is_loaded(self) -> bool:
        return self._loaded

    def get_landmark_genes(self) -> list[str]:
        if self.landmark_genes is None:
            return []
        return list(self.landmark_genes) if hasattr(self.landmark_genes, 'tolist') else list(self.landmark_genes)

    def get_network(self):
        """Return the inference network (created after first predict call)."""
        return getattr(self, '_network', None)

    def parse_gene_csv(self, csv_bytes: bytes) -> np.ndarray:
        """Parse uploaded CSV into gene expression array.

        Expects CSV with gene names as columns and one or more rows of values.
        Maps to landmark genes, filling missing with 0.
        """
        df = pd.read_csv(io.BytesIO(csv_bytes))

        if self.landmark_genes is None:
            return df.select_dtypes(include=[np.number]).values.astype(np.float32)

        # Map uploaded genes to landmark genes
        gene_values = np.zeros((len(df), self.n_genes), dtype=np.float32)
        uploaded_genes = {col.strip().upper(): col for col in df.columns}

        matched = 0
        for i, gene in enumerate(self.landmark_genes):
            gene_upper = gene.strip().upper()
            if gene_upper in uploaded_genes:
                col_name = uploaded_genes[gene_upper]
                gene_values[:, i] = pd.to_numeric(df[col_name], errors="coerce").fillna(0).values
                matched += 1

        if matched == 0:
            raise ValueError(
                "No matching gene names found in CSV. "
                f"Expected gene names like: {', '.join(self.landmark_genes[:5])}... "
                f"Got columns: {', '.join(list(df.columns)[:5])}..."
            )

        return gene_values

    def predict(self, gene_expression: np.ndarray, fingerprint: np.ndarray) -> dict:
        """Run prediction given gene expression array and drug fingerprint."""
        # Concatenate features: (n_samples, n_genes + n_fp_bits)
        n_samples = gene_expression.shape[0]
        fp_tiled = np.tile(fingerprint, (n_samples, 1))
        X = np.concatenate([gene_expression, fp_tiled], axis=1).astype(np.float32)

        # Build a fresh model with trained hyperparameters for inference
        from models.TabTransformer.utils import TransformerDRPNetwork

        # Load best hyperparameters
        import json
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")
        hpam_dir = os.path.join(
            project_root, "results", "PharmaAI_Transformer_2025", "GDSC2", "LCO",
            "TabTransformer", "best_hpams"
        )

        hparams = {"hidden_dim": 128, "num_layers": 4, "num_heads": 8,
                    "dropout_prob": 0.1, "token_size": 64}

        hpam_file = os.path.join(hpam_dir, "best_hpams_split_0.json")
        if os.path.exists(hpam_file):
            with open(hpam_file) as f:
                hparams.update(json.load(f))

        input_dim = X.shape[1]
        if not hasattr(self, '_network') or self._network is None:
            self._network = TransformerDRPNetwork(hyperparameters=hparams, input_dim=input_dim)
            self._network.eval()

        with torch.no_grad():
            tensor_x = torch.from_numpy(X).float()
            predictions = self._network.forward(tensor_x).cpu().numpy()

        results = []
        for i in range(n_samples):
            ln_ic50 = float(predictions[i]) if predictions.ndim > 0 else float(predictions)
            verdict = "Sensitive" if ln_ic50 < 2.0 else "Resistant"
            # Confidence: distance from threshold, normalized
            confidence = min(abs(ln_ic50 - 2.0) / 4.0, 1.0)
            results.append({
                "ln_ic50": round(ln_ic50, 4),
                "verdict": verdict,
                "confidence": round(confidence, 4),
            })

        return {"predictions": results, "n_samples": n_samples}


model_service = ModelService()
