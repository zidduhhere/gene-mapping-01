"""SHAP explainability service."""

import os
import json

import numpy as np
import torch
import shap


class ShapService:
    def __init__(self):
        self.gene_descriptions = {}
        self._load_gene_descriptions()

    def _load_gene_descriptions(self):
        desc_file = os.path.join(os.path.dirname(__file__), "..", "data", "gene_descriptions.json")
        if os.path.exists(desc_file):
            with open(desc_file) as f:
                self.gene_descriptions = json.load(f)

    def explain(
        self,
        gene_expression: np.ndarray,
        fingerprint: np.ndarray,
        network,
        gene_names: list[str],
        n_background: int = 50,
        top_n: int = 15,
    ) -> dict:
        """Compute SHAP values for the prediction."""
        n_samples = gene_expression.shape[0]
        fp_tiled = np.tile(fingerprint, (n_samples, 1))
        X = np.concatenate([gene_expression, fp_tiled], axis=1).astype(np.float32)

        # Use first sample for explanation, rest as background
        if n_samples > 1:
            background = X[:min(n_background, n_samples - 1)]
            explain_sample = X[:1]
        else:
            # Single sample — use it as both background and explanation
            background = X
            explain_sample = X

        def model_predict(x):
            network.eval()
            with torch.no_grad():
                preds = network.forward(torch.from_numpy(x.astype(np.float32)).float())
            return preds.cpu().numpy()

        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(explain_sample, nsamples=100)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_row = shap_values[0] if shap_values.ndim > 1 else shap_values

        # Build feature names (genes + fingerprint bits)
        n_genes = len(gene_names)
        feature_names = list(gene_names) + [f"FP_bit_{i}" for i in range(X.shape[1] - n_genes)]

        # Get top features by absolute SHAP value
        abs_shap = np.abs(shap_row)
        top_indices = np.argsort(abs_shap)[::-1][:top_n]

        top_genes = []
        for rank, idx in enumerate(top_indices, 1):
            name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            value = float(shap_row[idx])
            is_gene = idx < n_genes

            description = ""
            if is_gene:
                description = self.gene_descriptions.get(
                    name, self.gene_descriptions.get(name.upper(), "")
                )

            direction = "toward_sensitive" if value < 0 else "toward_resistant"

            top_genes.append({
                "rank": int(rank),
                "feature": str(name),
                "shap_value": round(float(value), 6),
                "abs_shap_value": round(abs(float(value)), 6),
                "direction": direction,
                "is_gene": bool(is_gene),
                "description": str(description),
            })

        return {
            "top_genes": top_genes,
            "total_features": int(len(feature_names)),
            "n_gene_features": int(n_genes),
        }


shap_service = ShapService()
