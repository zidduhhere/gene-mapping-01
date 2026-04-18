"""SHAP explainability for the trained TabTransformer model.

Generates feature importance plots showing which genes and drug features
contribute most to predictions.

Usage:
    python explain.py --model-dir results/PharmaAI_Transformer_2025/TabTransformer
    python explain.py --model-dir results/PharmaAI_TOY_test/TabTransformer
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.TabTransformer.tab_transformer import TabTransformer
from drevalpy.datasets.loader import load_dataset
from drevalpy.models.utils import load_and_select_gene_features, load_drug_fingerprint_features


def get_feature_names(cell_line_input, drug_input):
    """Extract feature names from FeatureDataset objects."""
    feature_names = []

    # Gene expression feature names
    if cell_line_input.meta_info and "gene_expression" in cell_line_input.meta_info:
        feature_names.extend(cell_line_input.meta_info["gene_expression"])
    else:
        sample_id = list(cell_line_input.features.keys())[0]
        n_gex = cell_line_input.features[sample_id]["gene_expression"].shape[0]
        feature_names.extend([f"Gene_{i}" for i in range(n_gex)])

    # Drug fingerprint feature names
    sample_drug = list(drug_input.features.keys())[0]
    n_fp = drug_input.features[sample_drug]["fingerprints"].shape[0]
    feature_names.extend([f"FP_bit_{i}" for i in range(n_fp)])

    return feature_names


def build_feature_matrix(dataset, cell_line_input, drug_input, max_samples=200):
    """Build concatenated feature matrix from dataset."""
    n_samples = min(len(dataset.response), max_samples)
    features_list = []

    for i in range(n_samples):
        cl_id = dataset.cell_line_ids[i]
        drug_id = dataset.drug_ids[i]

        if cl_id not in cell_line_input.features or drug_id not in drug_input.features:
            continue

        gex = cell_line_input.features[cl_id]["gene_expression"]
        fp = drug_input.features[drug_id]["fingerprints"]
        features_list.append(np.concatenate([gex, fp]).astype(np.float32))

    return np.stack(features_list)


def main():
    parser = argparse.ArgumentParser(description="SHAP Explainability for TabTransformer")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with saved model files")
    parser.add_argument("--dataset", type=str, default="TOYv2", help="Dataset for background samples")
    parser.add_argument("--path-data", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="shap_results", help="Output directory for plots")
    parser.add_argument("--n-background", type=int, default=100, help="Number of background samples for SHAP")
    parser.add_argument("--n-explain", type=int, default=50, help="Number of samples to explain")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load trained model
    print(f"Loading model from: {args.model_dir}")
    model = TabTransformer.load(args.model_dir)

    # Load dataset and features
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(dataset_name=args.dataset, path_data=args.path_data)

    cell_line_input = load_and_select_gene_features(
        feature_type="gene_expression",
        gene_list="landmark_genes_reduced",
        data_path=args.path_data,
        dataset_name=args.dataset,
    )
    drug_input = load_drug_fingerprint_features(args.path_data, args.dataset, fill_na=True)

    feature_names = get_feature_names(cell_line_input, drug_input)

    # Build feature matrices
    print("Building feature matrices...")
    X = build_feature_matrix(dataset, cell_line_input, drug_input, max_samples=args.n_background + args.n_explain)

    background = X[: args.n_background]
    explain_samples = X[args.n_background : args.n_background + args.n_explain]

    if len(explain_samples) == 0:
        explain_samples = X[:args.n_explain]

    # Wrap model for SHAP (needs a callable that takes numpy, returns numpy)
    def model_predict(x):
        model.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(x).float()
            preds = model.model.forward(tensor_x)
        return preds.cpu().numpy()

    # Use KernelExplainer (works with any model, more robust than DeepExplainer)
    print(f"Computing SHAP values ({args.n_background} background, {len(explain_samples)} explain samples)...")
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(explain_samples, nsamples=100)

    # Truncate feature names to match actual feature count
    if len(feature_names) > X.shape[1]:
        feature_names = feature_names[:X.shape[1]]
    elif len(feature_names) < X.shape[1]:
        feature_names.extend([f"Feature_{i}" for i in range(len(feature_names), X.shape[1])])

    # 1. Summary bar plot (top 20 features)
    print("Generating summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, explain_samples, feature_names=feature_names, plot_type="bar",
                      max_display=20, show=False)
    plt.title("Top 20 Features by SHAP Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "shap_summary_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Beeswarm plot (top 20 features)
    print("Generating beeswarm plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, explain_samples, feature_names=feature_names,
                      max_display=20, show=False)
    plt.title("SHAP Feature Impact on Drug Response", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Extract and save top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]

    print("\nTop 20 most important features:")
    print("-" * 50)
    top_features = []
    for rank, idx in enumerate(top_indices, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        importance = mean_abs_shap[idx]
        print(f"  {rank:2d}. {name:<30s} SHAP={importance:.4f}")
        top_features.append({"rank": rank, "feature": name, "mean_abs_shap": float(importance)})

    # Save top features to file
    import json
    with open(os.path.join(args.output_dir, "top_features.json"), "w") as f:
        json.dump(top_features, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - shap_summary_bar.png")
    print(f"  - shap_beeswarm.png")
    print(f"  - top_features.json")


if __name__ == "__main__":
    main()
