"""PharmaAI Predictor - Streamlit Web Application.

AI-driven drug response prediction using a Transformer model trained on
GDSC/CCLE pharmacogenomic data via drevalpy.

Usage:
    streamlit run app.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.TabTransformer.tab_transformer import TabTransformer


# --- Page Config ---
st.set_page_config(
    page_title="PharmaAI Predictor",
    page_icon="logo.png" if os.path.exists("logo.png") else None,
    layout="wide",
)

st.title("PharmaAI Predictor")
st.caption("AI-Driven Drug Response Prediction using Transformer Architecture")

# --- Sidebar ---
st.sidebar.header("Configuration")

model_dir = st.sidebar.text_input(
    "Model directory",
    value="results/PharmaAI_Transformer_2025/TabTransformer",
    help="Path to directory containing model.pt, hyperparameters.json, and scaler.pkl",
)

shap_dir = st.sidebar.text_input(
    "SHAP results directory",
    value="shap_results",
    help="Path to SHAP analysis results",
)


# --- Model Loading ---
@st.cache_resource
def load_model(directory):
    """Load the trained TabTransformer model."""
    try:
        model = TabTransformer.load(directory)
        return model, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error loading model: {e}"


# --- Main Tabs ---
tab_predict, tab_explain, tab_about = st.tabs(["Predict", "Explainability", "About"])


# --- Prediction Tab ---
with tab_predict:
    st.header("Drug Response Prediction")

    model, error = load_model(model_dir)

    if error:
        st.warning(f"Model not loaded: {error}")
        st.info(
            "To use predictions, first train the model:\n"
            "```bash\n"
            "cd PharmaAI-Predictor\n"
            "python train_pharmaai.py --toy  # quick test\n"
            "python train_pharmaai.py --dataset GDSC2  # full training\n"
            "```"
        )
    else:
        st.success("Model loaded successfully!")

        input_mode = st.radio("Input mode", ["Upload CSV", "Manual Entry"], horizontal=True)

        if input_mode == "Upload CSV":
            st.markdown(
                "Upload a CSV with concatenated features: "
                "gene expression (landmark genes) + drug fingerprints. "
                "Each row is one cell-line/drug pair."
            )
            uploaded_file = st.file_uploader("Upload feature CSV", type=["csv"])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {df.shape[0]} samples, {df.shape[1]} features")

                    X = df.values.astype(np.float32)

                    if st.button("Predict", type="primary"):
                        with st.spinner("Running predictions..."):
                            model.model.eval()
                            with torch.no_grad():
                                tensor_x = torch.from_numpy(X).float()
                                predictions = model.model.forward(tensor_x).cpu().numpy()

                        results_df = pd.DataFrame({
                            "Sample": range(1, len(predictions) + 1),
                            "Predicted LN_IC50": predictions.round(4),
                            "Classification": ["Sensitive" if p < 0 else "Resistant" for p in predictions],
                        })

                        st.dataframe(results_df, use_container_width=True)

                        # Distribution plot
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(predictions, bins=30, edgecolor="black", alpha=0.7, color="#4CAF50")
                        ax.axvline(x=0, color="red", linestyle="--", label="Sensitive/Resistant threshold")
                        ax.set_xlabel("Predicted LN_IC50")
                        ax.set_ylabel("Count")
                        ax.set_title("Distribution of Predicted Drug Responses")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()

                        # Download results
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            "Download predictions CSV",
                            csv_data,
                            file_name="pharmaai_predictions.csv",
                            mime="text/csv",
                        )

                except Exception as e:
                    st.error(f"Error processing file: {e}")

        else:  # Manual Entry
            st.markdown("Enter feature values manually (comma-separated).")
            st.markdown(
                "Expected format: gene expression values followed by drug fingerprint bits."
            )

            n_features = (
                model.hyperparameters.get("input_dim_gex", 0)
                + model.hyperparameters.get("input_dim_fp", 0)
            )
            st.info(f"Model expects {n_features} features per sample.")

            feature_text = st.text_area(
                "Feature values (comma-separated)",
                height=100,
                placeholder="0.5, -0.3, 1.2, ... (one row of features)",
            )

            if feature_text and st.button("Predict", type="primary"):
                try:
                    values = [float(x.strip()) for x in feature_text.split(",") if x.strip()]
                    if len(values) != n_features:
                        st.warning(f"Expected {n_features} features, got {len(values)}. Padding/truncating.")
                        if len(values) < n_features:
                            values.extend([0.0] * (n_features - len(values)))
                        else:
                            values = values[:n_features]

                    X = np.array([values], dtype=np.float32)
                    model.model.eval()
                    with torch.no_grad():
                        pred = model.model.forward(torch.from_numpy(X).float()).cpu().numpy()

                    ic50 = pred[0]
                    classification = "Sensitive" if ic50 < 0 else "Resistant"
                    color = "green" if ic50 < 0 else "red"

                    st.markdown(f"### Predicted LN_IC50: `{ic50:.4f}`")
                    st.markdown(f"### Classification: :{color}[{classification}]")

                except ValueError:
                    st.error("Invalid input. Please enter comma-separated numbers.")


# --- Explainability Tab ---
with tab_explain:
    st.header("Model Explainability (SHAP)")

    if os.path.exists(shap_dir):
        col1, col2 = st.columns(2)

        bar_path = os.path.join(shap_dir, "shap_summary_bar.png")
        bee_path = os.path.join(shap_dir, "shap_beeswarm.png")
        features_path = os.path.join(shap_dir, "top_features.json")

        if os.path.exists(bar_path):
            with col1:
                st.image(bar_path, caption="Top 20 Features by SHAP Importance")

        if os.path.exists(bee_path):
            with col2:
                st.image(bee_path, caption="SHAP Beeswarm Plot")

        if os.path.exists(features_path):
            st.subheader("Top Important Features")
            with open(features_path) as f:
                top_features = json.load(f)
            features_df = pd.DataFrame(top_features)
            st.dataframe(features_df, use_container_width=True)
        else:
            st.info("No top_features.json found.")
    else:
        st.info(
            "No SHAP results found. Generate them with:\n"
            "```bash\n"
            "python explain.py --model-dir <model_dir> --dataset TOYv2\n"
            "```"
        )


# --- About Tab ---
with tab_about:
    st.header("About PharmaAI Predictor")
    st.markdown("""
**PharmaAI Predictor** uses a Transformer-based deep learning architecture to predict
drug response (LN_IC50) from cancer cell line gene expression profiles and drug molecular
fingerprints.

### Architecture
- **Input**: Concatenated gene expression (landmark genes) + Morgan fingerprints
- **Tokenization**: Features are chunked into fixed-size tokens
- **Encoder**: Transformer encoder with multi-head self-attention
- **Output**: [CLS] token regression head predicts LN_IC50

### Built with
- [drevalpy](https://github.com/daisybio/drevalpy) - Drug Response Evaluation Framework
- PyTorch + PyTorch Lightning
- SHAP for model interpretability
- Streamlit for the web interface

### Datasets
- **GDSC** (Genomics of Drug Sensitivity in Cancer)
- **CCLE** (Cancer Cell Line Encyclopedia)

### How to Use
1. **Train**: `python train_pharmaai.py --dataset GDSC2`
2. **Explain**: `python explain.py --model-dir results/.../TabTransformer`
3. **Run App**: `streamlit run app.py`
""")
