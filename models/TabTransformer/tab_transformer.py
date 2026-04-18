"""TabTransformer: Transformer-based drug response prediction model.

Implements the DRPModel interface from drevalpy with a Transformer encoder
architecture for predicting LN_IC50 drug response values from concatenated
gene expression and drug fingerprint features.
"""

import json
import os
import platform
import warnings

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_and_select_gene_features, load_drug_fingerprint_features, scale_gene_expression

from .utils import TransformerDRPNetwork


class TabTransformer(DRPModel):
    """Transformer-based model for drug response prediction.

    Uses gene expression (landmark genes) and drug fingerprints as input.
    Features are chunked into tokens and processed by a Transformer encoder
    with a [CLS] token for regression output.
    """

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self):
        super().__init__()
        self.model = None
        self.hyperparameters = None
        self.gene_expression_scaler = StandardScaler()

    @classmethod
    def get_model_name(cls) -> str:
        return "TabTransformer"

    def build_model(self, hyperparameters: dict) -> None:
        self.hyperparameters = hyperparameters
        self.hyperparameters.setdefault("input_dim_gex", None)
        self.hyperparameters.setdefault("input_dim_fp", None)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for TabTransformer.")

        # Scale gene expression
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

        input_dim = dim_gex + dim_fingerprint
        self.model = TransformerDRPNetwork(
            hyperparameters=self.hyperparameters,
            input_dim=input_dim,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have many workers.*")
            warnings.filterwarnings("ignore", message="Starting from v1\\.9\\.0.*")

            if output_earlystopping is not None and len(output_earlystopping) == 0:
                output_earlystopping = output
                print("TabTransformer: Early stopping dataset empty. Using training data.")

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
                num_workers=1 if platform.system() == "Windows" else 8,
                model_checkpoint_dir=model_checkpoint_dir,
            )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
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

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes_reduced",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))
        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)
        joblib.dump(self.gene_expression_scaler, os.path.join(directory, "scaler.pkl"))

    @classmethod
    def load(cls, directory: str) -> "TabTransformer":
        hyperparam_file = os.path.join(directory, "hyperparameters.json")
        scaler_file = os.path.join(directory, "scaler.pkl")
        model_file = os.path.join(directory, "model.pt")

        if not all(os.path.exists(f) for f in [hyperparam_file, scaler_file, model_file]):
            raise FileNotFoundError("Missing model files. Required: model.pt, hyperparameters.json, scaler.pkl")

        instance = cls()
        with open(hyperparam_file) as f:
            instance.hyperparameters = json.load(f)

        instance.gene_expression_scaler = joblib.load(scaler_file)

        dim_gex = instance.hyperparameters["input_dim_gex"]
        dim_fp = instance.hyperparameters["input_dim_fp"]

        instance.model = TransformerDRPNetwork(instance.hyperparameters, input_dim=dim_gex + dim_fp)
        instance.model.load_state_dict(torch.load(model_file, weights_only=True))
        instance.model.eval()
        return instance
