"""Transformer network and dataset utilities for TabTransformer."""

import os
import secrets

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, Dataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


class RegressionDataset(Dataset):
    """Dataset that concatenates cell line + drug features for regression."""

    def __init__(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
        cell_line_views: list[str],
        drug_views: list[str],
    ):
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
    """Transformer-based network for drug response prediction.

    Architecture: Linear embedding -> Transformer Encoder -> Regression head.
    Uses multi-head self-attention over chunked feature tokens.
    """

    def __init__(self, hyperparameters: dict, input_dim: int) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hyperparameters.get("hidden_dim", 128)
        self.num_layers = hyperparameters.get("num_layers", 4)
        self.num_heads = hyperparameters.get("num_heads", 8)
        self.dropout_prob = hyperparameters.get("dropout_prob", 0.1)
        self.token_size = hyperparameters.get("token_size", 64)
        self.input_dim = input_dim

        # Number of tokens = ceil(input_dim / token_size)
        self.n_tokens = (input_dim + self.token_size - 1) // self.token_size
        # Pad input to be divisible by token_size
        self.padded_dim = self.n_tokens * self.token_size

        # Token embedding: project each chunk to hidden_dim
        self.token_embed = nn.Linear(self.token_size, self.hidden_dim)

        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens + 1, self.hidden_dim) * 0.02)

        # Transformer encoder
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

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self.loss = nn.MSELoss()
        self.checkpoint_callback = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Pad input if needed
        if x.shape[1] < self.padded_dim:
            padding = torch.zeros(batch_size, self.padded_dim - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)

        # Reshape into tokens: (batch, n_tokens, token_size)
        x = x.view(batch_size, self.n_tokens, self.token_size)

        # Embed tokens: (batch, n_tokens, hidden_dim)
        x = self.token_embed(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_tokens+1, hidden_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer encoder
        x = self.transformer(x)
        x = self.layer_norm(x)

        # Use CLS token output for regression
        cls_output = x[:, 0, :]  # (batch, hidden_dim)
        return self.reg_head(cls_output).squeeze(-1)

    def _forward_loss_and_log(self, x, y, log_as: str):
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

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """Predict from numpy array, returns numpy array."""
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
        output_train: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
        cell_line_views: list[str],
        drug_views: list[str],
        output_earlystopping: DrugResponseDataset | None = None,
        trainer_params: dict | None = None,
        batch_size: int = 32,
        patience: int = 5,
        num_workers: int = 2,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
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
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=True,
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
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True,
            )

        monitor = "train_loss" if val_loader is None else "val_loss"
        early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=patience)

        unique_subfolder = os.path.join(model_checkpoint_dir, "run_" + secrets.token_hex(8))
        os.makedirs(unique_subfolder, exist_ok=True)

        name = "version-" + "".join([secrets.choice("0123456789abcdef") for _ in range(10)])
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=unique_subfolder,
            monitor=monitor,
            mode="min",
            save_top_k=1,
            filename=name,
        )

        progress_bar = TQDMProgressBar(refresh_rate=trainer_params.get("progress_bar_refresh_rate", 500))
        trainer_params_copy = {k: v for k, v in trainer_params.items() if k != "progress_bar_refresh_rate"}

        trainer = pl.Trainer(
            callbacks=[early_stop_callback, self.checkpoint_callback, progress_bar],
            default_root_dir=model_checkpoint_dir,
            devices=1,
            **trainer_params_copy,
        )
        if val_loader is None:
            trainer.fit(self, train_loader)
        else:
            trainer.fit(self, train_loader, val_loader)

        # Load best model
        if self.checkpoint_callback.best_model_path:
            checkpoint = torch.load(self.checkpoint_callback.best_model_path, weights_only=True)
            self.load_state_dict(checkpoint["state_dict"])
        else:
            print("TabTransformer: No best model found, using last model.")
