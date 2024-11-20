"""Mixture of experts model implementation"""

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as L

from kindel.models.torch import FingerprintDataModule
from kindel.models.torch import TorchModel
from kindel.models.basic import Dataset, Example
from kindel.utils.data import featurize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MixtureOfExpertsWithKindel(TorchModel):
    """Mixture of experts model class"""

    def _create_model(self, **hyperparameters):
        """
        Creates and returns the MixtureOfExpertsWithKindelModule.
        """
        self.hyperparams = hyperparameters
        return MixtureOfExpertsWithKindelModule(**hyperparameters).to(device)

    def prepare_dataset(self, df_train, df_valid, df_test):
        self.data_module = FingerprintDataModule(df_train, df_valid, df_test)
        X_train, y_train = self.featurize(df_train)
        X_valid, y_valid = self.featurize(df_valid)
        X_test, y_test = self.featurize(df_test)
        self.data = Dataset(
            train=Example(x=X_train, y=y_train),
            valid=Example(x=X_valid, y=y_valid),
            test=Example(x=X_test, y=y_test),
        )
        return self.data

    def predict(self, x):
        """
        Performs inference on inputs using the trained model.

        Args:
            inputs: Input features for the model.

        Returns:
            Predictions from the model.
        """
        self.model.eval()

        dataset = TensorDataset(
            torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.model.device)

                # generate Y_preds
                expert_1_output = self.model.expert_1.predict(batch)
                expert_1_output = (
                    torch.from_numpy(expert_1_output).float()
                    if isinstance(expert_1_output, np.ndarray)
                    else expert_1_output
                )

                expert_2_output = self.model.expert_2.predict(batch)
                expert_2_output = (
                    torch.from_numpy(expert_2_output).float()
                    if isinstance(expert_2_output, np.ndarray)
                    else expert_2_output
                )

                # Stack outputs for weighted summation
                expert_outputs = torch.stack(
                    [expert_1_output, expert_2_output],
                    dim=1,
                ).to(
                    device=device
                )  # Shape: [batch_size, 2, output_size]

                gate_weights = self.model.gate(batch)  # Shape: [batch_size, 2]

                # Weighted sum of expert outputs
                y_pred = torch.sum(
                    gate_weights.to(device=device) * expert_outputs.to(device=device),
                    dim=1,
                )  # Shape: [batch_size, output_size]

                preds.append(y_pred.flatten())
            preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        return preds

    def featurize(self, df):
        return featurize(df, smiles_col="smiles", label_col="y")


class MixtureOfExpertsWithKindelModule(L.LightningModule):
    """Mixture of Experts model module, following the same implementation as the remaining models in the Kindel repository"""

    def __init__(self, expert_1, expert_2, gating_input_size: int = 2048):
        """
        Mixture of Experts Module with two DeepNeuralNetwork experts.

        Args:
            dnn_config_1: Configuration for the first DeepNeuralNetwork.
            dnn_config_2: Configuration for the second DeepNeuralNetwork.
            gating_input_size: Input size for the gating network.
        """
        super().__init__()
        self.expert_1 = expert_1
        self.expert_2 = expert_2

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(gating_input_size, 2),  # Outputs scores for 2 experts
            nn.Softmax(dim=-1),  # Normalize scores
        )

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Compute outputs from both experts
        # and ensure the outputs are tensors
        expert_1_output = self.expert_1.predict(x)
        expert_1_output = (
            torch.from_numpy(expert_1_output).float()
            if isinstance(expert_1_output, np.ndarray)
            else expert_1_output
        )

        expert_2_output = self.expert_2.predict(x)
        expert_2_output = (
            torch.from_numpy(expert_2_output).float()
            if isinstance(expert_2_output, np.ndarray)
            else expert_2_output
        )

        # Stack outputs for weighted summation
        expert_outputs = torch.stack(
            [expert_1_output, expert_2_output],
            dim=1,
        ).to(device=device)

        gate_weights = self.gate(x)  # Shape: [batch_size, 2]

        # Weighted sum of expert outputs
        y_pred = torch.sum(
            gate_weights * expert_outputs.to(device=device), dim=1
        )  # Shape: [batch_size, output_size]

        loss = nn.functional.mse_loss(y_pred.flatten(), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # prediction

        # Compute outputs from both experts
        # and ensure the outputs are tensors
        expert_1_output = self.expert_1.predict(x)
        expert_1_output = (
            torch.from_numpy(expert_1_output).float()
            if isinstance(expert_1_output, np.ndarray)
            else expert_1_output
        )

        expert_2_output = self.expert_2.predict(x)
        expert_2_output = (
            torch.from_numpy(expert_2_output).float()
            if isinstance(expert_2_output, np.ndarray)
            else expert_2_output
        )

        # Stack outputs for weighted summation
        expert_outputs = torch.stack(
            [expert_1_output, expert_2_output],
            dim=1,
        ).to(
            device=device
        )  # Shape: [batch_size, 2, output_size]

        gate_weights = self.gate(x)  # Shape: [batch_size, 2]

        # Weighted sum of expert outputs
        y_pred = torch.sum(
            gate_weights * expert_outputs.to(device=device), dim=1
        )  # Shape: [batch_size, output_size]

        loss = nn.functional.mse_loss(y_pred.flatten(), y)
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def forward(self, x):
        """
        Forward pass for Mixture of Experts.

        Args:
            x: Input tensor for the gating and expert networks.

        Returns:
            Weighted output from the two experts.
        """
        # Compute outputs from both experts
        expert_1_output = (
            torch.from_numpy(self.expert_1.predict(x)).float()
            if isinstance(self.expert_1.predict(x), np.ndarray)
            else self.expert_1.predict(x)
        )

        expert_2_output = (
            torch.from_numpy(self.expert_2.predict(x)).float()
            if isinstance(self.expert_2.predict(x), np.ndarray)
            else self.expert_2.predict(x)
        )

        # Stack outputs for weighted summation
        expert_outputs = torch.stack([expert_1_output, expert_2_output], dim=1)

        # Compute gating weights
        gate_weights = self.gate(x)  # Shape: [batch_size, 2]

        # Weighted sum of expert outputs
        output = torch.sum(
            gate_weights.unsqueeze(-1) * expert_outputs, dim=1
        )  # Shape: [batch_size, output_size]

        return output
