import pytorch_lightning as L
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.nn import GIN
from torch_geometric.nn import global_mean_pool

from kindel.models.basic import Dataset, Example
from kindel.models.torch import TorchModel
from kindel.utils.graph_feat import featurize_graph


class GraphIsomorphismNetwork(TorchModel):
    def _create_model(self, **hyperparameters):
        return GraphIsomorphismNetworkModule(**hyperparameters)

    def prepare_dataset(self, df_train, df_valid, df_test):
        self.data_module = GraphDataModule(df_train, df_valid, df_test)
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
        self.model.eval()
        dataloader = GraphDataLoader(x, batch_size=32, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.model.device)
                preds.append(self.model(batch).flatten())
            preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        return preds

    def featurize(self, df):
        graphs = featurize_graph(df, smiles_col="smiles", label_col="y")
        return graphs, [graph.y for graph in graphs]


class GraphIsomorphismNetworkModule(L.LightningModule):
    def __init__(self, input_size=48, hidden_size=256):
        super().__init__()
        self.gin = GIN(input_size, hidden_size, num_layers=5)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def training_step(self, batch, batch_idx):
        h = self.gin(batch.x, batch.edge_index)
        h = global_mean_pool(h, batch.batch)
        y_pred = self.projection(h)
        loss = nn.functional.mse_loss(y_pred.flatten(), batch.y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        h = self.gin(batch.x, batch.edge_index)
        h = global_mean_pool(h, batch.batch)
        y_pred = self.projection(h)
        loss = nn.functional.mse_loss(y_pred.flatten(), batch.y)
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def forward(self, batch):
        h = self.gin(batch.x, batch.edge_index)
        h = global_mean_pool(h, batch.batch)
        y_pred = self.projection(h)
        return y_pred


class GraphDataModule(L.LightningDataModule):
    def __init__(self, df_train, df_valid, df_test, batch_size: int = 128):
        super().__init__()
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train_dataset = featurize_graph(
            self.df_train, smiles_col="smiles", label_col="y"
        )
        self.valid_dataset = featurize_graph(
            self.df_valid, smiles_col="smiles", label_col="y"
        )
        self.test_dataset = featurize_graph(
            self.df_test, smiles_col="smiles", label_col="y"
        )

    def train_dataloader(self):
        return GraphDataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return GraphDataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return GraphDataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def predict_dataloader(self):
        return GraphDataLoader(self.mnist_predict, batch_size=self.batch_size)
