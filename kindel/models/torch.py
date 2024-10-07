import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from kindel.models.basic import Dataset, Example
from kindel.models.basic import Model
from kindel.utils.data import featurize


class FingerprintDataModule(L.LightningDataModule):
    def __init__(
        self,
        df_train,
        df_valid,
        df_test,
        batch_size: int = 32,
        featurizer_fn=featurize,
        dataloader_cls=DataLoader,
    ):
        super().__init__()
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.batch_size = batch_size
        self.featurizer_fn = featurizer_fn
        self.dataloader_cls = dataloader_cls

    def setup(self, stage: str):
        X_train, y_train = self.featurizer_fn(
            self.df_train, smiles_col="smiles", label_col="y"
        )
        self.train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        )

        X_valid, y_valid = self.featurizer_fn(
            self.df_valid, smiles_col="smiles", label_col="y"
        )
        self.valid_dataset = TensorDataset(
            torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).float()
        )

        X_test, y_test = self.featurizer_fn(
            self.df_test, smiles_col="smiles", label_col="y"
        )
        self.test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        )

    def train_dataloader(self):
        return self.dataloader_cls(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return self.dataloader_cls(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return self.dataloader_cls(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )


class TorchModel(Model):
    def __init__(self, wandb_logger=None, **hyperparameters):
        self.wandb_logger = wandb_logger
        super().__init__(**hyperparameters)

    def train(self):
        checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="valid_loss")
        early_stopping_callback = EarlyStopping(
            monitor="valid_loss", min_delta=1e-3, patience=5, verbose=True, mode="min"
        )
        trainer = L.Trainer(
            max_epochs=30,
            logger=self.wandb_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )
        trainer.fit(model=self.model, datamodule=self.data_module)
        self.model = self.model.__class__.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )


class DeepNeuralNetwork(TorchModel):
    def _create_model(self, **hyperparameters):
        return DeepNeuralNetworkModule(**hyperparameters)

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
        self.model.eval()

        dataset = TensorDataset(torch.from_numpy(x).float())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.model.device)
                preds.append(self.model.network(batch).flatten())
            preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        return preds

    def featurize(self, df):
        return featurize(df, smiles_col="smiles", label_col="y")


class DeepNeuralNetworkModule(L.LightningModule):
    def __init__(self, input_size=2048, hidden_size=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.network(x)
        loss = nn.functional.mse_loss(y_pred.flatten(), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.network(x)
        loss = nn.functional.mse_loss(y_pred.flatten(), y)
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def forward(self, batch):
        return self.network(batch)
