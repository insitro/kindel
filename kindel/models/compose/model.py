import numpy as np
import omegaconf
import pandas as pd
import torch

from kindel.models.basic import Dataset, Example
from kindel.models.compose.dataset import DELDataModule
from kindel.models.compose.models.del_models import DELModel
from kindel.models.compose.utils.fingerprints import get_fps
from kindel.models.torch import TorchModel


class DELCompose(TorchModel):
    def _create_model(self, **hyperparameters):
        cfg = {
            "verbose": False,
            "log_on_step": False,
            "debug_mode": False,
            "lr": 5e-5,
            "patience": 3,
            "data": {
                "batch_size": 64,
                "shuffle": True,
                "n_control": 3,
                "n_target": 3,
                "fps_dim": 2048,
                "pre_col": "seq_load",
                "control_col": "seq_matrix",
                "target_col": "seq_target",
            },
            "model": {
                "model_type": hyperparameters.get("model_type", "full"),
                "synthon_agg_model": hyperparameters.get(
                    "synthon_agg_model", "attention"
                ),
                "embed_type": hyperparameters.get("embed_type", "fps"),
                "output_dist": hyperparameters.get("output_dist", "zip"),
                "n_layers": 4,
                "hidden_dim": 128,
                "dropout": 0.0,
                "agg_fun": "sum",
                "beta": 1.0,
                "share_embeddings": True,
                "use_pre": True,
                "rep_embed": True,
                "rep_func": "product",
                "detach_control": False,
                "mask": None,
            },
        }
        cfg = omegaconf.OmegaConf.create(cfg)
        self.embed_type = hyperparameters.get("embed_type", "fps")
        self.model_type = hyperparameters.get("model_type", "full")
        return DELModel(cfg)

    def prepare_dataset(self, df_train, df_valid, df_test, **data_parameters):
        df_train["split"] = "train"
        df_valid["split"] = "dev"
        df_test["split"] = "test"
        df = pd.concat([df_train, df_valid, df_test], axis=0)
        df["smiles_c"] = df.smiles_c.str.replace(", ", ".")
        self.data_module = DELDataModule(
            df,
            embed_type=self.embed_type,
            model_type=self.model_type,
            **data_parameters,
        )
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

        embedding_dict = self.model.enrichment_model.compute_embeddings(x)
        scores_dict = self.model.enrichment_model.compute_enrichments(
            embedding_dict["z"]
        )
        target_scores = (
            torch.exp(scores_dict["log_target_scores"]).detach().cpu().numpy()
        )

        probs_dict = self.model.enrichment_model.compute_zero_probs(embedding_dict["z"])
        target_scores = target_scores * (
            1 - probs_dict["target_zero_probs"].detach().cpu().numpy()
        )

        return target_scores

    def featurize(self, df):
        smiles_list = list(df["smiles"])
        labels = list(df["y"])

        fps_mat = np.stack(get_fps(smiles_list, as_numpy=True))
        fps_tensor = torch.tensor(fps_mat).to(device="cpu", dtype=torch.float)
        data_dict = {"smiles": smiles_list, "fps": fps_tensor}

        if self.model_type == "factorized":
            for cycle in ("a", "b", "c"):
                smiles_list = list(df[f"smiles_{cycle}"].str.replace(", ", "."))
                fps_mat = np.stack(get_fps(smiles_list, as_numpy=True))
                fps_tensor = torch.tensor(fps_mat).to(device="cpu", dtype=torch.float)
                data_dict.update({f"fps_{cycle}": fps_tensor})
        return data_dict, labels
