from functools import partial
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from kindel.models.compose.utils.fingerprints import get_fps


def parse_data(
    batch_df: pd.DataFrame,
    embed_type: str,
    pre_col: str = "load",
    fp_dim: int = 2048,
    model_type: Optional[Literal["factorized", "full"]] = None,
):
    """
    Helper function to parse each data batch
    """
    batch_size = batch_df.shape[0]

    control_counts = []
    control_idx = 1
    while True:
        cur_col_name = "control_%d" % control_idx
        if cur_col_name not in batch_df.columns:
            break
        cur_col = (batch_df[cur_col_name].fillna(0.0).to_numpy()).astype(int)
        control_counts.append(cur_col)
        control_idx += 1
    control_counts = torch.tensor(np.stack(control_counts, axis=1))

    target_counts = []
    target_idx = 1
    while True:
        cur_col_name = "target_%d" % target_idx
        if cur_col_name not in batch_df.columns:
            break
        cur_col = (batch_df[cur_col_name].fillna(0.0).to_numpy()).astype(int)
        target_counts.append(cur_col)
        target_idx += 1
    target_counts = torch.tensor(np.stack(target_counts, axis=1))

    smiles = list(batch_df["smiles"])
    if pre_col != "none":
        pre_tensor = torch.tensor(np.array(batch_df[pre_col].fillna(0.0)))
    else:
        pre_tensor = None
    if pre_col == "pre":
        pre_tensor += 1
    data_dict = {
        "control_counts": control_counts,
        "target_counts": target_counts,
        "pre": pre_tensor,
        "smiles": smiles,
        "batch_size": batch_size,
    }

    # Add individual synthon smiles--if available
    for col_name in ["smiles_a", "smiles_b", "smiles_c"]:
        if col_name in batch_df.columns:
            data_dict.update({col_name: list(batch_df[col_name])})

    if embed_type == "onehot":
        for id_name in ["a_idx", "b_idx", "c_idx"]:
            id_input = torch.tensor(list(batch_df[id_name])).to(dtype=torch.int64)
            data_dict.update({id_name: id_input})
    elif embed_type == "fps":
        for col_name in ["fps", "fps_a", "fps_b", "fps_c"]:
            if col_name in batch_df.columns:
                fps_list = np.array(list(batch_df[col_name]))
                fps_tensor = torch.tensor(fps_list).to(dtype=torch.float)
                data_dict.update({col_name: fps_tensor})

            if "fps" not in batch_df.columns:
                smiles_list = list(batch_df["smiles"])
                fps_mat = np.array(get_fps(smiles_list, fp_dim=fp_dim, as_numpy=True))
                fps_tensor = torch.tensor(fps_mat).to(dtype=torch.float)
                data_dict.update({"fps": fps_tensor})

            if model_type == "factorized":
                if "fps_a" not in batch_df.columns:
                    smiles_list = list(batch_df["smiles_a"])
                    fps_mat = np.array(
                        get_fps(smiles_list, fp_dim=fp_dim, as_numpy=True)
                    )
                    fps_tensor = torch.tensor(fps_mat).to(dtype=torch.float)
                    data_dict.update({"fps_a": fps_tensor})

                if "fps_b" not in batch_df.columns:
                    smiles_list = list(batch_df["smiles_b"])
                    fps_mat = np.array(
                        get_fps(smiles_list, fp_dim=fp_dim, as_numpy=True)
                    )
                    fps_tensor = torch.tensor(fps_mat).to(dtype=torch.float)
                    data_dict.update({"fps_b": fps_tensor})

                if "fps_c" not in batch_df.columns:
                    smiles_list = list(batch_df["smiles_c"])
                    fps_mat = np.array(
                        get_fps(smiles_list, fp_dim=fp_dim, as_numpy=True)
                    )
                    fps_tensor = torch.tensor(fps_mat).to(dtype=torch.float)
                    data_dict.update({"fps_c": fps_tensor})
    else:
        raise ValueError(f"Unrecognized embedding type: {embed_type}")
    return data_dict


class DELDataModule(pl.LightningDataModule):
    """
    Data module responsible for parsing input data and creating necessary train/val/test datasets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size=64,
        shuffle=True,
        n_control=3,
        n_target=3,
        fps_dim=2048,
        pre_col="seq_load",
        control_col="seq_matrix",
        target_col="seq_target",
        embed_type="fps",
        model_type="full",
    ):
        super().__init__()
        self.df = df.reset_index()
        self.n_control = n_control
        self.n_target = n_target
        self.control_col = control_col
        self.target_col = target_col
        self.pre_col = pre_col
        self.shuffle = shuffle
        self.split_column = "split"
        self.embed_type = embed_type
        self.model_type = model_type
        self.fps_dim = fps_dim
        self.batch_size = batch_size
        self.num_workers = 2

    def collate_fn(self, data_list):
        return parse_data(
            pd.DataFrame(data_list),
            self.embed_type,
            pre_col=self.pre_col,
            fp_dim=self.fps_dim,
            model_type=self.model_type,
        )

    def setup(self, stage: Literal["fit", "validate", "test"]):
        col_name_dict = {}
        for i in range(self.n_control):
            raw_col_name = "%s_%d" % (self.control_col, i + 1)
            col_name_dict[raw_col_name] = "control_%d" % (i + 1)
        for i in range(self.n_target):
            raw_col_name = "%s_%d" % (self.target_col, i + 1)
            col_name_dict[raw_col_name] = "target_%d" % (i + 1)
        self.df.rename(col_name_dict, inplace=True, axis="columns")

        if stage == "fit":
            self.train = DELDataset(self.df[self.df[self.split_column] == "train"])
            self.val = DELDataset(self.df[self.df[self.split_column] == "dev"])
        elif stage == "test":
            self.test = DELDataset(self.df[self.df[self.split_column] == "test"])
        else:
            raise ValueError(f"Unexpected stage {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            collate_fn=partial(self.collate_fn),
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            collate_fn=partial(self.collate_fn),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            collate_fn=partial(self.collate_fn),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class DELDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        return self.df.iloc[index]

    def __len__(self):
        return self.df.shape[0]
