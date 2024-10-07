import os

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

from kindel.utils.fingerprint_feat import CircularFingerprint

DATA_ROOT = "s3://kin-del-2024/data"


def download_kindel(target):
    return pd.read_parquet(os.path.join(DATA_ROOT, f"{target}.parquet"))


def featurize(df, smiles_col, label_col=None):
    fingerprinter = CircularFingerprint()
    fps = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        fp = fingerprinter._featurize(mol)
        fps.append(fp)
    if label_col is not None:
        return np.array(fps), np.array(df[label_col])
    else:
        return np.array(fps)


def get_training_data(target, split_index, split_type):
    df = pd.read_parquet(os.path.join(DATA_ROOT, f"{target}_1M.parquet")).rename(
        {"target_enrichment": "y"}, axis="columns"
    )
    df_split = pd.read_parquet(
        os.path.join(DATA_ROOT, "splits", f"{target}_{split_type}.parquet")
    )
    return (
        df[df_split[f"split{split_index}"] == "train"],
        df[df_split[f"split{split_index}"] == "valid"],
        df[df_split[f"split{split_index}"] == "test"],
    )


def get_testing_data(target, in_library=False):
    data = {
        "on": pd.read_csv(
            os.path.join(DATA_ROOT, "heldout", f"{target}_ondna.csv"), index_col=0
        ).rename({"kd": "y"}, axis="columns"),
        "off": pd.read_csv(
            os.path.join(DATA_ROOT, "heldout", f"{target}_offdna.csv"), index_col=0
        ).rename({"kd": "y"}, axis="columns"),
    }
    if in_library:
        data["on"] = data["on"].dropna(subset="molecule_hash")
        data["off"] = data["off"].dropna(subset="molecule_hash")
    return data


def evaluate(preds, target, condition):
    df = pd.read_csv(
        os.path.join(DATA_ROOT, "heldout", f"{target}_{condition}dna.csv"), index_col=0
    )
    return spearmanr(preds, df.kd)[0], kendalltau(preds, df.kd)[0]


def spearman(preds, target):
    return spearmanr(preds, target)[0]


def kendall(preds, target):
    return kendalltau(preds, target)[0]


def rmse(preds, target):
    return np.sqrt(np.mean((preds - target) ** 2))
