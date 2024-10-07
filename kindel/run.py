import os
from typing import List, Optional

import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger
from redun import Dir, File, task

from kindel.models.basic import RandomForest, KNeareastNeighbors, XGBoost
from kindel.models.gnn import GraphIsomorphismNetwork
from kindel.models.torch import DeepNeuralNetwork
from kindel.utils.data import (
    get_training_data,
    get_testing_data,
    kendall,
    spearman,
    rmse,
)
from kindel.utils.helpers import set_seed
from kindel.models.compose import DELCompose

redun_namespace = "kindel"

api = wandb.Api()
BATCH_ENV_VAR_DICT = {
    "containerProperties": {
        "environment": [
            {"name": "WANDB_BASE_URL", "value": api.client.app_url},
            {"name": "WANDB_API_KEY", "value": api.api_key},
        ]
    }
}


def get_model(model_name: str, hyperparameters: dict, wandb_logger: WandbLogger):
    if model_name.lower().startswith("xgboost"):
        model = XGBoost(**hyperparameters)
    elif model_name.lower().startswith("rf"):
        model = RandomForest(**hyperparameters)
    elif model_name.lower().startswith("knn"):
        model = KNeareastNeighbors(**hyperparameters)
    elif model_name.lower().startswith("dnn"):
        model = DeepNeuralNetwork(wandb_logger, **hyperparameters)
    elif model_name.lower().startswith("gin"):
        model = GraphIsomorphismNetwork(wandb_logger, **hyperparameters)
    elif model_name.lower().startswith("compose"):
        model = DELCompose(wandb_logger, **hyperparameters)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


@task(executor="main", vcpus=5, job_def_extra=BATCH_ENV_VAR_DICT)
def training_subjob(
    model_name: str,
    output_dir: Dir,
    split_index: int,
    split_type: str,
    target: str,
    wandb_project: str | None = None,
    hyperparameters: dict | None = None,
) -> File:
    set_seed(123)
    df_train, df_valid, df_test = get_training_data(
        target, split_index=split_index, split_type=split_type
    )

    if hyperparameters is None:
        hyperparameters = {}

    if wandb_project:
        wandb_logger = WandbLogger(project=wandb_project)
        wandb_logger.experiment.config.update(
            {
                "model": model_name,
                "target": target,
                "split_index": split_index,
                "split_type": split_type,
            }
        )
    else:
        wandb_logger = None

    model = get_model(model_name, hyperparameters, wandb_logger)
    data = model.prepare_dataset(df_train, df_valid, df_test)
    model.train()

    results = {}

    print("computing internal test set performance")
    preds = model.predict(data.test.x)
    rho, tau = spearman(preds, data.test.y), kendall(preds, data.test.y)
    results["test"] = {"rho": rho, "tau": tau, "rmse": rmse(preds, data.test.y)}

    print("computing performance on the extended held-out set")
    testing_data = get_testing_data(target)
    results["all"] = {}
    for condition in ("on", "off"):
        X_test, y_test = model.featurize(testing_data[condition])
        preds = model.predict(X_test)
        rho, tau = spearman(preds, y_test), kendall(preds, y_test)
        results["all"][condition] = {"rho": rho, "tau": tau}

    print("computing performance on the in-library held-out set")
    testing_data = get_testing_data(target, in_library=True)
    results["lib"] = {}
    for condition in ("on", "off"):
        X_test, y_test = model.featurize(testing_data[condition])
        preds = model.predict(X_test)
        rho, tau = spearman(preds, y_test), kendall(preds, y_test)
        results["lib"][condition] = {"rho": rho, "tau": tau}

    print("saving results")
    results_file = File(
        os.path.join(
            output_dir.path,
            model_name,
            f"results_{split_type}",
            f"results_metrics_s{split_index}_{target}.yml",
        )
    )
    with results_file.open("w") as fp:
        yaml.dump(results, fp)

    print("exit")
    return results_file


@task(job_def_extra=BATCH_ENV_VAR_DICT)
def train(
    model: str,
    output_dir: Dir,
    wandb_project: str | None = None,
    targets: List[str] = ["ddr1", "mapk14"],
    splits: List[str] = ["random", "disynthon"],
    split_indexes: List[int] = [1, 2, 3, 4, 5],
    hyperparameters: Optional[File] = None,
) -> List[File]:
    if hyperparameters is not None:
        with hyperparameters.open("r") as fp:
            hyperparameters = yaml.safe_load(fp)

    train_partial = training_subjob.partial(
        model_name=model,
        output_dir=output_dir,
        wandb_project=wandb_project,
        hyperparameters=hyperparameters,
    )

    return [
        train_partial(split_index=split_index, target=target, split_type=split_type)
        for split_index in split_indexes
        for target in targets
        for split_type in splits
    ]
