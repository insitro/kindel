import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from scipy import stats


def log_data_table(test_output_list, n_control, n_target, output_dist) -> dict:
    assert len(test_output_list) > 0

    if n_control <= 0 and n_target <= 0:
        return {}

    data_dict = {}

    smiles_list = sum([x["smiles"] for x in test_output_list], [])
    data_dict["smiles"] = smiles_list

    smiles_b_list = sum([x["smiles_b"] for x in test_output_list], [])
    data_dict["smiles_b"] = smiles_b_list

    smiles_c_list = sum([x["smiles_c"] for x in test_output_list], [])
    data_dict["smiles_c"] = smiles_c_list

    for i in range(n_control):
        count_list = np.concatenate(
            [x["control_%d" % i] for x in test_output_list], axis=0
        )
        data_dict["control_%d" % i] = count_list

    for j in range(n_target):
        count_list = np.concatenate(
            [x["target_%d" % j] for x in test_output_list], axis=0
        )
        data_dict["target_%d" % j] = count_list

    control_p = (
        torch.cat([x["control_p"] for x in test_output_list], dim=0)
        .detach()
        .cpu()
        .numpy()
        .squeeze(1)
    )
    target_p = (
        torch.cat([x["target_p"] for x in test_output_list], dim=0)
        .detach()
        .cpu()
        .numpy()
        .squeeze(1)
    )
    data_dict["control_p"] = control_p
    data_dict["target_p"] = target_p

    log_control_scores = (
        torch.cat([x["log_control_scores"] for x in test_output_list], axis=0)
        .detach()
        .cpu()
        .numpy()
    )
    log_target_scores = (
        torch.cat([x["log_target_scores"] for x in test_output_list], axis=0)
        .detach()
        .cpu()
        .numpy()
    )

    data_dict["log_control_scores"] = log_control_scores.squeeze(axis=1)
    data_dict["log_target_scores"] = log_target_scores.squeeze(axis=1)

    control_dists = [x["control_dists"] for x in test_output_list]
    target_dists = [x["target_dists"] for x in test_output_list]

    if len(control_dists[0]) == 1:
        # no replicates
        if output_dist in ["zip", "poisson"]:
            control_means = (
                torch.cat([x[0].rate for x in control_dists], axis=0)
                .detach()
                .cpu()
                .numpy()
                .squeeze(1)
            )
        elif output_dist == "zigp":
            control_means = (
                torch.cat([x[0].total_count for x in control_dists], axis=0)
                .detach()
                .cpu()
                .numpy()
                .squeeze(1)
            )
        data_dict["control_means"] = control_means
    else:
        for i in range(n_control):
            if output_dist in ["zip", "poisson"]:
                cur_means = (
                    torch.cat([x[i].rate for x in control_dists], axis=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(1)
                )
            elif output_dist == "zigp":
                cur_means = (
                    torch.cat([x[i].total_count for x in control_dists], axis=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(1)
                )
            data_dict["control_means_%d" % i] = cur_means

    if len(control_dists[0]) == 1:
        if output_dist in ["zip", "poisson"]:
            target_means = (
                torch.cat([x[0].rate for x in target_dists], axis=0)
                .detach()
                .cpu()
                .numpy()
                .squeeze(1)
            )
        elif output_dist == "zigp":
            target_means = (
                torch.cat([x[0].total_count for x in target_dists], axis=0)
                .detach()
                .cpu()
                .numpy()
                .squeeze(1)
            )
        data_dict["target_means"] = target_means
    else:
        for j in range(n_target):
            if output_dist in ["zip", "poisson"]:
                cur_means = (
                    torch.cat([x[j].rate for x in target_dists], axis=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(1)
                )
            elif output_dist == "zigp":
                cur_means = (
                    torch.cat([x[j].total_count for x in target_dists], axis=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(1)
                )
            data_dict["target_means_%d" % j] = cur_means

    test_table = wandb.Table(dataframe=pd.DataFrame(data_dict))

    max_rep = max(n_control, n_target)
    fig, axs = plt.subplots(2, max_rep, figsize=(18, 12))
    for i in range(n_control):
        cur_counts = data_dict["control_%d" % i]
        cur_scores = data_dict["log_control_scores"]
        cur_means = (
            data_dict["control_means"]
            if len(control_dists[0]) == 1
            else data_dict["control_means_%d" % i]
        )
        cur_means_p = cur_means * (1 - data_dict["control_p"])
        axs[0, i].scatter(cur_counts, cur_means, s=1.5, alpha=0.35, label="means")
        axs[0, i].scatter(cur_counts, cur_means_p, s=1.5, alpha=0.35, label="means_p")
        axs[0, i].scatter(
            cur_counts,
            data_dict["log_control_scores"],
            s=1.5,
            alpha=0.35,
            label="log scores",
        )

        corr = stats.pearsonr(cur_counts, cur_means)[0]
        corr_p = stats.pearsonr(cur_counts, cur_means_p)[0]
        corr_scores = stats.spearmanr(cur_counts, cur_scores).correlation
        axs[0, i].set_title(
            "Control %d, corr: %.2f, _p: %.2f, _scores: %.2f"
            % (i, corr, corr_p, corr_scores)
        )
        axs[0, i].set_xlabel("Control Counts")
        axs[0, i].set_ylabel("Predicted Enrichment")
        axs[0, i].legend()

    for j in range(n_target):
        cur_counts = data_dict["target_%d" % j]
        cur_scores = data_dict["log_control_scores"]
        cur_means = (
            data_dict["target_means"]
            if len(target_dists[0]) == 1
            else data_dict["target_means_%d" % j]
        )
        cur_means_p = cur_means * (1 - data_dict["target_p"])
        axs[1, j].scatter(cur_counts, cur_means, s=15.0, alpha=0.35, label="means")
        axs[1, j].scatter(cur_counts, cur_means_p, s=1.5, alpha=0.35, label="means_p")
        axs[1, j].scatter(
            cur_counts,
            data_dict["log_target_scores"],
            s=1.5,
            alpha=0.35,
            label="scores",
        )

        corr = stats.pearsonr(cur_counts, cur_means)[0]
        corr_p = stats.pearsonr(cur_counts, cur_means_p)[0]
        corr_scores = stats.spearmanr(cur_counts, cur_scores).correlation
        axs[1, j].set_title(
            "Target %d, corr: %.2f, _p: %.2f, _scores: %.2f"
            % (i, corr, corr_p, corr_scores)
        )
        axs[1, j].set_xlabel("Target Counts")
        axs[1, j].set_ylabel("Predicted Enrichment")
        axs[1, j].legend()

    fig.tight_layout()
    count_plots = [wandb.Image(fig)]
    return {"test/test_table": test_table, "test/count_plots": count_plots}
