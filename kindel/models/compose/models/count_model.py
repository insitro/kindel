import torch
import torch.nn as nn
from torch.distributions import NegativeBinomial, Poisson


class CountModel(nn.Module):
    """
    Takes in enrichment scores for control and target, and computes the observation distributions
    """

    def __init__(self, cfg, n_control, n_target):
        super().__init__()

        self.cfg = cfg

        self.n_control = n_control
        self.n_target = n_target

        self.output_dist = cfg.output_dist

        # Load bias
        if cfg.use_pre:
            self.log_load_control_beta = nn.Parameter(
                data=torch.tensor(1.0), requires_grad=True
            )
            self.log_load_target_beta = nn.Parameter(
                data=torch.tensor(1.0), requires_grad=True
            )
            nn.init.normal_(self.log_load_control_beta)
            nn.init.normal_(self.log_load_target_beta)

        # Replicate bias
        if cfg.rep_embed:
            self.log_control_rep_beta = nn.Parameter(
                data=torch.ones(n_control), requires_grad=True
            )
            self.log_target_rep_beta = nn.Parameter(
                data=torch.ones(n_target), requires_grad=True
            )

            nn.init.normal_(self.log_control_rep_beta)
            nn.init.normal_(self.log_target_rep_beta)

        # Gamma dispersion parameter
        if cfg.output_dist == "zigp":
            self.log_control_gamma_dispersion = nn.Parameter(
                data=torch.tensor(1.0), requires_grad=True
            )
            self.log_target_gamma_dispersion = nn.Parameter(
                data=torch.tensor(1.0), requires_grad=True
            )

            nn.init.normal_(self.log_control_gamma_dispersion)
            nn.init.normal_(self.log_target_gamma_dispersion)

    def compute_negative_log_likelihood(self, dists, zero_prob, counts):
        """
        Parameters
        ----------
        dists: list of distributions for each replicate
            each distribution is: [batch_size,  n_samples]
            if no replicate bias is incorporated, then there is only one distribution
        zero_prob: A tensor of the zero_probabilities, [batch_size, n_samples]
            Currently shared between different replicates
        counts: [batch_size, n_replicate] observed count data
        """
        n_replicates = counts.shape[1]

        sum_nll = []
        # iterate over every replicate, because some models have per-replicate factors
        for rep_idx in range(n_replicates):
            cur_counts = counts[:, rep_idx].unsqueeze(
                1
            )  # [batch_size, 1] get the current replicate counts
            cur_dist = dists[0] if len(dists) == 1 else dists[rep_idx]
            if zero_prob is None:
                sum_nll += -1 * cur_dist.log_prob(cur_counts)
            else:
                count_mask = (
                    cur_counts != 0
                ).float()  # [batch_size, 1] create a 1/0 mask for counts

                log_zero_prob = torch.log(
                    zero_prob
                    + (1 - zero_prob)
                    * torch.exp(cur_dist.log_prob(torch.zeros_like(zero_prob)))
                )
                log_nonzero_prob = torch.log(1 - zero_prob) + cur_dist.log_prob(
                    cur_counts
                )

                sum_nll.append(
                    -1
                    * ((1 - count_mask) * log_zero_prob + count_mask * log_nonzero_prob)
                )

        return torch.cat(sum_nll, dim=1)

    def forward(self, log_control_scores, log_target_scores, pre=None, control_p=None):
        """
        Parameters
        ----------
        log_control_scores: [batch_size] tensor of log(control scores)
        log_target_scores: [batch_size] tensor of log(target scores)
        pre: [batch_size] tensor of pre populations; if None, all compounds have equal weight
        control_p: [batch_size] tensor of control zero probabilities for zero-inflated distributions
        """

        control_enrichment = torch.exp(log_control_scores)
        target_enrichment = torch.exp(log_target_scores)

        if self.cfg.use_pre:
            assert pre is not None
            pre = pre.unsqueeze(axis=1)
            control_pre_factor = pre
            target_pre_factor = pre
        else:
            control_pre_factor, target_pre_factor = 1.0, 1.0

        control_means, target_means = [], []
        if self.cfg.rep_embed:
            for idx in range(self.n_control):
                cur_control_mean = (
                    control_pre_factor
                    * control_enrichment
                    * torch.exp(self.log_control_rep_beta[idx])
                )
                control_means.append(cur_control_mean)

            for idx in range(self.n_target):
                control_to_use = (
                    control_enrichment
                    if not self.cfg.detach_control
                    else control_enrichment.detach()
                )
                cur_target_mean = (
                    target_pre_factor
                    * (target_enrichment * control_to_use)
                    * torch.exp(self.log_target_rep_beta[idx])
                )
                target_means.append(cur_target_mean)
        else:
            control_means.append(control_pre_factor * control_enrichment)

            control_to_use = (
                control_enrichment
                if not self.cfg.detach_control
                else control_enrichment.detach()
            )
            target_means.append(
                target_pre_factor * (target_enrichment * control_to_use)
            )

        control_dists, target_dists = [], []
        if self.output_dist in ["poisson", "zip"]:
            for mean in control_means:
                control_dists.append(Poisson(rate=mean))
            for mean in target_means:
                target_dists.append(Poisson(rate=mean))
        elif self.output_dist in ["zigp"]:
            for mean in control_means:
                gp_probs = (
                    torch.sigmoid(self.log_control_gamma_dispersion).tile(mean.shape)
                    + 1e-6
                )
                control_dists.append(NegativeBinomial(total_count=mean, probs=gp_probs))
            for mean in target_means:
                gp_probs = (
                    torch.sigmoid(self.log_target_gamma_dispersion).tile(mean.shape)
                    + 1e-6
                )
                target_dists.append(NegativeBinomial(total_count=mean, probs=gp_probs))

        return control_dists, target_dists
