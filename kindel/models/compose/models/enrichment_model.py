import torch
import torch.nn as nn

from kindel.models.compose.models.layers import get_mlp_layer


class EnrichmentModel(nn.Module):
    """
    Computes representations of molecules or decomposed synthons
    """

    def __init__(self, cfg, input_dim, output_dim):
        """
        Parameters
        ----------
        cfg: model config
        input_dim: the dimensions of the input vector
        output_dim: the dimensions of the output vector
        """
        super().__init__()
        self.cfg = cfg
        self.output_dim = output_dim
        hidden_dim = cfg.hidden_dim

        model_type = cfg.model_type
        # n_models denote the number of distinct molecule representation models
        if model_type == "full":
            n_models = 1
        elif model_type == "factorized":
            n_models = 1 if cfg.share_embeddings else 3
        else:
            raise ValueError("Unrecognized model type: {model_type}")

        if cfg.embed_type == "gnn":
            n_models = 1

        self.mask = cfg.mask

        self.mol_nn = nn.ModuleList([])
        for _ in range(n_models):
            if cfg.embed_type == "fps":
                self.mol_nn.append(
                    get_mlp_layer(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        n_layers=int(cfg.n_layers),
                    )
                )
            else:
                raise ValueError(f"Unrecognized embedding type: {cfg.embed_type}")

        synthon_agg_model = cfg.synthon_agg_model
        if model_type == "factorized":
            if synthon_agg_model == "rnn":
                self.synthon_lstm = nn.LSTM(
                    input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=False
                )
            elif synthon_agg_model in ["mlp", "attention"]:
                self.synthon_AB = get_mlp_layer(
                    input_dim=hidden_dim * 2,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                )
                self.synthon_AC = get_mlp_layer(
                    input_dim=hidden_dim * 2,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                )
                self.synthon_BC = get_mlp_layer(
                    input_dim=hidden_dim * 2,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                )
                self.synthon_ABC = get_mlp_layer(
                    input_dim=hidden_dim * 2,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                )

                if synthon_agg_model == "attention":
                    self.synthon_attention = nn.MultiheadAttention(hidden_dim, 4)

        # Computes the enrichment scores for both the control and target
        self.enrichment_score_mlp = get_mlp_layer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim * 2
        )

        # Computes the zero probability scores for both the control and the target
        self.zero_prob_mlp = get_mlp_layer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim * 2
        )

    def compute_embeddings(self, data_dict):
        """
        Computes the latent representations

        Parameters
        ----------
        data_dict:
            fps_a, fps_b, fps_c: are tensors of fingerprints for synthons
            fps: a tensor of fingerprints for the entire DEL molecules
            smiles_a, smiles_b, smiles_c: list of smiles for synthons
            smiles: list of smiles for entire DEL molecules
        """
        embed_dict = {}  # output dictionary that contains all the embeddings
        if self.cfg.model_type == "factorized":
            if self.cfg.embed_type == "fps":
                input_a, input_b, input_c = (
                    data_dict["fps_a"],
                    data_dict["fps_b"],
                    data_dict["fps_c"],
                )
            else:
                assert False

            i_a, i_b, i_c = 0, 0, 0
            if not self.cfg.share_embeddings:
                i_b, i_c = 1, 2

            z_a = self.mol_nn[i_a](input_a)
            if self.mask is not None and "a" in self.mask:
                z_a = torch.ones_like(z_a)
            z_b = self.mol_nn[i_b](input_b)
            if self.mask is not None and "b" in self.mask:
                z_b = torch.ones_like(z_b)
            z_c = self.mol_nn[i_c](input_c)
            if self.mask is not None and "c" in self.mask:
                z_c = torch.ones_like(z_c)

            embed_dict.update({"z_a": z_a, "z_b": z_b, "z_c": z_c})

            if self.cfg.synthon_agg_model == "rnn":
                rnn_input = torch.stack(
                    [z_a, z_b, z_c], dim=0
                )  # [seq_length=3, batch, hidden_dim]
                z_out, _ = self.synthon_lstm(
                    rnn_input
                )  # [seq_length=3, batch, hidden_dim * 2]
                z_out = z_out[-1, :, :]
            elif self.cfg.synthon_agg_model == "mlp":
                z_ab = self.synthon_AB(torch.concat([z_a, z_b], dim=1))
                z_out = self.synthon_ABC(torch.concat([z_ab, z_c], dim=1))
            elif self.cfg.synthon_agg_model == "attention":
                z_ab = self.synthon_AB(torch.concat([z_a, z_b], dim=1))
                z_ac = self.synthon_AC(torch.concat([z_a, z_c], dim=1))
                z_bc = self.synthon_BC(torch.concat([z_b, z_c], dim=1))

                z_abc = self.synthon_ABC(torch.concat([z_ab, z_c], dim=1))

                z_all = torch.stack([z_a, z_b, z_c, z_ab, z_bc, z_ac, z_abc], dim=0)
                z_out, attn_probs = self.synthon_attention(
                    query=z_all[-1, :, :].unsqueeze(0), key=z_all, value=z_all
                )
                z_out = z_out.squeeze(0)

                embed_dict.update({"attn_probs": attn_probs})

            else:
                assert False

            embed_dict.update({"z": z_out})

        elif self.cfg.model_type == "full":
            if self.cfg.embed_type == "fps":
                input = data_dict["fps"]
            else:
                raise ValueError(f"Unrecognized embedding type: {self.cfg.embed_type}")

            z_out = self.mol_nn[0](input)
            embed_dict.update({"z": z_out})
        return embed_dict

    def compute_enrichments(self, z):
        """
        Computes the enrichments given latent representations
        """

        log_enrichment_scores = self.enrichment_score_mlp(z)
        log_control_scores = log_enrichment_scores[:, 0].unsqueeze(1)
        log_target_scores = log_enrichment_scores[:, 1].unsqueeze(1)

        scores_dict = {
            "log_control_scores": log_control_scores,
            "log_target_scores": log_target_scores,
        }
        return scores_dict

    def compute_zero_probs(self, z):
        """
        Computes the zero probabilities given latent representations
        """
        zero_probs = torch.sigmoid(self.zero_prob_mlp(z))
        control_zero_probs = zero_probs[:, 0].unsqueeze(1)
        target_zero_probs = zero_probs[:, 1].unsqueeze(1)

        probs_dict = {
            "control_zero_probs": control_zero_probs,
            "target_zero_probs": target_zero_probs,
        }
        return probs_dict
