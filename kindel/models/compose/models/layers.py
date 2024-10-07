import torch.nn as nn


def get_mlp_layer(
    input_dim, hidden_dim, output_dim, n_layers=2, activation_fct=nn.ReLU
):
    layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), activation_fct())
    for _ in range(n_layers - 1):
        layer.append(nn.Linear(hidden_dim, hidden_dim))
        layer.append(activation_fct())

    layer.append(nn.Linear(hidden_dim, output_dim))
    return layer
