import torch
import torch.nn as nn


def physics_features(x: torch.Tensor) -> torch.Tensor:
    """Expand the 12-D kinematic state into 33 physics-informed features.

    Mirrors ``hydro_MLP/src/model.py::physics_features`` — the deployed model
    must build exactly the same basis it was trained on.
    """
    nu = x[..., :6]
    nud = x[..., 6:12]
    quad = nu * nu.abs()
    iu = torch.triu_indices(6, 6, offset=1, device=x.device)
    cross = nu[..., iu[0]] * nu[..., iu[1]]
    return torch.cat([nu, nud, quad, cross], dim=-1)


class DeepHydroMLP(nn.Module):
    def __init__(self, input_dim=12, output_dim=6):
        super(DeepHydroMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class PhysFeatureMLP(nn.Module):
    """Physics-informed residual model (paper Table I, ``PhysFeature``)."""

    def __init__(self, input_dim=12, output_dim=6):
        super().__init__()
        self.feature_dim = input_dim + 6 + 15
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(physics_features(x))


MODEL_REGISTRY = {
    "baseline": DeepHydroMLP,
    "physfeature": PhysFeatureMLP,
}


def build_model(model_type: str, input_dim: int = 12, output_dim: int = 6) -> nn.Module:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. "
                         f"Choose from {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[model_type](input_dim=input_dim, output_dim=output_dim)