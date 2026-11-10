import torch
import torch.nn as nn


def physics_features(x: torch.Tensor) -> torch.Tensor:
    """Expand the 12-D kinematic state into 33 physics-informed features.

    ``x`` is ``[nu(6), nu_dot(6)]``. The output is

        [nu(6), nu_dot(6), nu_i|nu_i|(6), nu_i*nu_j for i<j (15)]

    i.e. the raw state plus the quadratic drag basis and the cross-axis
    products. Giving the network this basis stops it from using the
    acceleration channel as a shortcut for the nonlinear drag law, which is
    what destabilises the plain MLP in closed loop (see the paper, Sec. V-A).
    """
    nu = x[..., :6]
    nud = x[..., 6:12]
    quad = nu * nu.abs()
    iu = torch.triu_indices(6, 6, offset=1, device=x.device)
    cross = nu[..., iu[0]] * nu[..., iu[1]]
    return torch.cat([nu, nud, quad, cross], dim=-1)


class DeepHydroMLP(nn.Module):
    """
    MLP that maps 6-DOF velocity + acceleration (12 inputs) to 6-DOF
    hydrodynamic force/torque (6 outputs).

    Architecture: 12 -> 128 -> 256 -> 128 -> 6  (bottleneck hourglass)
    Activation: LeakyReLU(0.1) to avoid dead neurons.
    """

    def __init__(self, input_dim=12, output_dim=6, hidden_dims=(128, 256, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.1)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)
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
    """
    Physics-informed residual model (paper Table I, ``PhysFeature``).

    The 12-D kinematic input is expanded to 33 features by
    :func:`physics_features` and then passed through the same
    128-256-128 hourglass used by the baseline (~71K parameters).
    """

    def __init__(self, input_dim=12, output_dim=6, hidden_dims=(128, 256, 128)):
        super().__init__()
        self.feature_dim = input_dim + 6 + 15
        layers = []
        prev = self.feature_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.1)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)
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


def build_model(model_type: str, input_dim: int = 12, output_dim: int = 6,
                hidden_dims=(128, 256, 128)) -> nn.Module:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. "
                         f"Choose from {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[model_type](input_dim=input_dim, output_dim=output_dim,
                                      hidden_dims=hidden_dims)
