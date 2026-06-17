import torch.nn as nn

class BaselineRegressor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.layers = self._build_regressor()

    def _build_regressor(self):
        classifier = nn.Sequential(
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(True),
            nn.Linear(self.latent_dim, 1),
        )

        return classifier

    def forward(self, inputs):
        return self.layers(inputs).squeeze(1)
