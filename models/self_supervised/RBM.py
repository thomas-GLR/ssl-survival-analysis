import torch
from pytorch_probgraph import (
    InteractionModule,
    RestrictedBoltzmannMachineCD,
)

from models.self_supervised.base.layers import GaussianSequenceLayer, RectifiedLinearLayer


class RBM(RestrictedBoltzmannMachineCD):
    def __init__(self, in_units, out_units, he_init=False):
        l0bias = torch.zeros([1, in_units, 1])
        l0bias.requires_grad = True
        l1bias = torch.zeros([1, out_units, 1])
        l1bias.requires_grad = True

        l0 = GaussianSequenceLayer(l0bias, torch.ones_like(l0bias))
        l0.logsigma.requires_grad = False
        l1 = RectifiedLinearLayer(l1bias)

        module = torch.nn.Conv1d(14, out_units, kernel_size=3, bias=False)
        i0 = InteractionModule(module, l0.bias.shape[1:])
        if he_init:
            torch.nn.init.kaiming_uniform_(i0.weight, nonlinearity="relu")

        # build the RBM
        super().__init__(l0, l1, i0, ksteps=1)
