from torch import nn

class TestModel(nn.Module):
    def __init__(self, num_features):
        super(TestModel, self).__init__()
        self.norm = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        print(x)
        x = self.norm(x)
        print(x)

        return x
