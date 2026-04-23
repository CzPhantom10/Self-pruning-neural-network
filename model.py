import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.uniform_(self.gate_scores, -0.5, 0.5)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        gates = torch.sigmoid(self.gate_scores * 5)  # sharper pruning
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def gate_values(self) -> Tensor:
        return torch.sigmoid(self.gate_scores)


class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def prunable_layers(self):
        return [self.fc1, self.fc2, self.fc3]

    def sparsity_loss(self) -> Tensor:
        return sum(layer.gate_values().sum() for layer in self.prunable_layers())

    def gate_values(self) -> Tensor:
        return torch.cat([layer.gate_values().view(-1) for layer in self.prunable_layers()])

    def gate_sparsity_percent(self, threshold: float = 1e-2) -> float:
        gates = self.gate_values()
        return (gates < threshold).float().mean().item() * 100.0