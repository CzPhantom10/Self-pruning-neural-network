"""Shared training and visualization helpers."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ensures plotting works without GUI

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def collect_gate_values(model: torch.nn.Module) -> np.ndarray:
    if hasattr(model, "gate_values"):
        return model.gate_values().detach().cpu().numpy()
    return np.array([])


def gate_sparsity_percent(model: torch.nn.Module, threshold: float = 1e-2) -> float:
    gates = collect_gate_values(model)
    if gates.size == 0:
        return 0.0
    return float((gates < threshold).mean() * 100.0)


def plot_gate_histogram(gate_values: np.ndarray, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(gate_values, bins=100, range=(0.0, 1.0), alpha=0.85)
    plt.title(title)
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def format_results_table(results: list[dict[str, float]]) -> str:
    headers = ["Lambda", "Accuracy", "Sparsity %"]

    rows = [
        headers,
        *[
            [
                f"{r['lambda']:.4g}",
                f"{r['accuracy']:.2f}",
                f"{r['sparsity']:.2f}",
            ]
            for r in results
        ],
    ]

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    lines = []
    lines.append(" | ".join(rows[0][i].ljust(widths[i]) for i in range(len(headers))))
    lines.append(" | ".join("-" * widths[i] for i in range(len(headers))))

    for row in rows[1:]:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    return "\n".join(lines)


def save_checkpoint(state: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)