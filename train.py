from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from dataset import build_dataloaders
from model import PrunableNet
from utils import (
    LOGGER,
    configure_logging,
    evaluate,
    format_results_table,
    gate_sparsity_percent,
    get_device,
    plot_gate_histogram,
    save_checkpoint,
    set_seed,
)

DEFAULT_LAMBDAS = [1e-4, 1e-3, 1e-2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-pruning neural network on CIFAR-10.")

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--lambda-value", type=float, default=None)
    parser.add_argument(
        "--lambda-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_LAMBDAS),
    )

    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    return parser.parse_args()


def train_one_epoch(
    model: PrunableNet,
    loader,
    optimizer,
    criterion,
    device,
    lambda_value: float,
):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        classification_loss = criterion(outputs, targets)

        # 🔥 NORMALIZED sparsity loss (IMPORTANT FIX)
        sparsity = model.sparsity_loss() / model.gate_values().numel()

        loss = classification_loss + lambda_value * sparsity

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def run_experiment(
    lambda_value: float,
    train_loader,
    test_loader,
    device,
    epochs,
    lr,
    output_dir: Path,
):
    model = PrunableNet().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_model_path = output_dir / f"best_model_lambda_{lambda_value:.0e}.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, lambda_value
        )

        test_loss, test_acc = evaluate(model, test_loader, device)
        sparsity_now = gate_sparsity_percent(model)

        LOGGER.info(
            "λ=%s | epoch=%d/%d | train_loss=%.4f | test_acc=%.2f%% | sparsity=%.2f%%",
            lambda_value,
            epoch,
            epochs,
            train_loss,
            test_acc,
            sparsity_now,
        )

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": test_acc,
                    "lambda": lambda_value,
                },
                best_model_path,
            )

    # 🔥 LOAD BEST MODEL (IMPORTANT FIX)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_loss, final_acc = evaluate(model, test_loader, device)
    sparsity = gate_sparsity_percent(model)

    # Plot gate distribution
    gate_vals = model.gate_values().detach().cpu().numpy()
    plot_gate_histogram(
        gate_vals,
        output_dir / f"gate_hist_lambda_{lambda_value:.0e}.png",
        title=f"Gate Distribution (λ={lambda_value})",
    )

    return {
        "lambda": lambda_value,
        "accuracy": final_acc,
        "sparsity": sparsity,
        "loss": final_loss,
    }


def main():
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    device = get_device()
    LOGGER.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Cleaner lambda handling
    if args.lambda_value is not None:
        lambda_values = [args.lambda_value]
    else:
        lambda_values = [float(v) for v in args.lambda_values.split(",") if v.strip()]

    results = []

    for lambda_value in lambda_values:
        LOGGER.info(f"\nStarting experiment for λ={lambda_value}")
        res = run_experiment(
            lambda_value,
            train_loader,
            test_loader,
            device,
            args.epochs,
            args.lr,
            output_dir,
        )
        results.append(res)

    LOGGER.info("\n" + format_results_table(results))


if __name__ == "__main__":
    main()