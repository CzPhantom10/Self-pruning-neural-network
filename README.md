# Self-Pruning Neural Network

This project implements a PyTorch classifier that learns to prune its own weights during training. Each linear layer is augmented with learnable gates so the network can suppress unimportant connections while learning CIFAR-10.

## How It Works

The core building block is `PrunableLinear`. It stores three learnable tensors:

- `weight`
- `bias`
- `gate_scores`

At each forward pass:

- `gates = sigmoid(gate_scores)`
- `pruned_weights = weight * gates`
- `linear(x, pruned_weights, bias)`

Because gates are differentiable, gradients flow through both the dense weights and the gate scores.

The full network, `PrunableNet`, is a simple multilayer perceptron for flattened CIFAR-10 images:

- `PrunableLinear -> ReLU`
- `PrunableLinear -> ReLU`
- `PrunableLinear -> 10-class output`

## Why L1 Encourages Sparsity

The training objective combines classification loss with a sparsity penalty:

$$
\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum \sigma(\text{gate\_scores})
$$

The penalty pushes gate values downward. Since sigmoid outputs are non-negative, minimizing the sum of gate activations encourages many gates to move close to zero. In practice this produces a sparse connectivity pattern without a separate pruning step.

## Project Structure

- [model.py](model.py) contains the custom prunable layer and the network.
- [dataset.py](dataset.py) loads CIFAR-10 with normalization and DataLoader setup.
- [utils.py](utils.py) contains evaluation, sparsity, checkpointing, and plotting helpers.
- [train.py](train.py) runs training, evaluation, and the lambda sweep.

## Installation

```bash
pip install -r requirements.txt
```

## How To Run

Train a single model:

```bash
python train.py --lambda-value 0.001 --epochs 5
```

Run the requested lambda sweep:

```bash
python train.py --epochs 5
```

Useful options:

- `--epochs`: number of training epochs per lambda value
- `--batch-size`: DataLoader batch size
- `--lr`: Adam learning rate
- `--data-dir`: CIFAR-10 download/cache directory
- `--output-dir`: directory for checkpoints and gate histograms

## Evaluation Metrics

For each experiment the script reports:

- test accuracy (percent)
- sparsity percent = fraction of gates below `1e-2`

Artifacts written to `--output-dir`:

- `best_model_lambda_*.pt` (best checkpoint by test accuracy)
- `gate_hist_lambda_*.png` (histogram of gate values)

## Observations On Lambda Tradeoff

- Lower lambda values usually preserve accuracy better, but the network stays denser.
- Higher lambda values typically increase sparsity, but they can reduce test accuracy if the penalty is too strong.
- The gate histogram should show a visible concentration near zero when pruning is effective.

## Notes

- CIFAR-10 is downloaded automatically through `torchvision`.
- Training uses GPU if available; otherwise it runs on CPU.
