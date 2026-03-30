## Overview

This project benchmarks three attention families on three tasks:
- **MNIST** (classification)
- **Tiny ImageNet** (classification)
- **MIL** (AUC on tiger/fox/elephant)

Implemented model families:
- `kf_*` (Karcher Flow)
- `hf_*` (Hopfield)
- `ein_*` (Einstein-midpoint aggregation)

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train Single Runs

### MNIST (`test_mnist.py`)

```bash
python test_mnist.py [OPTIONS]
```

Key options and defaults:
- `--model` choices: `kf_attention`, `kf_layer`, `kf_pooling`, `hf_attention`, `hf_layer`, `hf_pooling`, `ein_attention`, `ein_layer`, `ein_pooling`
- `--batch-size`: `64`
- `--epochs`: `14`
- `--lr`: `0.001`
- `--gamma`: `0.96`
- `--hidden-dim`: `8`
- `--beta`: `None` (core derives to `1/sqrt(d)`)
- `--num-states`: `1`
- `--num-memories`: `64`
- `--seed`: `1`

Example:
```bash
python test_mnist.py --model ein_attention --hidden-dim 32
```

### Tiny ImageNet (`test_tiny_imagenet.py`)

```bash
python test_tiny_imagenet.py [OPTIONS]
```

Example:
```bash
python test_tiny_imagenet.py --model ein_attention --hidden-dim 128 --epochs 1 --dry-run
```

### MIL (`test_mil.py`)

```bash
python test_mil.py [OPTIONS]
```

Key options and defaults:
- `--model` choices: `kf_attention`, `kf_layer`, `kf_pooling`, `hf_attention`, `hf_layer`, `hf_pooling`, `ein_attention`, `ein_layer`, `ein_pooling`
- `--dataset` choices: `tiger`, `fox`, `elephant`
- `--batch-size`: `16`
- `--epochs`: `100`
- `--lr`: `0.001`
- `--gamma`: `0.96`
- `--hidden-dim`: `128`
- `--beta`: `None` (core derives to `1/sqrt(d)`)
- `--num-states`: `1`
- `--num-memories`: `64`
- `--bag-dropout`: `0.5`
- `--seed`: `1`

Example:
```bash
python test_mil.py --dataset fox --model ein_pooling
```

## Benchmark Scripts

### MNIST Table (`run_mnist_table.py`)

```bash
python run_mnist_table.py
```

Current benchmark defaults:
- Models: `["kf_attention", "hf_attention", "ein_attention"]`
- Hidden dims: `[4, 8, 32]`
- Trials: `5`
- Epochs: `14`
- Optimizer/LR/Gamma: `AdamW`, `0.001`, `0.96`

Output:
- `results/mnist/mnist_benchmark_results.csv`

### Tiny ImageNet Table (`run_tiny_imagenet.py`)

```bash
python run_tiny_imagenet.py
```

Current benchmark defaults:
- Models: `["kf_attention", "hf_attention", "ein_attention"]`
- Hidden dim: `64`
- Trials: `5` (seeds `42..46`)
- Epochs: `14`
- Optimizer/LR/Gamma: `AdamW`, `0.001`, `0.96`

Output:
- `results/tiny_imagenet/tiny_imagenet_benchmark_results.csv`

### MIL Table (`run_mil_table.py`)

```bash
python run_mil_table.py
```

Current benchmark defaults:
- Datasets: `["tiger", "fox", "elephant"]`
- Models: `["kf_pooling", "hf_pooling", "ein_pooling"]`
- Trials: `5` (seeds `42..46`)
- Epochs: `100`
- Batch size/LR/Gamma: `16`, `0.001`, `0.96`

Output:
- `results/mil/mil_benchmark_results.csv`