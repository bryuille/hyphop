import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_tiny_imagenet import main as run_tiny_imagenet_experiment


def run_experiment(model_name, hidden_dim, trial_seed, epochs=14, device="auto"):
    test_args = [
        "test_tiny_imagenet.py",
        "--model", model_name,
        "--hidden-dim", str(hidden_dim),
        "--epochs", str(epochs),
        "--lr", "0.001",
        "--gamma", "0.96",
        "--seed", str(trial_seed),
        "--log-interval", "100",
        "--device", device,
    ]

    with patch.object(sys, 'argv', test_args):
        return run_tiny_imagenet_experiment()


def main():
    device = "auto"

    model_names = ["kf_attention", "hf_attention", "ein_attention"]
    hidden_dim = 64
    num_trials = 5
    epochs = 14

    result_rows = []

    for model_name in model_names:
        trial_scores = []

        print("\n" + "#" * 70)
        print(f"Evaluating Model={model_name}, hidden_dim={hidden_dim}")
        print("#" * 70)

        for trial_idx in range(num_trials):
            trial_seed = 42 + trial_idx
            print(f"Trial {trial_idx + 1}/{num_trials} (seed={trial_seed})")

            acc = run_experiment(
                model_name=model_name,
                hidden_dim=hidden_dim,
                trial_seed=trial_seed,
                epochs=epochs,
                device=device,
            )
            trial_scores.append(acc)
            print(f"Final Accuracy: {acc:.2f}%")

        mean_acc = np.mean(trial_scores)
        std_acc = np.std(trial_scores)

        result_rows.append({
            "Model": model_name,
            "Hidden Dim": hidden_dim,
            "Mean Accuracy": f"{mean_acc:.2f}%",
            "Std Dev": f"{std_acc:.2f}%"
        })

    df = pd.DataFrame(result_rows)
    output_dir = os.path.join("results", "tiny_imagenet")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tiny_imagenet_benchmark_results.csv")
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 40)
    print("FINAL TINY IMAGENET ATTENTION BENCHMARK")
    print("=" * 40)
    print(f"Saved to: {os.path.abspath(output_file)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

