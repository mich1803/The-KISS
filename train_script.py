"""Train the single final Conditional NCA model used by the future web app."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import torch

from kiss_nca.config import load_config, resolve_device, validate_config
from kiss_nca.dataset import load_paintings
from kiss_nca.model import ConditionalNCA
from kiss_nca.training import save_checkpoint, train_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train The KISS final web-app NCA model.")
    parser.add_argument("--config", default="webapp/model_config.yaml", help="Path to YAML model configuration.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible training runs.")
    parser.add_argument("--iterations", type=int, default=None, help="Optional quick-run override for training iterations.")
    return parser.parse_args()


def main() -> None:
    """Load config, train the model, and save checkpoint plus metadata."""

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    validate_config(config)
    if args.iterations is not None:
        config["training"]["iterations"] = args.iterations

    device = resolve_device(config.get("device", "auto"))
    targets, records = load_paintings(
        dataset_dir=config.get("dataset_dir", "paintings/64"),
        painting_names=config["paintings"],
        device=device,
    )
    model = ConditionalNCA(
        state_channels=int(config["state_channels"]),
        hidden_channels=int(config["hidden_channels"]),
        num_paintings=len(records),
        condition_dim=int(config["condition_dim"]),
        neighborhood_size=int(config["neighborhood_size"]),
        update_rate=float(config["update_rate"]),
    ).to(device)

    train_model(model, targets, config["training"])
    save_checkpoint(
        model=model,
        config=config,
        records=records,
        model_path=config["output"]["model_path"],
        metadata_path=config["output"]["metadata_path"],
    )
    print(f"Saved model to {config['output']['model_path']}")
    print(f"Saved metadata to {config['output']['metadata_path']}")


if __name__ == "__main__":
    main()
