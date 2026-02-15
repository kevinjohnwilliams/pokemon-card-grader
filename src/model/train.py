"""
PokéGrader — Training Pipeline

Trains the card grading model on labeled card images.

Usage:
    python -m src.model.train --config configs/default.yaml

TODO: Implement once training data is available.
"""

import argparse


def train(config_path: str):
    """Train the card grading model."""
    # TODO: Load config
    # TODO: Load and split dataset
    # TODO: Set up data loaders with augmentation
    # TODO: Initialize model
    # TODO: Training loop with validation
    # TODO: Save best model checkpoint
    raise NotImplementedError("Training will be implemented once data is collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PokéGrader model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
