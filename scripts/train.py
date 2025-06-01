import sys
import os

import yaml
import torch

from src.training.trainer import ModelTrainer
from src.training.data_loader import load_cifar10
from src.training.utils import set_random_seed

def load_config(path="configs/training_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config()
    seed = config.get("seed", 233)
    set_random_seed(seed)

    # Load CIFAR-10
    D0_loader, Du_loader, DuX_loader, X_loader, test_loader = load_cifar10(
        data_path=config["data"]["path"],
        batch_size=config["train"]["batch_size"],
        public_split=config["data"]["public_split"],
        augment=config["data"]["augment"],
        seed=seed,
        excluded_class=config["data"]["class"],
        excluded_proportion=config["data"]["proportion"]
    )


    # Create trainer
    trainer = ModelTrainer(config)

    # --- Train on D₀ ---
    trainer.train_model(
        train_loader=D0_loader,
        epochs=config["train"]["epochs"],
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]
    )
    trainer.save_model("theta_u.pth")
    print("✅ Saved pretrained model (θᵤ)")

    # --- Fine-tune on Dᵤ ---
    trainer.fine_tune_model(
        train_loader=Du_loader,
        epochs=config["fine_tune"]["epochs"],
        lr=config["fine_tune"]["lr"]
    )
    trainer.save_model("theta.pth")
    print("Saved fine-tuned model (θ)")

if __name__ == "__main__":
    main()
