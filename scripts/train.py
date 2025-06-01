import os

import yaml

from src.training.trainer import ModelTrainer
from src.training.data_loader import load_cifar10
from src.training.utils import set_random_seed

def load_config(path="configs/debug_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config()
    seed = config.get("seed", 233)
    set_random_seed(seed)

    # Load CIFAR-10 in unlearning setting
    D0_loader, Du_loader, DuX_loader, X_loader, test_loader = load_cifar10(
        data_path=config["data"]["path"],
        batch_size=config["train"]["batch_size"],
        public_split=config["data"]["public_split"],
        augment=config["data"]["augment"],
        seed=seed,
        exclude_class=config["data"]["class"],
        exclude_prop=config["data"]["proportion"]
    )

    os.makedirs(config["train"]["model_save_folder"], exist_ok=True)
    os.makedirs(config["fine_tune"]["model_save_folder"], exist_ok=True)


    # Create trainer
    trainer = ModelTrainer(config)

    # --- Train on D₀ ---
    trainer.train_model(
        train_loader=D0_loader,
        epochs=config["train"]["epochs"],
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]
    )
    trainer.save_model(os.path.join(config["train"]["model_save_folder"],"pretrained.pth"))
    print("✅ Saved pretrained model")

    # --- Fine-tune on Dᵤ ---
    trainer.fine_tune_model(
        train_loader=Du_loader,
        epochs=config["fine_tune"]["epochs"],
        lr=config["fine_tune"]["lr"]
    )
    trainer.save_model(os.path.join(config["fine_tune"]["model_save_folder"],"theta.pth"))
    print("Saved fine-tuned model (θ)")

if __name__ == "__main__":
    main()
