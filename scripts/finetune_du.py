import os

from src.training.trainer import ModelTrainer
from src.training.data_loader import load_cifar10
from src.training.utils import set_random_seed
from src.utils import load_config

def main():
    # Load config
    config_path = "configs/finetune_D0_config.yaml"
    config = load_config(config_path)
    seed = config.get("seed", 233)
    set_random_seed(seed)

    # Load CIFAR-10 in unlearning setting
    _, Du_loader, _, _, val_loader = load_cifar10(
        data_path=config["data"]["path"],
        batch_size=config["fine_tune"]["batch_size"],
        public_split=config["data"]["public_split"],
        augment=config["data"]["augment"],
        seed=seed
    )

    os.makedirs(config["fine_tune"]["model_save_folder"], exist_ok=True)

    # Create trainer
    trainer = ModelTrainer(config)

    trainer.load_model(config["fine_tune"]["weights_load_path"])

    # --- finetune on D_u ---
    trainer.fine_tune_model(
        train_loader=Du_loader,
        val_loader=val_loader,
        epochs=config["fine_tune"]["epochs"],
        lr=config["fine_tune"]["lr"],
    )
    trainer.save_model(os.path.join(config["fine_tune"]["model_save_folder"],"finetuned_model.pth"))
    print("✅ Saved finetuned model")

if __name__ == "__main__":
    main()
