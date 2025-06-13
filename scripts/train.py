import os

from src.training.trainer import ModelTrainer
from src.training.data_loader import load_cifar10
from src.training.utils import set_random_seed
from src.utils import load_config


def main():
    config_path = "configs/training_config.yaml"
    config = load_config(config_path)
    seed = config.get("seed", 233)
    set_random_seed(seed)

    D0_loader, Du_loader, DuX_loader, X_loader, val_loader = load_cifar10(
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


    trainer = ModelTrainer(config)

    trainer.train_model(
        train_loader=D0_loader,
        val_loader=val_loader,
        epochs=config["train"]["epochs"],
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        scheduler=True
    )
    trainer.save_model(os.path.join(config["train"]["model_save_folder"],"pretrained.pth"))
    print("Saved pretrained model")

if __name__ == "__main__":
    main()
