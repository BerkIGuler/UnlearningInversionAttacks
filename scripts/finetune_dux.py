import os
import torch

from src.training.trainer import ModelTrainer
from src.training.data_loader import load_cifar10
from src.training.utils import set_random_seed
from src.utils import load_config


def main():
    config_path = "configs/finetune_Dux_config.yaml"
    config = load_config(config_path)
    seed = config.get("seed", 233)
    set_random_seed(seed)

    classes_to_attack = config["attack"]["classes"]
    attack_proportions = config["attack"]["proportion"]

    os.makedirs(config["fine_tune"]["model_save_folder"], exist_ok=True)

    total_experiments = len(classes_to_attack) * len(attack_proportions)
    current = 0
    successful_experiments = 0
    failed_experiments = []

    for attack_class_id in classes_to_attack:
        for prop in attack_proportions:
            current += 1
            print(f"\n{'=' * 60}")
            print(f"Experiment {current}/{total_experiments}: class={attack_class_id}, prop={prop}")
            print(f"{'=' * 60}")

            try:
                _, _, DuX_loader, _, val_loader = load_cifar10(
                    data_path=config["data"]["path"],
                    batch_size=config["fine_tune"]["batch_size"],
                    public_split=config["data"]["public_split"],
                    augment=config["data"]["augment"],
                    seed=seed,
                    exclude_class=attack_class_id,
                    exclude_prop=prop
                )

                trainer = ModelTrainer(config)

                trainer.load_model(config["fine_tune"]["weights_load_path"])

                print(f"Starting fine-tuning for class {attack_class_id} with proportion {prop}...")
                trainer.fine_tune_model(
                    train_loader=DuX_loader,
                    val_loader=val_loader,
                    epochs=config["fine_tune"]["epochs"],
                    lr=config["fine_tune"]["lr"],
                )

                model_save_path = os.path.join(
                    config["fine_tune"]["model_save_folder"],
                    f"finetuned_model_{attack_class_id}_{prop}.pth"
                )
                trainer.save_model(model_save_path)
                print(f"✅ Saved finetuned model to {model_save_path}")
                successful_experiments += 1

                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in experiment {current}: {str(e)}")
                failed_experiments.append({
                    'experiment': current,
                    'class_id': attack_class_id,
                    'proportion': prop,
                    'error': str(e)
                })

                try:
                    if 'trainer' in locals():
                        del trainer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

                print(f"Continuing with remaining experiments...")
                continue

    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {len(failed_experiments)}")

    if failed_experiments:
        print(f"\nFailed experiments:")
        for failure in failed_experiments:
            print(f"  - Experiment {failure['experiment']}: class={failure['class_id']}, "
                  f"prop={failure['proportion']} | Error: {failure['error']}")
    else:
        print("All experiments completed successfully!")


if __name__ == "__main__":
    main()