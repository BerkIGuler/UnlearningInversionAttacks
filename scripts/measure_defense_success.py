import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from torchvision import transforms

from src.utils import load_config
from src.training.trainer import ModelTrainer
from src.training.data_loader import load_cifar10

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
CIFAR10_STD = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)


@dataclass
class DefenseResult:
    """Result for a single model pair evaluation"""
    class_id: int
    proportion: float
    prune_rate: float
    original_accuracy: float
    pruned_accuracy: float
    original_attack_success: bool
    pruned_attack_success: bool
    accuracy_drop: float
    defense_success: bool  # True if attack failed after pruning but succeeded before


class PruningDefenseEvaluator:
    """Evaluates pruning as defense against unlearning attacks"""

    def __init__(self, config_path: str, pruned_models_dir: str):
        self.config = load_config(config_path)
        self.pruned_models_dir = Path(pruned_models_dir)
        self.normalizer = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

        # Load test data and probing samples
        _, _, _, _, self.test_loader = load_cifar10(
            batch_size=128, augment=False
        )
        self.probing_samples = self._load_probing_samples()

    def _load_probing_samples(self) -> Dict[int, List[torch.Tensor]]:
        """Load probing samples for attack evaluation"""
        samples_path = Path(self.config["attack"]["probe_samples_path"])
        samples_dict = {}

        for pkl_file in samples_path.glob("*.pkl"):
            class_id = int(pkl_file.name.replace('.pkl', '').split('_')[-1])
            with open(pkl_file, "rb") as f:
                samples = pickle.load(f)
            samples_dict[class_id] = samples

        return samples_dict

    def _load_model(self, model_path: str):
        """Load model handling both original and pruned checkpoint formats"""
        trainer = ModelTrainer(self.config)
        checkpoint = torch.load(model_path, map_location='cpu')

        # Handle pruned model format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

            # Check if this is a pruned model (has weight_orig and weight_mask)
            has_pruning_masks = any('weight_orig' in key for key in state_dict.keys())

            if has_pruning_masks:
                # Convert pruned weights back to regular weights
                converted_state_dict = {}
                for key, value in state_dict.items():
                    if key.endswith('_orig'):
                        # Get the base key (remove _orig suffix)
                        base_key = key[:-5]  # Remove '_orig'
                        mask_key = base_key + '_mask'

                        if mask_key in state_dict:
                            # Apply mask to original weights
                            masked_weight = value * state_dict[mask_key]
                            converted_state_dict[base_key] = masked_weight
                        else:
                            # No mask found, use original weight
                            converted_state_dict[base_key] = value
                    elif not key.endswith('_mask'):
                        # Keep non-weight parameters as is
                        converted_state_dict[key] = value

                trainer.model.load_state_dict(converted_state_dict)
            else:
                # Regular checkpoint format
                trainer.model.load_state_dict(state_dict)
        else:
            # Handle original format (fallback to trainer's method)
            trainer.load_model(model_path)
            return trainer

        trainer.model.to(trainer.device)
        trainer.model.eval()
        return trainer

    def _evaluate_accuracy(self, model_path: str) -> float:
        """Evaluate model accuracy on test set"""
        trainer = self._load_model(model_path)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
                outputs = trainer.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return accuracy

    def _evaluate_attack_success(self, original_model_path: str, target_model_path: str, class_id: int) -> bool:
        """Check if attack succeeds by comparing confidence differences"""
        # Get original model confidence
        trainer = self._load_model(original_model_path)

        original_confs = []
        for sample in self.probing_samples[class_id]:
            with torch.no_grad():
                normalized = self.normalizer(sample.to(trainer.device))
                pred = trainer.model(normalized).softmax(dim=1).detach().cpu().numpy()
                original_confs.append(pred.flatten())

        original_pred = np.array(original_confs).mean(axis=0)
        del trainer

        # Get target model confidence
        trainer = self._load_model(target_model_path)

        target_confs = []
        for sample in self.probing_samples[class_id]:
            with torch.no_grad():
                normalized = self.normalizer(sample.to(trainer.device))
                pred = trainer.model(normalized).softmax(dim=1).detach().cpu().numpy()
                target_confs.append(pred.flatten())

        target_pred = np.array(target_confs).mean(axis=0)
        del trainer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Attack succeeds if target class has maximum absolute confidence difference
        conf_differences = original_pred - target_pred
        max_diff_class = np.argmax(np.abs(conf_differences))

        return max_diff_class == class_id

    def evaluate_model_pair(self, class_id: int, proportion: float, prune_rate: float) -> DefenseResult:
        """Evaluate a single original/pruned model pair"""
        # Construct paths
        original_name = f"finetuned_model_{class_id}_{proportion}.pth"
        pruned_name = f"pruned_{prune_rate}_{original_name}"

        original_path = Path(self.config["fine_tune"]["dux_weights_load_dir"]) / original_name
        pruned_path = self.pruned_models_dir / pruned_name
        du_path = self.config["fine_tune"]["du_weights_load_path"]

        # Evaluate accuracies
        original_acc = self._evaluate_accuracy(str(original_path))
        pruned_acc = self._evaluate_accuracy(str(pruned_path))

        # Evaluate attack success
        original_attack = self._evaluate_attack_success(du_path, str(original_path), class_id)
        pruned_attack = self._evaluate_attack_success(du_path, str(pruned_path), class_id)

        return DefenseResult(
            class_id=class_id,
            proportion=proportion,
            prune_rate=prune_rate,
            original_accuracy=original_acc,
            pruned_accuracy=pruned_acc,
            original_attack_success=original_attack,
            pruned_attack_success=pruned_attack,
            accuracy_drop=original_acc - pruned_acc,
            defense_success=(original_attack and not pruned_attack)
        )

    def run_evaluation(self, sample_classes: List[int], sample_proportions: List[float],
                       prune_rates: List[float]) -> List[DefenseResult]:
        """Run evaluation on selected model pairs"""
        results = []

        print(
            f"Evaluating {len(sample_classes)} classes × {len(sample_proportions)} proportions × {len(prune_rates)} prune rates")
        print(f"Total evaluations: {len(sample_classes) * len(sample_proportions) * len(prune_rates)}")

        for class_id in sample_classes:
            for proportion in sample_proportions:
                for prune_rate in prune_rates:
                    try:
                        result = self.evaluate_model_pair(class_id, proportion, prune_rate)
                        results.append(result)

                        defense_status = "SUCCESS" if result.defense_success else "FAILED"
                        print(f"Class {class_id}, prop {proportion}, prune {prune_rate}: "
                              f"acc_drop={result.accuracy_drop:.3f}, defense={defense_status}")

                    except Exception as e:
                        print(f"Error evaluating class {class_id}, prop {proportion}, prune {prune_rate}: {e}")

        return results

    def create_summary_plots(self, results: List[DefenseResult], save_dir: str = "defense_plots"):
        """Create summary visualizations"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Group results by prune rate
        by_prune_rate = {}
        for result in results:
            if result.prune_rate not in by_prune_rate:
                by_prune_rate[result.prune_rate] = []
            by_prune_rate[result.prune_rate].append(result)

        prune_rates = sorted(by_prune_rate.keys())

        # Calculate averages for each prune rate
        avg_accuracy_drop = []
        defense_success_rate = []

        for rate in prune_rates:
            rate_results = by_prune_rate[rate]
            avg_accuracy_drop.append(np.mean([r.accuracy_drop for r in rate_results]))
            defense_success_rate.append(np.mean([r.defense_success for r in rate_results]) * 100)

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Accuracy Drop vs Prune Rate
        ax1.bar(range(len(prune_rates)), avg_accuracy_drop, alpha=0.7, color='red')
        ax1.set_xlabel('Pruning Rate')
        ax1.set_ylabel('Average Accuracy Drop')
        ax1.set_title('Accuracy Cost of Pruning')
        ax1.set_xticks(range(len(prune_rates)))
        ax1.set_xticklabels([f'{rate:.1f}' for rate in prune_rates])
        ax1.grid(True, alpha=0.3)

        # Plot 2: Defense Success Rate vs Prune Rate
        ax2.bar(range(len(prune_rates)), defense_success_rate, alpha=0.7, color='green')
        ax2.set_xlabel('Pruning Rate')
        ax2.set_ylabel('Defense Success Rate (%)')
        ax2.set_title('Defense Effectiveness')
        ax2.set_xticks(range(len(prune_rates)))
        ax2.set_xticklabels([f'{rate:.1f}' for rate in prune_rates])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / 'pruning_defense_summary.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'pruning_defense_summary.pdf', bbox_inches='tight')
        plt.close()

        # Print summary statistics
        print(f"\n{'=' * 50}")
        print("PRUNING DEFENSE SUMMARY")
        print(f"{'=' * 50}")

        for i, rate in enumerate(prune_rates):
            print(f"Prune Rate {rate:.1f}: "
                  f"Avg Accuracy Drop = {avg_accuracy_drop[i]:.3f}, "
                  f"Defense Success = {defense_success_rate[i]:.1f}%")


def main():
    config_path = "configs/attack_config.yaml"
    pruned_models_dir = "outputs/pruned_models"

    # sample_classes = [0, 2, 4, 6, 8]
    sample_classes = [0, 2]
    # sample_proportions = [0.003, 0.01, 0.1, 0.25, 0.5, 0.9]
    sample_proportions = [0.01]
    # prune_rates = [0.1, 0.25, 0.5, 0.9]
    prune_rates = [0.5]

    evaluator = PruningDefenseEvaluator(config_path, pruned_models_dir)
    results = evaluator.run_evaluation(sample_classes, sample_proportions, prune_rates)
    evaluator.create_summary_plots(results)

    with open("pruning_defense_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Results saved to pruning_defense_results.pkl")


if __name__ == "__main__":
    main()