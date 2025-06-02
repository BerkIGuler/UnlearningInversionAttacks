import pickle
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torchvision import transforms
from src.utils import load_config
from src.training.trainer import ModelTrainer

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
CIFAR10_STD = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)


@dataclass
class AttackResult:
    """Represents the result of a single unlearning attack"""
    target_class_id: int
    exclude_proportion: float
    original_confidence: float
    unlearned_confidence: float
    original_prediction: np.ndarray  # Full prediction vector
    unlearned_prediction: np.ndarray  # Full prediction vector
    confidence_differences: np.ndarray  # Signed differences (original - unlearned)
    confidence_drop: float
    max_diff_class: int
    max_diff_value: float
    attack_success: bool

    def __str__(self):
        status = "SUCCESS" if self.attack_success else "FAILED"
        return (f"Class {self.target_class_id}, prop {self.exclude_proportion}: {status} "
                f"(drop: {self.confidence_drop:.4f}, max_diff_class: {self.max_diff_class})")


class UnlearningAttackAnalyzer:
    """Analyzes unlearning attacks by comparing model confidence differences"""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.normalizer = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        self.probing_samples = self._load_probing_samples()

    def _load_probing_samples(self) -> Dict[int, List[torch.Tensor]]:
        """Load probing samples from configured directory"""
        samples_path = Path(self.config["attack"]["probe_samples_path"])
        samples_dict = {}

        for pkl_file in samples_path.glob("*.pkl"):
            class_id = int(pkl_file.name.replace('.pkl', '').split('_')[-1])

            with open(pkl_file, "rb") as f:
                samples = pickle.load(f)

            samples_dict[class_id] = samples
            print(f"Loaded {len(samples)} samples for class {class_id}")

        return samples_dict

    def _predict_sample(self, model: torch.nn.Module, sample: torch.Tensor, device: str) -> np.ndarray:
        """Get model prediction for a single sample"""
        with torch.no_grad():
            normalized = self.normalizer(sample.to(device))
            pred = model(normalized).softmax(dim=1).detach().cpu().numpy()
            return pred.flatten()

    def _evaluate_model_on_class(self, model_path: str, class_id: int) -> Tuple[float, np.ndarray]:
        """Evaluate model on samples from a specific class, return average target confidence and full prediction"""
        trainer = ModelTrainer(self.config)
        trainer.load_model(model_path)

        samples = self.probing_samples[class_id]
        predictions = []

        for sample in samples:
            pred = self._predict_sample(trainer.model, sample, trainer.device)
            predictions.append(pred)

        avg_prediction = np.array(predictions).mean(axis=0)
        target_confidence = avg_prediction[class_id]

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return target_confidence, avg_prediction

    def analyze_single_attack(self, attack_class_id: int, exclude_prop: float) -> AttackResult:
        """Analyze a single unlearning attack scenario"""
        # Get original model confidence
        original_conf, original_pred = self._evaluate_model_on_class(
            self.config["fine_tune"]["du_weights_load_path"],
            attack_class_id
        )

        # Get unlearned model confidence
        unlearned_model_name = f"finetuned_model_{attack_class_id}_{exclude_prop}.pth"
        unlearned_model_path = Path(self.config["fine_tune"]["dux_weights_load_dir"]) / unlearned_model_name

        if not unlearned_model_path.exists():
            raise FileNotFoundError(f"Unlearned model not found: {unlearned_model_path}")

        unlearned_conf, unlearned_pred = self._evaluate_model_on_class(
            str(unlearned_model_path),
            attack_class_id
        )

        # Calculate signed confidence differences (original - unlearned)
        conf_differences = original_pred - unlearned_pred
        abs_conf_differences = np.abs(conf_differences)
        max_diff_class = np.argmax(abs_conf_differences)
        max_diff_value = abs_conf_differences[max_diff_class]

        # Attack succeeds if target class has maximum absolute difference
        attack_success = (max_diff_class == attack_class_id)
        confidence_drop = original_conf - unlearned_conf

        return AttackResult(
            target_class_id=attack_class_id,
            exclude_proportion=exclude_prop,
            original_confidence=original_conf,
            unlearned_confidence=unlearned_conf,
            original_prediction=original_pred,
            unlearned_prediction=unlearned_pred,
            confidence_differences=conf_differences,
            confidence_drop=confidence_drop,
            max_diff_class=max_diff_class,
            max_diff_value=max_diff_value,
            attack_success=attack_success
        )

    def run_full_analysis(self) -> List[AttackResult]:
        """Run analysis on all configured attack scenarios"""
        print("Starting unlearning attack analysis...")
        print(f"Attack classes: {self.config['attack']['classes']}")
        print(f"Exclude proportions: {self.config['attack']['proportion']}")
        print()

        results = []

        for attack_class_id in self.config["attack"]["classes"]:
            print(f"Analyzing attacks on class {attack_class_id}:")

            for exclude_prop in self.config["attack"]["proportion"]:
                try:
                    result = self.analyze_single_attack(attack_class_id, exclude_prop)
                    results.append(result)
                    print(f"  {result}")

                except FileNotFoundError as e:
                    print(f"  Class {attack_class_id}, prop {exclude_prop}: SKIPPED ({e})")

        self._print_summary(results)
        return results

    @staticmethod
    def _print_summary(results: List[AttackResult]):
        """Print analysis summary"""
        print(f"\n{'=' * 60}")
        print("ATTACK ANALYSIS SUMMARY")
        print(f"{'=' * 60}")

        total_attacks = len(results)
        successful_attacks = sum(1 for r in results if r.attack_success)
        success_rate = (successful_attacks / total_attacks * 100) if total_attacks > 0 else 0

        print(f"Total attacks: {total_attacks}")
        print(f"Successful attacks: {successful_attacks}")
        print(f"Overall success rate: {success_rate:.1f}%")

        # Group by class for detailed analysis
        by_class = {}
        for result in results:
            if result.target_class_id not in by_class:
                by_class[result.target_class_id] = []
            by_class[result.target_class_id].append(result)

        print(f"\nPer-class analysis:")
        for class_id in sorted(by_class.keys()):
            class_results = by_class[class_id]
            class_successes = sum(1 for r in class_results if r.attack_success)
            class_rate = (class_successes / len(class_results) * 100)

            print(f"  Class {class_id}: {class_successes}/{len(class_results)} "
                  f"({class_rate:.1f}% success)")

            # Show confidence drops by proportion
            for result in sorted(class_results, key=lambda x: x.exclude_proportion):
                print(f"    Prop {result.exclude_proportion}: "
                      f"drop={result.confidence_drop:.4f}, "
                      f"max_diff={result.max_diff_value:.4f}")

    def create_visualizations(self, results: List[AttackResult], save_dir: str = "attack_visualizations"):
        """Create comprehensive visualizations of attack results"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Group results by exclude_proportion
        by_proportion = {}
        for result in results:
            prop = result.exclude_proportion
            if prop not in by_proportion:
                by_proportion[prop] = []
            by_proportion[prop].append(result)

        # Create a figure for each exclude_proportion
        for exclude_prop in sorted(by_proportion.keys()):
            self._create_proportion_figure(by_proportion[exclude_prop], exclude_prop, save_path)

        print(f"Visualizations saved to {save_path}")

    @staticmethod
    def _create_proportion_figure(results: List[AttackResult], exclude_prop: float, save_path: Path):
        """Create a figure with subplots for each target class for a given exclude_proportion"""
        # Group by target class and take the first result for each class (to handle duplicates)
        by_target_class = {}
        for result in results:
            if result.target_class_id not in by_target_class:
                by_target_class[result.target_class_id] = result

        # Sort by target class ID
        sorted_results = [by_target_class[class_id] for class_id in sorted(by_target_class.keys())]

        print(f"Creating figure for exclude_prop {exclude_prop} with {len(sorted_results)} unique target classes")

        # Create figure with subplots (2 rows, 5 columns for 10 classes)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        # Flatten axes for easier indexing
        axes_flat = axes.flatten()

        # CIFAR-10 class names for better labeling
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        for i, result in enumerate(sorted_results):
            if i >= 10:  # Safety check
                print(f"Warning: More than 10 target classes found, skipping extras")
                break

            ax = axes_flat[i]
            target_class = result.target_class_id
            conf_diffs = result.confidence_differences

            # Create bar chart
            x_pos = np.arange(10)
            bars = ax.bar(x_pos, conf_diffs, alpha=0.7)

            # Highlight the target class bar
            bars[target_class].set_color('red')
            bars[target_class].set_alpha(1.0)

            # Color other bars based on positive/negative
            for j, bar in enumerate(bars):
                if j != target_class:
                    if conf_diffs[j] > 0:
                        bar.set_color('lightblue')
                    else:
                        bar.set_color('lightcoral')

            y_min = min(conf_diffs) - 0.001
            y_max = max(conf_diffs) + 0.001
            ax.set_ylim(y_min, y_max)

            # Customize subplot
            ax.set_title(f'Target: Class {target_class} ({class_names[target_class]})', fontweight='bold')
            ax.set_xlabel('Class Index')
            ax.set_ylabel('Confidence Difference')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(i) for i in range(10)])
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Create legend
        red_patch = mpatches.Patch(color='red', label='Target Class')
        blue_patch = mpatches.Patch(color='lightblue', label='Positive Difference')
        coral_patch = mpatches.Patch(color='lightcoral', label='Negative Difference')
        fig.legend(handles=[red_patch, blue_patch, coral_patch],
                   loc='center', bbox_to_anchor=(0.5, 0.02), ncol=3)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        # Save figure
        filename = f'confidence_differences_exclude_{exclude_prop}.png'
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        plt.savefig(save_path / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"Created visualization: {filename}")

    def save_results(self, results: List[AttackResult], filepath: str):
        """Save attack results to file"""
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {filepath}")


def main():
    """Main function to run attack analysis"""
    config_path = "configs/attack_config.yaml"

    analyzer = UnlearningAttackAnalyzer(config_path)
    results = analyzer.run_full_analysis()

    analyzer.create_visualizations(results)


if __name__ == "__main__":
    main()