import pickle
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

from torchvision import transforms
from src.utils import load_config
from src.training.trainer import ModelTrainer

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
CIFAR10_STD = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)


@dataclass
class DefenseResult:
    """Represents the result of a single defense evaluation"""
    target_class_id: int
    exclude_proportion: float
    prune_rate: float
    original_confidence: float
    unlearned_confidence: float
    pruned_confidence: float
    original_prediction: np.ndarray
    unlearned_prediction: np.ndarray
    pruned_prediction: np.ndarray

    # Attack analysis on unlearned model
    unlearned_confidence_differences: np.ndarray
    unlearned_confidence_drop: float
    unlearned_max_diff_class: int
    unlearned_max_diff_value: float
    unlearned_attack_success: bool

    # Attack analysis on pruned model
    pruned_confidence_differences: np.ndarray
    pruned_confidence_drop: float
    pruned_max_diff_class: int
    pruned_max_diff_value: float
    pruned_attack_success: bool

    # Defense effectiveness
    defense_success: bool  # True if pruning prevented the attack
    defense_effectiveness: float  # Reduction in attack success (0-1)

    def __str__(self):
        unlearned_status = "SUCCESS" if self.unlearned_attack_success else "FAILED"
        pruned_status = "SUCCESS" if self.pruned_attack_success else "FAILED"
        defense_status = "SUCCESS" if self.defense_success else "FAILED"

        return (f"Class {self.target_class_id}, prop {self.exclude_proportion}, prune {self.prune_rate}: "
                f"Attack: Unlearned {unlearned_status} (pred_class={self.unlearned_max_diff_class}) -> "
                f"Pruned {pruned_status} (pred_class={self.pruned_max_diff_class}) | "
                f"Defense {defense_status} (effectiveness: {self.defense_effectiveness:.3f})")


class UnlearningDefenseAnalyzer:
    """Analyzes unlearning defense effectiveness using pruned models"""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.normalizer = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        self.probing_samples = self._load_probing_samples()

    def _load_probing_samples(self) -> Dict[int, List[torch.Tensor]]:
        """Load probing samples from configured directory"""
        samples_path = Path(self.config["defense"]["probe_samples_path"])
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

    def _evaluate_model_on_class(self, model_path: str, class_id: int, is_pruned: bool = False) -> Tuple[
        float, np.ndarray]:
        """Evaluate model on samples from a specific class, return average target confidence and full prediction"""
        trainer = ModelTrainer(self.config)

        if is_pruned:
            # Load pruned model with specific state_dict key and handle pruning masks
            checkpoint = torch.load(model_path, map_location=trainer.device)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Convert pruned state_dict to regular state_dict by removing masks
            clean_state_dict = {}
            for key, value in state_dict.items():
                if key.endswith('_orig'):
                    # This is the original weight before pruning
                    clean_key = key.replace('_orig', '')
                    mask_key = key.replace('_orig', '_mask')

                    if mask_key in state_dict:
                        # Apply the mask to get the final pruned weight
                        mask = state_dict[mask_key]
                        clean_state_dict[clean_key] = value * mask
                    else:
                        # No mask found, use original weight
                        clean_state_dict[clean_key] = value
                elif not key.endswith('_mask'):
                    # This is a regular parameter (not a mask)
                    clean_state_dict[key] = value

            trainer.model.load_state_dict(clean_state_dict)
            trainer.model.eval()
        else:
            # Load regular model
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

    def analyze_single_defense(self, attack_class_id: int, exclude_prop: float, prune_rate: float) -> DefenseResult:
        """Analyze a single defense scenario"""
        # Get original model confidence
        original_conf, original_pred = self._evaluate_model_on_class(
            self.config["fine_tune"]["du_weights_load_path"],
            attack_class_id,
            is_pruned=False
        )

        # Get unlearned model confidence
        unlearned_model_name = f"finetuned_model_{attack_class_id}_{exclude_prop}.pth"
        unlearned_model_path = Path(self.config["fine_tune"]["dux_weights_load_dir"]) / unlearned_model_name

        if not unlearned_model_path.exists():
            raise FileNotFoundError(f"Unlearned model not found: {unlearned_model_path}")

        unlearned_conf, unlearned_pred = self._evaluate_model_on_class(
            str(unlearned_model_path),
            attack_class_id,
            is_pruned=False
        )

        # Get pruned model confidence
        pruned_model_name = f"pruned_{prune_rate}_finetuned_model_{attack_class_id}_{exclude_prop}.pth"
        pruned_model_path = Path(self.config["defense"]["pruned_weights_load_dir"]) / pruned_model_name

        if not pruned_model_path.exists():
            raise FileNotFoundError(f"Pruned model not found: {pruned_model_path}")

        pruned_conf, pruned_pred = self._evaluate_model_on_class(
            str(pruned_model_path),
            attack_class_id,
            is_pruned=True
        )

        # Analyze attack on unlearned model
        unlearned_conf_diffs = original_pred - unlearned_pred
        unlearned_abs_conf_diffs = np.abs(unlearned_conf_diffs)
        unlearned_max_diff_class = np.argmax(unlearned_abs_conf_diffs)
        unlearned_max_diff_value = unlearned_abs_conf_diffs[unlearned_max_diff_class]

        # Attack succeeds if the predicted unlearned class (max confidence change) equals actual unlearned class
        unlearned_attack_success = (unlearned_max_diff_class == attack_class_id)
        unlearned_confidence_drop = original_conf - unlearned_conf

        # Analyze attack on pruned model
        pruned_conf_diffs = original_pred - pruned_pred
        pruned_abs_conf_diffs = np.abs(pruned_conf_diffs)
        pruned_max_diff_class = np.argmax(pruned_abs_conf_diffs)
        pruned_max_diff_value = pruned_abs_conf_diffs[pruned_max_diff_class]

        # Attack succeeds if the predicted unlearned class (max confidence change) equals actual unlearned class
        pruned_attack_success = (pruned_max_diff_class == attack_class_id)
        pruned_confidence_drop = original_conf - pruned_conf

        # Calculate defense effectiveness
        # Defense succeeds if:
        # 1. Attack was successful on unlearned model AND becomes unsuccessful on pruned model, OR
        # 2. Attack remains successful but confidence in correct prediction is reduced
        if unlearned_attack_success and not pruned_attack_success:
            # Case 1: Attack completely prevented
            defense_success = True
            defense_effectiveness = 1.0
        elif unlearned_attack_success and pruned_attack_success:
            # Case 2: Attack still succeeds, but check if confidence is reduced
            # Defense partially succeeds if confidence in the correct predicted class is reduced
            if pruned_max_diff_value < unlearned_max_diff_value:
                defense_success = True
                defense_effectiveness = (unlearned_max_diff_value - pruned_max_diff_value) / unlearned_max_diff_value
            else:
                defense_success = False
                defense_effectiveness = 0.0
        else:
            # Attack already failed on unlearned model or other cases
            defense_success = False
            defense_effectiveness = 0.0

        return DefenseResult(
            target_class_id=attack_class_id,
            exclude_proportion=exclude_prop,
            prune_rate=prune_rate,
            original_confidence=original_conf,
            unlearned_confidence=unlearned_conf,
            pruned_confidence=pruned_conf,
            original_prediction=original_pred,
            unlearned_prediction=unlearned_pred,
            pruned_prediction=pruned_pred,
            unlearned_confidence_differences=unlearned_conf_diffs,
            unlearned_confidence_drop=unlearned_confidence_drop,
            unlearned_max_diff_class=unlearned_max_diff_class,
            unlearned_max_diff_value=unlearned_max_diff_value,
            unlearned_attack_success=unlearned_attack_success,
            pruned_confidence_differences=pruned_conf_diffs,
            pruned_confidence_drop=pruned_confidence_drop,
            pruned_max_diff_class=pruned_max_diff_class,
            pruned_max_diff_value=pruned_max_diff_value,
            pruned_attack_success=pruned_attack_success,
            defense_success=defense_success,
            defense_effectiveness=defense_effectiveness
        )

    def run_full_analysis(self) -> List[DefenseResult]:
        """Run defense analysis on all configured scenarios"""
        print("Starting unlearning defense analysis...")
        print(f"Defense classes: {self.config['defense']['classes']}")
        print(f"Exclude proportions: {self.config['defense']['proportion']}")
        print(f"Prune rates: {self.config['defense']['prune_rates']}")
        print()

        results = []

        for attack_class_id in self.config["defense"]["classes"]:
            print(f"Analyzing defenses for class {attack_class_id}:")

            for exclude_prop in self.config["defense"]["proportion"]:
                for prune_rate in self.config["defense"]["prune_rates"]:
                    try:
                        result = self.analyze_single_defense(attack_class_id, exclude_prop, prune_rate)
                        results.append(result)
                        print(f"  {result}")

                    except FileNotFoundError as e:
                        print(f"  Class {attack_class_id}, prop {exclude_prop}, prune {prune_rate}: SKIPPED ({e})")

        self._print_summary(results)
        return results

    @staticmethod
    def _print_summary(results: List[DefenseResult]):
        """Print analysis summary"""
        print(f"\n{'=' * 80}")
        print("DEFENSE ANALYSIS SUMMARY")
        print(f"{'=' * 80}")

        total_scenarios = len(results)
        successful_defenses = sum(1 for r in results if r.defense_success)
        overall_defense_rate = (successful_defenses / total_scenarios * 100) if total_scenarios > 0 else 0

        # Calculate attack success rates
        unlearned_attacks = sum(1 for r in results if r.unlearned_attack_success)
        pruned_attacks = sum(1 for r in results if r.pruned_attack_success)

        unlearned_attack_rate = (unlearned_attacks / total_scenarios * 100) if total_scenarios > 0 else 0
        pruned_attack_rate = (pruned_attacks / total_scenarios * 100) if total_scenarios > 0 else 0

        print(f"Total defense scenarios: {total_scenarios}")
        print(f"Successful defenses: {successful_defenses}")
        print(f"Overall defense success rate: {overall_defense_rate:.1f}%")
        print(f"Unlearned model attack success rate: {unlearned_attack_rate:.1f}%")
        print(f"Pruned model attack success rate: {pruned_attack_rate:.1f}%")
        print(f"Attack rate reduction: {unlearned_attack_rate - pruned_attack_rate:.1f} percentage points")

        # Average defense effectiveness
        avg_effectiveness = np.mean([r.defense_effectiveness for r in results])
        print(f"Average defense effectiveness: {avg_effectiveness:.3f}")

        # Group by prune rate
        print(f"\nPer-prune-rate analysis:")
        by_prune_rate = defaultdict(list)
        for result in results:
            by_prune_rate[result.prune_rate].append(result)

        for prune_rate in sorted(by_prune_rate.keys()):
            rate_results = by_prune_rate[prune_rate]
            rate_defenses = sum(1 for r in rate_results if r.defense_success)
            rate_defense_rate = (rate_defenses / len(rate_results) * 100)

            rate_unlearned_attacks = sum(1 for r in rate_results if r.unlearned_attack_success)
            rate_pruned_attacks = sum(1 for r in rate_results if r.pruned_attack_success)

            rate_unlearned_rate = (rate_unlearned_attacks / len(rate_results) * 100)
            rate_pruned_rate = (rate_pruned_attacks / len(rate_results) * 100)

            rate_avg_effectiveness = np.mean([r.defense_effectiveness for r in rate_results])

            print(f"  Prune rate {prune_rate}: {rate_defenses}/{len(rate_results)} defenses "
                  f"({rate_defense_rate:.1f}%), attack success: {rate_unlearned_rate:.1f}% -> {rate_pruned_rate:.1f}%, "
                  f"avg effectiveness: {rate_avg_effectiveness:.3f}")

        # Group by exclude proportion
        print(f"\nPer-exclude-proportion analysis:")
        by_exclude_prop = defaultdict(list)
        for result in results:
            by_exclude_prop[result.exclude_proportion].append(result)

        for exclude_prop in sorted(by_exclude_prop.keys()):
            prop_results = by_exclude_prop[exclude_prop]
            prop_defenses = sum(1 for r in prop_results if r.defense_success)
            prop_defense_rate = (prop_defenses / len(prop_results) * 100)

            prop_unlearned_attacks = sum(1 for r in prop_results if r.unlearned_attack_success)
            prop_pruned_attacks = sum(1 for r in prop_results if r.pruned_attack_success)

            prop_unlearned_rate = (prop_unlearned_attacks / len(prop_results) * 100)
            prop_pruned_rate = (prop_pruned_attacks / len(prop_results) * 100)

            print(f"  Exclude prop {exclude_prop}: {prop_defenses}/{len(prop_results)} defenses "
                  f"({prop_defense_rate:.1f}%), attack success: {prop_unlearned_rate:.1f}% -> {prop_pruned_rate:.1f}%")

        # Group by class for detailed analysis
        print(f"\nPer-class analysis:")
        by_class = defaultdict(list)
        for result in results:
            by_class[result.target_class_id].append(result)

        for class_id in sorted(by_class.keys()):
            class_results = by_class[class_id]
            class_defenses = sum(1 for r in class_results if r.defense_success)
            class_defense_rate = (class_defenses / len(class_results) * 100)

            print(f"  Class {class_id}: {class_defenses}/{len(class_results)} "
                  f"({class_defense_rate:.1f}% defense success)")

    def create_visualizations(self, results: List[DefenseResult], save_dir: Optional[str] = None):
        """Create comprehensive visualizations of defense results"""
        if save_dir is None:
            save_dir = self.config["visualization"]["save_dir"]

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        # 1. Defense success rate heatmap by prune rate and exclude proportion
        self._create_defense_heatmap(results, save_path)

        # 2. Attack success rate comparison
        self._create_attack_comparison_plot(results, save_path)

        # 3. Defense effectiveness by prune rate
        self._create_effectiveness_plot(results, save_path)

        # 4. Confidence difference comparison plots
        self._create_confidence_comparison_plots(results, save_path)

        print(f"Defense visualizations saved to {save_path}")

    def _create_defense_heatmap(self, results: List[DefenseResult], save_path: Path):
        """Create heatmap showing defense success rates"""
        # Group results by prune rate and exclude proportion
        prune_rates = sorted(set(r.prune_rate for r in results))
        exclude_props = sorted(set(r.exclude_proportion for r in results))

        defense_matrix = np.zeros((len(prune_rates), len(exclude_props)))

        for i, prune_rate in enumerate(prune_rates):
            for j, exclude_prop in enumerate(exclude_props):
                relevant_results = [r for r in results
                                    if r.prune_rate == prune_rate and r.exclude_proportion == exclude_prop]
                if relevant_results:
                    defense_success_rate = sum(r.defense_success for r in relevant_results) / len(relevant_results)
                    defense_matrix[i, j] = defense_success_rate

        plt.figure(figsize=(10, 6))
        sns.heatmap(defense_matrix,
                    xticklabels=[f"{p:.3f}" for p in exclude_props],
                    yticklabels=[f"{p:.1f}" for p in prune_rates],
                    annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title('Defense Success Rate by Prune Rate and Exclude Proportion')
        plt.xlabel('Exclude Proportion')
        plt.ylabel('Prune Rate')
        plt.tight_layout()

        plt.savefig(save_path / 'defense_success_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'defense_success_heatmap.pdf', bbox_inches='tight')
        plt.close()

    def _create_attack_comparison_plot(self, results: List[DefenseResult], save_path: Path):
        """Create bar plot comparing attack success rates"""
        prune_rates = sorted(set(r.prune_rate for r in results))

        unlearned_rates = []
        pruned_rates = []

        for prune_rate in prune_rates:
            relevant_results = [r for r in results if r.prune_rate == prune_rate]
            unlearned_success = sum(r.unlearned_attack_success for r in relevant_results) / len(relevant_results)
            pruned_success = sum(r.pruned_attack_success for r in relevant_results) / len(relevant_results)

            unlearned_rates.append(unlearned_success)
            pruned_rates.append(pruned_success)

        x = np.arange(len(prune_rates))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, unlearned_rates, width, label='Unlearned Models', alpha=0.8)
        bars2 = ax.bar(x + width / 2, pruned_rates, width, label='Pruned Models', alpha=0.8)

        ax.set_xlabel('Prune Rate')
        ax.set_ylabel('Attack Success Rate')
        ax.set_title('Attack Success Rate: Unlearned vs Pruned Models')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p:.1f}" for p in prune_rates])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path / 'attack_success_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'attack_success_comparison.pdf', bbox_inches='tight')
        plt.close()

    def _create_effectiveness_plot(self, results: List[DefenseResult], save_path: Path):
        """Create box plot showing defense effectiveness by prune rate"""
        prune_rates = sorted(set(r.prune_rate for r in results))
        effectiveness_data = []

        for prune_rate in prune_rates:
            relevant_results = [r for r in results if r.prune_rate == prune_rate]
            effectiveness_values = [r.defense_effectiveness for r in relevant_results]
            effectiveness_data.append(effectiveness_values)

        plt.figure(figsize=(10, 6))
        box_plot = plt.boxplot(effectiveness_data, labels=[f"{p:.1f}" for p in prune_rates], patch_artist=True)

        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.xlabel('Prune Rate')
        plt.ylabel('Defense Effectiveness')
        plt.title('Defense Effectiveness Distribution by Prune Rate')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plt.savefig(save_path / 'defense_effectiveness_boxplot.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'defense_effectiveness_boxplot.pdf', bbox_inches='tight')
        plt.close()

    def _create_confidence_comparison_plots(self, results: List[DefenseResult], save_path: Path):
        """Create sample confidence difference comparison plots"""
        # Group by exclude proportion for visualization
        by_proportion = defaultdict(list)
        for result in results:
            by_proportion[result.exclude_proportion].append(result)

        # Create comparison plots for each exclude proportion
        for exclude_prop in sorted(by_proportion.keys())[:3]:  # Limit to first 3 for clarity
            self._create_single_confidence_comparison(by_proportion[exclude_prop], exclude_prop, save_path)

    def _create_single_confidence_comparison(self, results: List[DefenseResult], exclude_prop: float, save_path: Path):
        """Create confidence comparison plot for a specific exclude proportion"""
        # Take one example from each prune rate for a single target class
        target_class = results[0].target_class_id
        examples = {}

        for result in results:
            if result.target_class_id == target_class and result.prune_rate not in examples:
                examples[result.prune_rate] = result

        if len(examples) < 2:
            return

        prune_rates = sorted(examples.keys())
        n_rates = len(prune_rates)

        fig, axes = plt.subplots(2, n_rates, figsize=(4 * n_rates, 8))
        if n_rates == 1:
            axes = axes.reshape(-1, 1)

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        for i, prune_rate in enumerate(prune_rates):
            result = examples[prune_rate]

            # Plot unlearned model differences
            ax1 = axes[0, i]
            conf_diffs = result.unlearned_confidence_differences
            x_pos = np.arange(10)
            bars = ax1.bar(x_pos, conf_diffs, alpha=0.7)

            # Highlight target class
            bars[target_class].set_color('red')
            for j, bar in enumerate(bars):
                if j != target_class:
                    bar.set_color('lightblue' if conf_diffs[j] > 0 else 'lightcoral')

            ax1.set_title(f'Unlearned (Prune {prune_rate:.1f})')
            ax1.set_ylabel('Confidence Difference')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([str(i) for i in range(10)])
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Plot pruned model differences
            ax2 = axes[1, i]
            conf_diffs = result.pruned_confidence_differences
            bars = ax2.bar(x_pos, conf_diffs, alpha=0.7)

            # Highlight target class
            bars[target_class].set_color('red')
            for j, bar in enumerate(bars):
                if j != target_class:
                    bar.set_color('lightblue' if conf_diffs[j] > 0 else 'lightcoral')

            ax2.set_title(f'Pruned (Prune {prune_rate:.1f})')
            ax2.set_ylabel('Confidence Difference')
            ax2.set_xlabel('Class Index')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([str(i) for i in range(10)])
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.suptitle(f'Confidence Differences Comparison\nTarget Class {target_class} ({class_names[target_class]}), '
                     f'Exclude Proportion {exclude_prop}')
        plt.tight_layout()

        filename = f'confidence_comparison_class_{target_class}_prop_{exclude_prop}.png'
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        plt.savefig(save_path / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

    def save_results(self, results: List[DefenseResult], filepath: Optional[str] = None):
        """Save defense results to file"""
        if filepath is None:
            results_dir = Path(self.config["defense"]["results_save_folder"])
            results_dir.mkdir(exist_ok=True, parents=True)
            filepath = results_dir / "defense_analysis_results.pkl"

        with open(filepath, "wb") as f:
            pickle.dump(results, f)
        print(f"Defense results saved to {filepath}")

    def load_results(self, filepath: str) -> List[DefenseResult]:
        """Load defense results from file"""
        with open(filepath, "rb") as f:
            results = pickle.load(f)
        print(f"Defense results loaded from {filepath}")
        return results


def main():
    """Main function to run defense analysis"""
    config_path = "configs/defense_analysis_config.yaml"

    analyzer = UnlearningDefenseAnalyzer(config_path)
    results = analyzer.run_full_analysis()

    # Save results
    if analyzer.config["analysis"]["save_results"]:
        analyzer.save_results(results)

    # Create visualizations
    analyzer.create_visualizations(results)


if __name__ == "__main__":
    main()