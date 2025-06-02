import pickle
import torch
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

    def _evaluate_model_on_class(self, model_path: str, class_id: int) -> float:
        """Evaluate model on samples from a specific class, return average target confidence"""
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

        # Calculate confidence differences across all classes
        conf_differences = np.abs(unlearned_pred - original_pred)
        max_diff_class = np.argmax(conf_differences)
        max_diff_value = conf_differences[max_diff_class]

        # Attack succeeds if target class has maximum difference
        attack_success = (max_diff_class == attack_class_id)
        confidence_drop = original_conf - unlearned_conf

        return AttackResult(
            target_class_id=attack_class_id,
            exclude_proportion=exclude_prop,
            original_confidence=original_conf,
            unlearned_confidence=unlearned_conf,
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

    def _print_summary(self, results: List[AttackResult]):
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

    # Optionally save results
    # analyzer.save_results(results, "attack_analysis_results.pkl")


if __name__ == "__main__":
    main()