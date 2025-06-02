import pickle
import sys

import torch
import numpy as np
import os

from torchvision import transforms

from src.utils import load_config
from src.training.trainer import ModelTrainer

cifar10_mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
cifar10_std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)


def predict(model, x):
    """
    Returns the probability dist. p(x) where p(x = i) is x being a member of class i

    Args:
        model (torch.nn.Module): PyTorch model
        x (torch.tensor): input tensor

    Returns:
        pred (np.array): array of probabilities summing to 1
    """
    normalizer = transforms.Normalize(cifar10_mean, cifar10_std)

    with torch.no_grad():
        pred = model(normalizer(x)).softmax(dim=1).detach().cpu().numpy()

    return pred


def load_probing_samples(path_to_probe_samples_folder):
    """
    Returns a dict containing (class_id, samples) key-value pair for each .pkl file

    Args:
        path_to_probe_samples_folder (str): directory path containing .pkl files

    Returns:
        probe_samples_dict (dict): dictionary containing pairs of (class_id, samples)
            where class_id is an int and samples is a list of torch.Tensor's
    """
    probe_samples_files = [item for item in os.listdir(path_to_probe_samples_folder) if item.endswith(".pkl")]
    probe_samples_dict = {}

    for file in probe_samples_files:
        underscore_index = file.rfind('_')
        dot_index = file.rfind('.')

        if underscore_index == -1 or dot_index == -1:
            raise ValueError(f"file {file} does not follow expected naming convention")

        try:
            class_id = int(file[underscore_index + 1:dot_index])
        except ValueError:
            raise ValueError(f"file {file} does not contain a valid class id")

        file_path = os.path.join(path_to_probe_samples_folder, file)
        with open(file_path, "rb") as f:
            samples = pickle.load(f)
        probe_samples_dict[class_id] = samples

    return probe_samples_dict


def evaluate_avg_confidence_per_class(trainer, probing_samples):
    """
        Returns a dict of pairs of (target_class_id, average_conf_per_class)

    Args:
        trainer (ModelTrainer): A trainer class containing model and device as an attribute
        probing_samples (dict): A dictionary containing pairs of (target_class_id, samples)
            where target_class_id is an int and samples is a list of torch.Tensor's

    Returns:
        conf_per_class_dict: A dict of average confidence per classes
            each key is the int target_class_id of target attack class
            each value is numpy array of probabilities (averaged over all samples)
    """
    trainer.model.eval()
    conf_per_class_dict = {}

    for class_id in sorted(probing_samples.keys()):
        samples = probing_samples[class_id]
        predictions = np.array([predict(trainer.model, sample.to(trainer.device)) for sample in samples])
        average_probs = predictions.mean(axis=0)
        conf_per_class_dict[class_id] = average_probs

    return conf_per_class_dict


def report_attack_results(du_confidence_per_class, all_dux_confidences_per_class):
    """
    Args:
        du_confidence_per_class (dict): A dict of pairs of target_class_id, average confidence array
            du is for the original private model
        all_dux_confidences_per_class (dict): A dict of pairs of (target_class_id, dict of pairs of (exclude_proportion, dict of pairs of (class_id, average confidence array)))
            dux is for the unlearned private model where exclude_proportion portion of target_class_id is unlearned.
    Returns:
         conf_difference_dict (dict): A dict of pairs of (target_class_id, dict of pairs of (exclude_proportion, abs. difference confidence array))
            For each pair of (target_class_id, exclude_proportion), the class with highest abs. difference is the predicted attack class
            The Ground truth is target_class_id
    """
    conf_difference_dict = {}

    sorted_class_ids = sorted(du_confidence_per_class.keys())
    du_confidence_array_per_class = np.array([du_confidence_per_class[class_id] for class_id in sorted_class_ids])

    for attack_class_id in all_dux_confidences_per_class:

        if attack_class_id not in conf_difference_dict:
            conf_difference_dict[attack_class_id] = {}
        else:
            raise ValueError("Something is wrong with reporting!")

        for exclude_prop in all_dux_confidences_per_class[attack_class_id]:
            dux_confidence_per_class_dict = all_dux_confidences_per_class[attack_class_id][exclude_prop]

            dux_confidence_array_per_class = np.array(
                [dux_confidence_per_class_dict[class_id] for class_id in sorted_class_ids])

            conf_difference_dict[attack_class_id][exclude_prop] = np.abs(
                dux_confidence_array_per_class - du_confidence_array_per_class)

    return conf_difference_dict


def main():
    config_path = "configs/attack_config.yaml"
    config = load_config(config_path)
    probing_samples_dict = load_probing_samples(config["attack"]["probe_samples_path"])

    # Load and evaluate original model (du)
    du_trainer = ModelTrainer(config)
    du_trainer.load_model(config["fine_tune"]["du_weights_load_path"])
    du_conf_per_class_dict = evaluate_avg_confidence_per_class(du_trainer, probing_samples_dict)

    del du_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dux_trainer = ModelTrainer(config)
    all_dux_confidences = {}

    for attack_class_id in config["attack"]["classes"]:

        if attack_class_id not in all_dux_confidences:
            all_dux_confidences[attack_class_id] = {}
        else:
            raise ValueError("Something is wrong here!")

        for exclude_prop in config["attack"]["proportion"]:
            dux_model_name = f"finetuned_model_{attack_class_id}_{exclude_prop}.pth"
            dux_model_path = os.path.join(config["fine_tune"]["dux_weights_load_dir"], dux_model_name)

            if not os.path.exists(dux_model_path):
                raise FileNotFoundError(f"Model file not found: {dux_model_path}")

            dux_trainer.load_model(dux_model_path)
            dux_conf_per_class_dict = evaluate_avg_confidence_per_class(dux_trainer, probing_samples_dict)
            all_dux_confidences[attack_class_id][exclude_prop] = dux_conf_per_class_dict

    del dux_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    conf_difference_dict = report_attack_results(du_conf_per_class_dict, all_dux_confidences)
    print("Original model confidences per class:")
    print(du_conf_per_class_dict)
    print("\nConfidence differences (attack analysis):")
    print(conf_difference_dict)


if __name__ == "__main__":
    main()