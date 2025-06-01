import pickle
import torch
import numpy as np
from torchvision import transforms

from src.utils import load_config
from src.training.trainer import ModelTrainer

cifar10_mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
cifar10_std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)

def predict(model, x):
    normalizer = transforms.Normalize(cifar10_mean, cifar10_std)

    with torch.no_grad():
        pred = model(normalizer(x)).softmax(dim=1).detach().cpu().numpy()

    return pred

def main():
    probe_samples_path = "outputs/probing_samples/zoo_max_conf_noiseinit_max_cifar10_0.pkl"
    model_weights_path = "outputs/private/finetuned_model.pth"
    config_path = "configs/probe_samples_config.yaml"

    config = load_config(config_path)
    samples = pickle.load(open(probe_samples_path, "rb"))

    trainer = ModelTrainer(config)
    trainer.load_model(model_weights_path)

    predictions = np.array([predict(trainer.model, sample.to(trainer.device)) for sample in samples])

    print(len(predictions))
    print(type(predictions))
    print(predictions[0])
    print(predictions[:, :, 0])


if __name__ == "__main__":
    main()