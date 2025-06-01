from src.training.utils import set_random_seed
from src.training.trainer import ModelTrainer
from src.attack.zoo_attack import ZooAttack
from src.utils import load_config
from torchvision import transforms

from art.estimators.classification import BlackBoxClassifierNeuralNetwork

import torch
import numpy as np
import pickle
import os


def main():
    config_path = "configs/probe_samples_config.yaml"
    cifar10_mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
    cifar10_std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)

    normalizer = transforms.Normalize(cifar10_mean, cifar10_std)

    config = load_config(config_path)
    num_channels, img_size = config["data"]['channels'], config["data"]['img_size']
    set_random_seed(config["seed"])

    os.makedirs(config["fine_tune"]["output_folder"], exist_ok=True)

    trainer = ModelTrainer(config)
    trainer.load_model(config["fine_tune"]["weights_load_path"])


    def black_box_predict(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        else:
            x = x

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = x.to(trainer.device)

        with torch.no_grad():
            pred = trainer.model(normalizer(x)).softmax(dim=1).detach().cpu().numpy()

        return pred


    classifier = BlackBoxClassifierNeuralNetwork(predict_fn=black_box_predict,
                                                 channels_first=True,
                                                 input_shape=(num_channels, img_size, img_size),
                                                 nb_classes=config["model"]['num_classes'],
                                                 clip_values=(0, 1))

    datats = []

    th = 0.9995  # stopping threshold for confidence
    c = config["data"]["class_id"]
    batch_size = 1

    while len(datats) < 20:
        adv = np.clip(np.random.randn(batch_size, num_channels, img_size, img_size).astype(np.float32), 0, 1)
        attack = ZooAttack(
            classifier=classifier, targeted=False, max_iter=20,
            learning_rate=0.1, nb_parallel=1024, batch_size=batch_size,
            abort_early=True, verbose=True)
        attack.classid = c

        for trytime in range(10):
            print(len(datats), "******", c, "try", trytime)

            adv, loss = attack.generate(x=adv)
            adv_conf = black_box_predict(adv)[:, c]
            print("******", loss, adv_conf)
            if adv_conf >= th:
                datats.append(torch.tensor(adv))
                pickle.dump(
                    datats,
                    open(os.path.join(config["fine_tune"]["output_folder"], f"zoo_max_conf_noiseinit_max_cifar10_{c}.pkl"), "wb"))
                break
            elif adv_conf > 0.9:
                attack.learning_rate = 0.02

if __name__ == "__main__":
    main()