from src.training.models import ConvNet
from src.defense.pruner import ModelPruner
from src.utils import load_config

import os

def main():

    config_path = "configs/defence_config.yaml"
    config = load_config(config_path)

    pruning_kwargs = {
        'width': config['model']['width'],
        'num_classes': config['model']['num_classes'],
        'num_channels': config['model']['num_channels']
    }

    os.makedirs(config["prune"]["path_to_save_dir"], exist_ok=True)

    pruner = ModelPruner(ConvNet, pruning_kwargs)

    for prune_rate in config['prune']['rates']:

        _ = pruner.prune_folder(
            config["prune"]["path_to_model_dir"],
            config["prune"]["path_to_save_dir"],
            pruning_amount=prune_rate
            )
