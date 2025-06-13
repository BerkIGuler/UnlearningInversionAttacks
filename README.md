# Reimplementation of Unlearning Inversion Attacks

## Requirements
Install the requirements with
```bash
pip install -r requirements.txt
```

## Training a Model on Public Dataset

The setup considered in this paper assumes an initial pretrained model for unlearning experiments. We pretrain the same convolutional network on CIFAR-10 with an 0.8/0.2 train/validation split. To start training with the default parameters in `configs/training_config.yaml`:

```bash
python -m scripts.train
```

This script trains the ConvNet model on $D_o$ and saves the model under `model_save_folder`.

## Obtaining the Original Model

The resulting model $f_\theta()$ is the model on which we want to perform unlearning. To start fine-tuning:

```bash
python -m scripts.finetune_du
```

This script fine-tunes the pretrained model using default parameters from `finetune_Du_config.yaml` and saves the resulting model under `model_save_folder`. The fine-tuning dataset $D_u$ contains the private user information.

## Obtaining Unlearned Models

Similar to how the original model is obtained, the unlearned models are generated via fine-tuning on the pretrained model. The approach here is to perform fine-tuning on the dataset $D_u - X$, where $X$ contains the unlearned data samples. This implementation corresponds to exact unlearning.

To start fine-tuning with the default parameters from `finetune_Dux_config.yaml`, run:

```bash
python -m scripts.finetune_dux
```

Configuration details:
- The `attack.classes` field in the config accepts a list of class indices. For each class index, a new fine-tuning is performed and the resulting model is saved.
- The `attack.proportion` field accepts a list of unlearning ratios $p$. For each $p$, the corresponding percentage of data from the current class is removed.

The resulting model is saved in `model_save_folder` with the name `finetuned_model_<class_idx>_<unlearning_ratio>.pth`.

## Generating Probing Samples for Label Inference Attack

The probing samples are generated using an implementation of the ZOO technique. To generate probing samples for class $i$:

```bash
python -m scripts.generate_probing_samples
```

The `data.class_id` field from `probe_samples_config.yaml` specifies the class of the generated probe samples.

## Measuring Attack Success

```bash
python -m scripts.measure_attack_accuracy
```

This script tests both the unlearned and original models on the same probing samples and reports the confidence change for each class.

Configuration details:
- The `attack.classes` field from `attack_config.yaml` accepts a list of class indices.
- The `attack.proportion` field from `attack_config.yaml` accepts a list of unlearning proportions.

For each pair of (attack class, proportion), the unlearned model `finetuned_model_<attack_class_idx>_<unlearning_ratio>.pth` is loaded and used to report the attack performance.

## Pruning as a Defense Mechanism

Creates a pruned model from each unlearned model located in the unlearned model folder.

```bash
python -m scripts.prune_models
```

Configuration details:
- The `prune.rates` field from `defense_config.yaml` specifies which pruning rates will be applied to each model.

The resulting pruned models are saved in a folder to be tested with the `measure_attack_accuracy` script later.