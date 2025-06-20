"""Modified from https://github.com/TASI-LAB/Unlearning-inversion-attacks/blob/main/gen_probing_samples.py"""
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

from scipy.ndimage import zoom
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import (
    compute_success,
    get_labels_np_array,
    check_and_transform_label_format,
)

import numpy as np


class ZooAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "binary_search_steps",
        "initial_const",
        "abort_early",
        "use_resize",
        "use_importance",
        "nb_parallel",
        "batch_size",
        "variable_h",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
            self,
            classifier: "CLASSIFIER_TYPE",
            learning_rate: float = 1e-2,
            targeted: bool = False,
            max_iter: int = 10,
            abort_early: bool = True,
            use_resize: bool = False,
            use_importance: bool = True,
            nb_parallel: int = 128,
            batch_size: int = 1,
            variable_h: float = 1e-4,
            verbose: bool = True,
    ):

        super().__init__(estimator=classifier)

        if len(classifier.input_shape) == 1:
            self.input_is_feature_vector = True
            if batch_size != 1:
                raise ValueError(
                    "The current implementation of Zeroth-Order Optimisation attack only supports "
                    "`batch_size=1` with feature vectors as input."
                )
        else:
            self.input_is_feature_vector = False

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.abort_early = abort_early
        self.use_resize = use_resize
        self.use_importance = use_importance
        self.nb_parallel = nb_parallel
        self.batch_size = batch_size
        self.variable_h = variable_h
        self.verbose = verbose

        self._init_size = 32
        if self.abort_early:
            self._early_stop_iters = self.max_iter // 10 if self.max_iter >= 10 else self.max_iter

        if self.input_is_feature_vector:
            self.use_resize = False
            self.use_importance = False
            if self.verbose:
                print(
                    "Disable resizing and importance sampling because feature vector input has been detected."
                )

        self._current_noise = np.zeros((batch_size,) + self.estimator.input_shape, dtype=ART_NUMPY_DTYPE)
        self._sample_prob = np.ones(self._current_noise.size, dtype=ART_NUMPY_DTYPE) / self._current_noise.size

        self.adam_mean: Optional[np.ndarray] = None
        self.adam_var: Optional[np.ndarray] = None
        self.adam_epochs: Optional[np.ndarray] = None

    def _loss(
            self, x_adv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the loss function values.

        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :param target: An array with the target class (one-hot encoded).
        :param c_weight: Weight of the loss term aiming for classification as target.
        :return: A tuple holding the current logits, `L_2` distortion and overall loss.
        """

        preds = self.estimator.predict(x_adv, batch_size=self.batch_size)

        loss = -(np.log(preds))[:, self.classid]

        return preds, loss

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: An array holding the adversarial examples.
        """
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        x_adv_list = []
        loweset_loss_list = []
        for batch_id in trange(nb_batches, desc="ZOO", disable=not self.verbose):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            res, loweset_loss = self._generate_batch(x_batch, )
            x_adv_list.append(res)
            loweset_loss_list.append(loweset_loss)
        x_adv = np.vstack(x_adv_list)

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            np.clip(x_adv, clip_min, clip_max, out=x_adv)

        if self.verbose:
            print(
                "Success rate of ZOO attack: %.2f%%",
                100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
            )

        return x_adv, loweset_loss_list

    def _generate_batch(self, x_batch: np.ndarray, ) -> np.ndarray:

        """
        Generate adversarial examples for a batch of inputs with a specific batch of constants.

        :param x_batch: A batch of original examples.
        :param y_batch: A batch of targets (0-1 hot).
        :param c_batch: A batch of constants.
        :return: A tuple of best elastic distances, best labels, best attacks.
        """

        prev_loss = 1e8 * np.ones(x_batch.shape[0])

        x_orig = x_batch
        self._reset_adam(np.prod(self.estimator.input_shape).item())
        if x_batch.shape == self._current_noise.shape:
            self._current_noise.fill(0)
        else:
            self._current_noise = np.zeros(x_batch.shape, dtype=ART_NUMPY_DTYPE)
        x_adv = x_orig.copy()

        loweset_loss = np.array([np.inf] * x_batch.shape[0])

        best_attack = x_orig.copy()

        tqdm_range = trange(0, self.max_iter, desc='Loss', leave=True)
        for iter_ in tqdm_range:

            x_adv = np.clip(self._optimizer(x_adv), 0, 1)
            preds, loss = self._loss(x_adv)

            update_idx = loss < loweset_loss
            loweset_loss[update_idx] = loss[update_idx]
            best_attack[update_idx] = x_adv[update_idx].copy()

            tqdm_range.set_description(f'Loss: {loss[0]:2.4f}, prev: {prev_loss[0]:2.4f}, best:{loweset_loss[0]:2.4f}')
            tqdm_range.refresh()

            if self.abort_early and iter_ % self._early_stop_iters == 0:
                if loss > 0.9999 * prev_loss:
                    break
                prev_loss = loss

        return best_attack, loweset_loss

    def _optimizer(self, x: np.ndarray) -> np.ndarray:
        coord_batch = np.repeat(self._current_noise, 2 * self.nb_parallel, axis=0)
        coord_batch = coord_batch.reshape(2 * self.nb_parallel * self._current_noise.shape[0], -1)

        if self.use_importance and np.unique(self._sample_prob).size != 1:
            try:
                indices = (
                        np.random.choice(
                            coord_batch.shape[-1] * x.shape[0],
                            self.nb_parallel * self._current_noise.shape[0],
                            replace=False,
                            p=self._sample_prob.flatten(),
                        )
                        % coord_batch.shape[-1]
                )
            except ValueError as error:  # pragma: no cover
                if "Cannot take a larger sample than population when 'replace=False'" in str(error):
                    raise ValueError(
                        "Too many samples are requested for the random indices. Try to reduce the number of parallel"
                        "coordinate updates `nb_parallel`."
                    ) from error
            except Exception as e:
                print(e)
        else:
            try:
                indices = (
                        np.random.choice(
                            coord_batch.shape[-1] * x.shape[0],
                            self.nb_parallel * self._current_noise.shape[0],
                            replace=False,
                        )
                        % coord_batch.shape[-1]
                )
            except ValueError as error:  # pragma: no cover
                if "Cannot take a larger sample than population when 'replace=False'" in str(error):
                    raise ValueError(
                        "Too many samples are requested for the random indices. Try to reduce the number of parallel"
                        "coordinate updates `nb_parallel`."
                    ) from error

                raise error
            except Exception as e:
                print(e)

        for i in range(self.nb_parallel * self._current_noise.shape[0]):
            coord_batch[2 * i, indices[i]] += self.variable_h
            coord_batch[2 * i + 1, indices[i]] -= self.variable_h

        expanded_x = np.repeat(x, 2 * self.nb_parallel, axis=0).reshape((-1,) + x.shape[1:])
        _, loss = self._loss(
            expanded_x + coord_batch.reshape(expanded_x.shape),
        )

        if self.adam_mean is not None and self.adam_var is not None and self.adam_epochs is not None:
            self._current_noise = self._optimizer_adam_coordinate(
                loss,
                indices,
                self.adam_mean,
                self.adam_var,
                self._current_noise,
                self.learning_rate,
                self.adam_epochs,
                True,
            )
        else:
            raise ValueError("Unexpected `None` in `adam_mean`, `adam_var` or `adam_epochs` detected.")

        if self.use_importance and self._current_noise.shape[2] > self._init_size:
            self._sample_prob = self._get_prob(self._current_noise).flatten()

        return x + self._current_noise

    def _optimizer_adam_coordinate(
            self,
            losses: np.ndarray,
            index: np.ndarray,
            mean: np.ndarray,
            var: np.ndarray,
            current_noise: np.ndarray,
            learning_rate: float,
            adam_epochs: np.ndarray,
            proj: bool,
    ) -> np.ndarray:
        """
        Implementation of the ADAM optimizer for coordinate descent.

        :param losses: Overall loss.
        :param index: Indices of the coordinates to update.
        :param mean: The mean of the gradient (first moment).
        :param var: The uncentered variance of the gradient (second moment).
        :param current_noise: Current noise.
        :param learning_rate: Learning rate for Adam optimizer.
        :param adam_epochs: Epochs to run the Adam optimizer.
        :param proj: Whether to project the noise to the L_p ball.
        :return: Updated noise for coordinate descent.
        """
        beta1, beta2 = 0.9, 0.999

        grads = np.array([(losses[i] - losses[i + 1]) / (2 * self.variable_h) for i in range(0, len(losses), 2)])

        mean[index] = beta1 * mean[index] + (1 - beta1) * grads
        var[index] = beta2 * var[index] + (1 - beta2) * grads ** 2

        corr = (np.sqrt(1 - np.power(beta2, adam_epochs[index]))) / (1 - np.power(beta1, adam_epochs[index]))
        orig_shape = current_noise.shape
        current_noise = current_noise.reshape(-1)
        current_noise[index] -= learning_rate * corr * mean[index] / (np.sqrt(var[index]) + 1e-8)
        adam_epochs[index] += 1

        if proj and hasattr(self.estimator, "clip_values") and self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            current_noise[index] = np.clip(current_noise[index], clip_min, clip_max)

        return current_noise.reshape(orig_shape)

    def _reset_adam(self, nb_vars: int, indices: Optional[np.ndarray] = None) -> None:
        if self.adam_mean is not None and self.adam_mean.size == nb_vars:
            if indices is None:
                self.adam_mean.fill(0)
                self.adam_var.fill(0)
                self.adam_epochs.fill(1)
            else:
                self.adam_mean[indices] = 0
                self.adam_var[indices] = 0
                self.adam_epochs[indices] = 1
        else:

            self.adam_mean = np.zeros(nb_vars, dtype=ART_NUMPY_DTYPE)
            self.adam_var = np.zeros(nb_vars, dtype=ART_NUMPY_DTYPE)
            self.adam_epochs = np.ones(nb_vars, dtype=int)

    def _get_prob(self, prev_noise: np.ndarray, double: bool = False) -> np.ndarray:
        dims = list(prev_noise.shape)
        channel_index = 1 if self.estimator.channels_first else 3

        if double:
            dims = [2 * size if i not in [0, channel_index] else size for i, size in enumerate(dims)]

        prob = np.empty(shape=dims, dtype=np.float32)
        image = np.abs(prev_noise)

        for channel in range(prev_noise.shape[channel_index]):
            if not self.estimator.channels_first:
                image_pool = self._max_pooling(image[:, :, :, channel], dims[1] // 8)
                if double:
                    prob[:, :, :, channel] = np.abs(zoom(image_pool, [1, 2, 2]))
                else:
                    prob[:, :, :, channel] = image_pool
            elif self.estimator.channels_first:
                image_pool = self._max_pooling(image[:, channel, :, :], dims[2] // 8)
                if double:
                    prob[:, channel, :, :] = np.abs(zoom(image_pool, [1, 2, 2]))
                else:
                    prob[:, channel, :, :] = image_pool

        prob += 0.01
        prob /= np.sum(prob)

        return prob

    @staticmethod
    def _max_pooling(image: np.ndarray, kernel_size: int) -> np.ndarray:
        img_pool = np.copy(image)
        for i in range(0, image.shape[1], kernel_size):
            for j in range(0, image.shape[2], kernel_size):
                img_pool[:, i: i + kernel_size, j: j + kernel_size] = np.max(
                    image[:, i: i + kernel_size, j: j + kernel_size],
                    axis=(1, 2),
                    keepdims=True,
                )

        return img_pool
