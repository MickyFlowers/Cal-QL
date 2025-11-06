from functools import partial

import flax
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState
from flax.traverse_util import path_aware_map
from jax.flatten_util import ravel_pytree
from ml_collections import ConfigDict

from utils.jax_utils import JaxRNG, collect_jax_metrics, next_rng, value_and_multi_grad, wrap_function_with_rng


class TrainStateWithBatchStats(TrainState):
    batch_stats: flax.core.FrozenDict = None


class Trainer(object):

    def __init__(self, config, policy, dataset):
        self.config = config
        self.policy = policy
        self.dataset = dataset
        self.image_size = policy.image_size
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim

        self._train_states = {}
        optimizer_cls = {
            "adam": optax.adam,
            "adamw": optax.adamw,
            "sgd": optax.sgd,
        }[self.config.optimizer_type]
        partition_optimizers = {
            "trainable": optimizer_cls(self.config.learning_rate),
            "frozen": optax.set_to_zero(),
        }

        variables = self.policy.init(
            next_rng(self.policy.rng_keys()),
            images=jnp.zeros((1, self.image_size, self.image_size, 3)),
            observations=jnp.zeros((1, self.observation_dim)),
        )
        policy_params = {"params": variables["params"]}
        param_partitions = path_aware_map(
            lambda path, param: "frozen" if "resnet" in path else "trainable", policy_params
        )
        tx = optax.multi_transform(partition_optimizers, param_partitions)
        policy_batch_stats = {"batch_stats": variables["batch_stats"]}
        self._train_states["policy"] = TrainStateWithBatchStats.create(
            params=policy_params,
            tx=tx,
            apply_fn=None,
            batch_stats=policy_batch_stats,
        )

        model_keys = ["policy"]

        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_step(self._train_states, next_rng(), batch)
        return metrics

    @partial(jax.jit, static_argnames=("self",))
    def _train_step(self, train_states, rng, batch):
        rng_generator = JaxRNG(rng)

        def loss_fn(train_params):
            observations = batch["observations"]
            actions = batch["actions"]
            images = batch["images"]
            loss_collections = {}
            new_batch_stats = {}

            @wrap_function_with_rng(rng_generator())
            def forward_policy(rng, *args, **kwargs):
                return self.policy.apply(*args, **kwargs, rngs=JaxRNG(rng)(self.policy.rng_keys()))

            variables = {**train_params["policy"], **train_states["policy"].batch_stats}
            log_probs, new_batch_stats["policy"] = forward_policy(
                variables,
                observations=observations,
                images=images,
                actions=actions,
                method=self.policy.log_prob,
                mutable=["batch_stats"],
            )

            policy_loss = -jnp.mean(log_probs)
            loss_collections["policy"] = policy_loss

            return tuple(loss_collections[key] for key in self.model_keys), (new_batch_stats, locals())

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, (new_batch_stats, aux_values)), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(
            train_params
        )
        policy_loss_gradient = jnp.linalg.norm(ravel_pytree(grads[self.model_keys.index("policy")]["policy"])[0])
        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[idx][key]).replace(batch_stats=new_batch_stats[key])
            for idx, key in enumerate(self.model_keys)
        }

        metrics = collect_jax_metrics(aux_values, ["policy_loss", "log_probs", "policy_loss_gradient"])
        metrics.update({"policy_loss_gradient": policy_loss_gradient})

        return new_train_states, metrics

    def set_train_states(self, train_states):
        self._train_states = train_states

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
