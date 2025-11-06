import os
import time
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import orbax
from flax.training import orbax_utils
from omegaconf import DictConfig, OmegaConf

from model.policy import ResNetPolicy
from utils.jax_utils import JaxRNG, next_rng
from utils.utils import set_random_seed
from viskit.logging import logger, setup_logger


class Sampler(object):
    def __init__(self, policy, params):
        self.policy = policy
        self.params = params

    def update_params(self, new_params):
        self.params = new_params
        return self

    @partial(jax.jit, static_argnames=("self"))
    def sample(self, rng, observations, images):
        output = self.policy.apply(
            self.params, images=images, observations=observations, rngs=JaxRNG(rng)(self.policy.rng_keys())
        )
        return jax.device_get(output)


@hydra.main(config_path="../config", config_name="rollout", version_base=None)
def main(args: DictConfig):
    print(args.ckpt_path)
    cfg = OmegaConf.load(os.path.join(args.ckpt_path, "config.json")).variant
    print("Hydra config:\n" + OmegaConf.to_yaml(cfg))
    variant = OmegaConf.to_container(cfg, resolve=True)

    setup_logger(
        variant=variant,
        exp_id=cfg.logging.experiment_id,
        seed=cfg.seed,
        base_log_dir=cfg.logging.output_dir,
        include_exp_prefix_sub_dir=False,
    )

    set_random_seed(cfg.seed)
    policy = ResNetPolicy(
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        image_size=cfg.image_size,
        obs_head_arch=cfg.policy_obs_head_arch,
        hidden_dim=cfg.policy_hidden_dim,
        out_head_arch=cfg.policy_out_head_arch,
        orthogonal_init=cfg.orthogonal_init,
        log_std_multiplier=cfg.policy_log_std_multiplier,
        log_std_offset=cfg.policy_log_std_offset,
    )
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(args.ckpt_path, orbax_checkpointer, options)
    if args.restore_step == "latest":
        restored_step = checkpoint_manager.latest_step()
    else:
        restored_step = args.restore_step
    restored_state = checkpoint_manager.restore(restored_step)
    print(f"Restored checkpoint from {args.ckpt_path} at step {restored_step}")
    sampler = Sampler(policy, restored_state["policy"]["params"])
    while True:
        output = sampler.sample(
            next_rng(),
            jnp.zeros((1, cfg.observation_dim)),
            jnp.zeros((1, cfg.image_size, cfg.image_size, 3)),
        )


if __name__ == "__main__":
    main()
