import json
import os

import hydra
import numpy as np
import orbax
import torch
from flax.training import orbax_utils
from omegaconf import DictConfig, OmegaConf

from bc.trainer import Trainer
from data.datasets import BehaviorCloningDataset
from model.policy import ResNetPolicy
from utils.jax_utils import batch_to_jax
from utils.utils import Timer, WandBLogger, numpy_collate, prefix_metrics, set_random_seed
from viskit.logging import logger, setup_logger


@hydra.main(config_path="../config", config_name="bc", version_base=None)
def main(cfg: DictConfig):
    print("Hydra config:\n" + OmegaConf.to_yaml(cfg))
    variant = OmegaConf.to_container(cfg, resolve=True)

    wandb_logger = WandBLogger(config=cfg.logging)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=cfg.seed,
        base_log_dir=cfg.logging.output_dir,
        include_exp_prefix_sub_dir=False,
    )

    assert (
        cfg.data_root is not None and cfg.data_root != "data_root"
    ), "Please provide real data_root (override +data_root=...)"
    dataset = BehaviorCloningDataset(data_root=cfg.data_root, image_size=(cfg.image_size, cfg.image_size))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        collate_fn=numpy_collate,
        pin_memory=False,
    )
    print(f"Loaded BC dataset from {cfg.data_root}, size={len(dataset)}")

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

    trainer = Trainer(
        config=cfg.trainer,
        policy=policy,
        dataset=dataset,
    )
    restore_epoch = 0
    if cfg.save_model or cfg.load_ckpt_dir:
        if cfg.load_ckpt_dir:
            ckpt_dir = cfg.load_ckpt_dir
        else:
            ckpt_dir = os.path.join(cfg.save_ckpt_dir, wandb_logger.config.project, wandb_logger.config.experiment_id)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=3, create=True, best_fn=lambda x: x, best_mode=cfg.best_mode
        )
        checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer, options)
        save_args = orbax_utils.save_args_from_target(trainer.train_states)
        if cfg.load_ckpt_dir:
            if cfg.restore_steps == "latest":
                restore_epoch = checkpoint_manager.latest_step()
            elif cfg.resotre_steps is not None:
                restore_epoch = cfg.restore_steps
            else:
                raise ValueError("Please provide restore_steps as 'latest' or an integer step number.")
            restored_state = checkpoint_manager.restore(restore_epoch, item=trainer.train_states)
            trainer.set_train_states(restored_state)
            print(f"Restored checkpoint from {ckpt_dir} at epoch {restore_epoch}")

    total_grad_steps = 0

    viskit_metrics = {}
    first_save = False
    train_metrics = None
    epoch = restore_epoch
    while epoch < cfg.n_train_epochs + restore_epoch:
        metrics = {}

        with Timer() as train_timer:
            for batch in dataloader:
                batch = batch_to_jax(batch)
                train_metrics = trainer.train(batch)
                train_metrics = prefix_metrics(train_metrics, "bc")
                total_grad_steps += 1

        metrics["grad_steps"] = total_grad_steps
        metrics["epoch"] = epoch
        metrics["train_time"] = train_timer()
        if train_metrics:
            metrics.update(train_metrics)

        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        if cfg.save_model and epoch > 0 and ((epoch + 1) % cfg.save_model_interval == 0):
            checkpoint_manager.save(
                epoch + 1,
                trainer.train_states,
                save_kwargs={"save_args": save_args},
                metrics=metrics[cfg.best_metrics_key].item(),
            )
            print(f"Saved checkpoint at epoch {epoch + 1} -> {ckpt_dir}")
            if not first_save:
                json.dump({"variant": variant}, open(os.path.join(ckpt_dir, "config.json"), "w"), indent=2)
                first_save = True
        epoch += 1

    print(f"Finished BC training at epoch {epoch}")


if __name__ == "__main__":
    main()
