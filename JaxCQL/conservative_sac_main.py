
import gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from JaxCQL.conservative_sac import ConservativeSAC
from JaxCQL.model import (FullyConnectedQFunction, SamplerPolicy,
                          TanhGaussianPolicy)
from JaxCQL.replay_buffer import (concatenate_batches,
                                  get_d4rl_dataset_with_mc_calculation,
                                  get_hand_dataset_with_mc_calculation,
                                  subsample_batch)
from JaxCQL.sampler import TrajSampler
from utils.jax_utils import batch_to_jax
from utils.utils import Timer, WandBLogger, prefix_metrics, set_random_seed
from viskit.logging import logger, setup_logger

from .replay_buffer import ReplayBuffer


@hydra.main(config_path="../config", config_name="sac", version_base=None)
def main(cfg: DictConfig):
    print("Hydra SAC config:\n" + OmegaConf.to_yaml(cfg))
    variant = OmegaConf.to_container(cfg, resolve=True)

    wandb_logger = WandBLogger(config=cfg.logging)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=cfg.seed,
        base_log_dir=cfg.logging.output_dir,
        include_exp_prefix_sub_dir=False,
    )

    # Load dataset depending on environment type
    if cfg.env in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0"]:
        import mj_envs  # noqa: F401
        dataset = get_hand_dataset_with_mc_calculation(
            cfg.env,
            gamma=cfg.cql.discount,
            reward_scale=cfg.reward_scale,
            reward_bias=cfg.reward_bias,
            clip_action=cfg.clip_action,
        )
        use_goal = True
    else:
        dataset = get_d4rl_dataset_with_mc_calculation(
            cfg.env,
            cfg.reward_scale,
            cfg.reward_bias,
            cfg.clip_action,
            gamma=cfg.cql.discount,
        )
        use_goal = False
    print("Loaded dataset for", cfg.env)
    assert dataset["next_observations"].shape == dataset["observations"].shape

    # Samplers & replay buffer
    set_random_seed(cfg.seed)
    eval_sampler = TrajSampler(gym.make(cfg.env).unwrapped, use_goal, gamma=cfg.cql.discount)
    train_sampler = TrajSampler(
        gym.make(cfg.env).unwrapped,
        use_goal,
        use_mc=True,
        gamma=cfg.cql.discount,
        reward_scale=cfg.reward_scale,
        reward_bias=cfg.reward_bias,
    )
    replay_buffer = ReplayBuffer(cfg.replay_buffer_size)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    # Policy & Q networks
    policy = TanhGaussianPolicy(
        observation_dim,
        action_dim,
        cfg.policy_arch,
        cfg.orthogonal_init,
        cfg.policy_log_std_multiplier,
        cfg.policy_log_std_offset,
    )
    qf = FullyConnectedQFunction(
        observation_dim,
        action_dim,
        cfg.qf_arch,
        cfg.orthogonal_init,
    )
    print("Initialized policy and Q networks")

    # Adjust target entropy if using automatic tuning
    cql_config = OmegaConf.to_container(cfg.cql, resolve=True)
    if cql_config["target_entropy"] >= 0.0:
        cql_config["target_entropy"] = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(cql_config, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

    # Training state variables
    viskit_metrics = {}
    n_train_step_per_epoch = cfg.n_train_step_per_epoch_offline
    cql_min_q_weight = cfg.cql_min_q_weight
    enable_calql = cfg.enable_calql
    use_cql = cfg.use_cql
    mixing_ratio = cfg.mixing_ratio
    total_grad_steps = 0
    is_online = False
    online_eval_counter = -1
    epoch = 0
    train_metrics = None
    expl_metrics = None

    while True:
        metrics = {"epoch": epoch}

        # Switch to online phase
        if epoch == cfg.n_pretrain_epochs:
            is_online = True
            if cfg.cql_min_q_weight_online >= 0:
                print(f"Adjusting CQL min Q weight to {cfg.cql_min_q_weight_online}")
                cql_min_q_weight = cfg.cql_min_q_weight_online
            if (not cfg.online_use_cql) and use_cql:
                print("Turning off CQL for online phase (pure SAC)")
                use_cql = False
                if sac.config.cql_lagrange:
                    model_keys = list(sac.model_keys)
                    if 'log_alpha_prime' in model_keys:
                        model_keys.remove('log_alpha_prime')
                    sac._model_keys = tuple(model_keys)

        # Evaluation conditions
        do_eval = (
            epoch == 0
            or (not is_online and epoch % cfg.offline_eval_every_n_epoch == 0)
            or (epoch == cfg.n_pretrain_epochs)
            or (
                is_online
                and replay_buffer.total_steps // cfg.online_eval_every_n_env_steps
                > online_eval_counter
            )
            or (replay_buffer.total_steps >= cfg.max_online_env_steps)
        )

        with Timer() as eval_timer:
            if do_eval:
                print(f"Evaluating epoch {epoch}")
                trajs = eval_sampler.sample(
                    sampler_policy.update_params(sac.train_params['policy']),
                    cfg.eval_n_trajs,
                    deterministic=True,
                )
                metrics['evaluation/average_return'] = np.mean([
                    np.sum(t['rewards']) for t in trajs
                ])
                metrics['evaluation/average_traj_length'] = np.mean([
                    len(t['rewards']) for t in trajs
                ])
                if use_goal:
                    metrics['evaluation/goal_achieved_rate'] = np.mean([
                        1 in t['goal_achieved'] for t in trajs
                    ])
                else:
                    metrics['evaluation/average_normalized_return'] = np.mean([
                        eval_sampler.env.get_normalized_score(np.sum(t['rewards']))
                        for t in trajs
                    ])

                if is_online:
                    online_eval_counter = (
                        replay_buffer.total_steps // cfg.online_eval_every_n_env_steps
                    )

                if cfg.save_model:
                    save_data = {
                        'sac': sac,
                        'variant': variant,
                        'epoch': epoch,
                    }
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['grad_steps'] = total_grad_steps
        if is_online:
            metrics['env_steps'] = replay_buffer.total_steps
        metrics['online_rollout_time'] = 0.0  # filled later
        metrics['train_time'] = 0.0
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = eval_timer()
        if cfg.n_pretrain_epochs >= 0:
            metrics['mixing_ratio'] = mixing_ratio
        if train_metrics:
            metrics.update(train_metrics)
        if expl_metrics:
            metrics.update(expl_metrics)

        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        if replay_buffer.total_steps >= cfg.max_online_env_steps:
            print("Reached max online env steps. Finishing training.")
            break

        with Timer() as online_rollout_timer:
            if is_online:
                print("Collecting online trajectories:", cfg.n_online_traj_per_epoch)
                trajs = train_sampler.sample(
                    sampler_policy.update_params(sac.train_params['policy']),
                    n_trajs=cfg.n_online_traj_per_epoch,
                    deterministic=False,
                    replay_buffer=replay_buffer,
                )
                expl_metrics = {
                    'exploration/average_return': np.mean([
                        np.sum(t['rewards']) for t in trajs
                    ]),
                    'exploration/average_traj_length': np.mean([
                        len(t['rewards']) for t in trajs
                    ]),
                }
                if use_goal:
                    expl_metrics['exploration/goal_achieved_rate'] = np.mean([
                        1 in t['goal_achieved'] for t in trajs
                    ])
                metrics['online_rollout_time'] = online_rollout_timer()

        if epoch == 0:
            print("JIT compiling training step...")

        with Timer() as train_timer:
            if (
                cfg.n_pretrain_epochs >= 0
                and epoch >= cfg.n_pretrain_epochs
                and cfg.online_utd_ratio > 0
            ):
                n_train_step_per_epoch = (
                    np.sum([len(t['rewards']) for t in trajs])
                    * cfg.online_utd_ratio
                )
            if cfg.n_pretrain_epochs >= 0:
                if cfg.mixing_ratio >= 0:
                    mixing_ratio = cfg.mixing_ratio
                else:
                    mixing_ratio = dataset['rewards'].shape[0] / (
                        dataset['rewards'].shape[0] + replay_buffer.total_steps
                    )
                batch_size_offline = int(cfg.batch_size * mixing_ratio)
                batch_size_online = cfg.batch_size - batch_size_offline
            for _ in range(n_train_step_per_epoch):
                if is_online:
                    offline_batch = subsample_batch(dataset, batch_size_offline)
                    online_batch = replay_buffer.sample(batch_size_online)
                    batch = concatenate_batches([offline_batch, online_batch])
                    batch = batch_to_jax(batch)
                else:
                    batch = batch_to_jax(subsample_batch(dataset, cfg.batch_size))
                train_metrics = prefix_metrics(
                    sac.train(
                        batch,
                        use_cql=use_cql,
                        cql_min_q_weight=cql_min_q_weight,
                        enable_calql=enable_calql,
                    ),
                    'sac',
                )
            total_grad_steps += n_train_step_per_epoch
            metrics['train_time'] = train_timer()
            metrics['epoch_time'] = metrics['train_time'] + metrics['eval_time']

        epoch += 1

    print("Finished SAC training")

if __name__ == '__main__':
    main()
