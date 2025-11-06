import os
import pprint
import random
import tempfile
import time
import uuid
from copy import copy
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import numpy as np
import torch
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags

from .jax_utils import init_rng
from omegaconf import OmegaConf

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (list, tuple)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], torch.Tensor):
        return np.stack([b.cpu().numpy() for b in batch])
    else:
        return np.array(batch)
class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):
    def __init__(self, config):
        self.config = config

        if self.config.experiment_id is None:
            self.config.experiment_id = ""
        

        if self.config.prefix != '':
            self.config.project = '{}--{}'.format(self.config.prefix, self.config.project)

        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

    
        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        try:
            wandb_config = self.config.wandb_config
            os.environ['WANDB_API_KEY'] = wandb_config.wandb_api_key
            os.environ['WANDB_USER_EMAIL']  = wandb_config.wandb_user_email
            os.environ['WANDB_USERNAME'] = wandb_config.wandb_username
            os.environ["WANDB_MODE"] = wandb_config.wandb_mode
        except:
            print("No wandb_config.py file found...")
            print("Are you sure to turn wandb logging off? Press c to continue, q to quit")
            import pdb; pdb.set_trace()
            os.environ["WANDB_MODE"] = "run"
            self.config.online = False
        date = time.strftime("%Y%m%d-%H%M%S")
        self.config.experiment_id += date
        self.run = wandb.init(
            reinit=True,
            config=OmegaConf.to_container(self.config, resolve=True),
            project=self.config.project,
            dir=self.config.output_dir,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode='online' if self.config.online else 'offline',
            entity=self.config.entity,
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), 'wb') as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs

def flags_to_dict(flgs=None):
    if flgs is None:
        flgs = absl.flags.FLAGS
    return {name: flgs[name] for name in flgs}

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def print_flags(flags, flags_def):
    logging.info(
        'Running training with hyperparameters: \n{}'.format(
            pprint.pformat(
                ['{}: {}'.format(key, val) for key, val in get_user_flags(flags, flags_def).items()]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = '{}.{}'.format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=next_prefix))
        else:
            output[next_prefix] = val
    return output



def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }

