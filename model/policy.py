import distrax
import jax
import jax.numpy as jnp
import ml_collections
import requests
from flax import linen as nn
from PIL import Image
from transformers import AutoImageProcessor, FlaxResNetModel

from utils.jax_utils import JaxRNG, extend_and_repeat, next_rng
from utils.utils import set_random_seed


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = "256-256"
    orthogonal_init: bool = False

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split("-")]
        for h in hidden_sizes:
            if self.orthogonal_init:
                x = nn.Dense(
                    h, kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)), bias_init=jax.nn.initializers.zeros
                )(x)
            else:
                x = nn.Dense(h)(x)
            x = nn.relu(x)

        if self.orthogonal_init:
            output = nn.Dense(
                self.output_dim, kernel_init=jax.nn.initializers.orthogonal(1e-2), bias_init=jax.nn.initializers.zeros
            )(x)
        else:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.variance_scaling(1e-2, "fan_in", "uniform"),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        return output


class ResNetPolicy(nn.Module):
    observation_dim: int
    action_dim: int
    image_size: int = 64
    orthogonal_init: bool = False
    obs_head_arch: str = "256-256"
    hidden_dim: int = 256
    out_head_arch: str = "256-256"
    log_std_multiplier: float = 1.0
    log_std_offset: float = -1.0
    train: bool = True

    def setup(self):
        self.resnet = FlaxResNetModel.from_pretrained("microsoft/resnet-50")
        self.resnet_params = self.param("resnet", lambda rng: self.resnet.params["params"])
        self.resnet_batch_stats = self.variable("batch_stats", "resnet", lambda: self.resnet.params["batch_stats"])

        self.obs_head = FullyConnectedNetwork(
            output_dim=self.hidden_dim, arch=self.obs_head_arch, orthogonal_init=self.orthogonal_init
        )
        self.out_head = FullyConnectedNetwork(
            output_dim=self.action_dim * 2, arch=self.out_head_arch, orthogonal_init=self.orthogonal_init
        )
        # for resnet 50, the output feature map is [batch, 7, 7, 2048]
        self.spatial_pooler = nn.Conv(2048, kernel_size=(7, 7), strides=(1, 1), padding="VALID")
        if self.orthogonal_init:
            self.bottleneck = nn.Dense(
                self.hidden_dim,
                kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=jax.nn.initializers.zeros,
            )
        else:
            self.bottleneck = nn.Dense(self.hidden_dim)
        self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
        self.log_std_offset_module = Scalar(self.log_std_offset)
        print("Initialized ResNetPolicy with image size:", self.image_size)

    def log_prob(self, observations, images, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
            images = extend_and_repeat(images, 1, actions.shape[1])
        images = jnp.transpose(images, (0, 3, 1, 2))  # bchw
        input_params = {"params": self.resnet_params, "batch_stats": self.resnet_batch_stats.value}
        out = self.resnet(images, params=input_params, train=self.train)  # bchw
        image_features = out[0].last_hidden_state  # bchw
        self.resnet_batch_stats.value = out[1]["batch_stats"]
        image_features = jnp.transpose(image_features, (0, 2, 3, 1))  # bhwc
        image_features = self.spatial_pooler(image_features)
        image_features = jnp.squeeze(image_features, axis=(-2, -3))
        image_features = self.bottleneck(image_features)
        obs_features = self.obs_head(observations)
        features = jnp.concatenate([image_features, obs_features], axis=-1)
        out = self.out_head(features)
        mean, log_std = jnp.split(out, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -5.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)), distrax.Block(distrax.Tanh(), ndims=1)
        )
        return action_distribution.log_prob(actions)

    def __call__(self, observations, images, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
            images = extend_and_repeat(images, 1, repeat)
        images = jnp.transpose(images, (0, 3, 1, 2))  # bchw
        input_params = {"params": self.resnet_params, "batch_stats": self.resnet_batch_stats.value}
        out = self.resnet(images, params=input_params, train=self.train)  # bchw
        image_features = out[0].last_hidden_state  # bchw
        self.resnet_batch_stats.value = out[1]["batch_stats"]
        image_features = jnp.transpose(image_features, (0, 2, 3, 1))  # bhwc
        image_features = self.spatial_pooler(image_features)
        image_features = jnp.squeeze(image_features, axis=(-2, -3))
        image_features = self.bottleneck(image_features)
        obs_features = self.obs_head(observations)
        features = jnp.concatenate([image_features, obs_features], axis=-1)
        out = self.out_head(features)
        mean, log_std = jnp.split(out, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -5.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)), distrax.Block(distrax.Tanh(), ndims=1)
        )
        if deterministic:
            samples = jnp.tanh(mean)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(seed=self.make_rng("noise"))
        return samples, log_prob

    @nn.nowrap
    def rng_keys(self):
        return ["params", "noise"]

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict(
            {
                "image_size": 64,
                "orthogonal_init": False,
                "obs_head_arch": "256-256",
                "hidden_dim": 256,
                "out_head_arch": "256-256",
                "log_std_multiplier": 1.0,
                "log_std_offset": -1.0,
            }
        )


if __name__ == "__main__":
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    policy = ResNetPolicy(observation_dim=6, action_dim=6, image_size=256, orthogonal_init=True)
    # image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image = image_processor(images=image, return_tensors="jax")["pixel_values"]
    # [1, 2048, 7, 7], convert to NHWC
    print("processed image shape:", image.shape)
    image = jnp.transpose(image, (0, 2, 3, 1))

    print("image type:", type(image))
    print("image shape:", image.shape)

    observations = jnp.zeros((1, 6))
    # observations = extend_and_repeat(observations, 1, 4)
    print("observation shape:", observations.shape)
    # policy_params = policy.init(next_rng(policy.rng_keys()), observations=jnp.zeros((1, 6)), images=image)
    # set seed
    set_random_seed(0)
    # test policy
    policy_params = policy.init(next_rng(policy.rng_keys()), observations=observations, images=image)
