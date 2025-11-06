from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


class Module1(nn.Module):
    feature_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.feature_dim)(x)
        x = nn.relu(x)
        return x


class Module2(object):
    def __init__(self, feature_dim):
        self.module = Module1(feature_dim)
        self.params = self.module.init(jax.random.PRNGKey(0), jnp.ones((1, feature_dim)))

    def __call__(self, params, x):
        return self.module.apply(params, x)


class Module3(nn.Module):
    feature_dim: int

    def setup(self):
        self.net1 = Module2(self.feature_dim)
        self.net2 = nn.Dense(self.feature_dim)
        self.net1_params = self.param("net1", lambda rng: self.net1.params)

    def __call__(self, x):
        x = self.net1(self.net1_params, x)
        x = self.net2(x)
        return x


if __name__ == "__main__":

    # 1. ========== 模型初始化 ==========
    seed = 0
    key = jax.random.PRNGKey(seed)
    model = Module3(feature_dim=128)

    # 准备虚拟输入数据
    batch_size = 8
    input_dim = 128
    dummy_x = jnp.ones((batch_size, input_dim))

    # 统一在顶层调用 init
    init_key = jax.random.PRNGKey(0)
    variables = model.init(init_key, dummy_x)

    print("成功初始化！初始参数包含:", variables["params"].keys())

    # 保存初始的 net1 参数用于后续比较
    initial_net1_params = variables["params"]["net1"]

    # 2. ========== 创建 TrainState ==========
    learning_rate = 0.001
    # 创建一个 TrainState 来统一管理参数、优化器等
    train_state = TrainState.create(
        apply_fn=None,
        params=variables["params"],
        tx=optax.adam(learning_rate),
    )

    # 3. ========== 定义训练步骤 ==========
    # 使用 @partial(jax.jit) 编译此函数以获得高性能
    @partial(jax.jit, static_argnames=('policy'))
    def train_step(policy, state, batch_x, batch_y):
        # 定义损失函数
        def loss_fn(params):
            # state.apply_fn 就是 model.apply
            # 我们将 state.params 传入，Flax 会自动分发给 net1 和 net2
            logits = policy.apply({"params": params}, batch_x)
            # 使用简单的均方误差作为损失
            loss = jnp.mean(jnp.square(logits - batch_y))
            return loss

        # 计算损失和梯度
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        # 更新 TrainState (包括参数和优化器状态)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    # 4. ========== 训练循环 ==========
    num_steps = 100
    print(f"\n开始训练 {num_steps} 步...")

    for i in range(num_steps):
        # 创建虚拟的目标数据
        dummy_y = jnp.ones((batch_size, 128))

        # 执行一步训练
        train_state, current_loss = train_step(model, train_state, dummy_x, dummy_y)

        if (i + 1) % 20 == 0:
            print(f"Step {i+1:3d}, Loss: {current_loss:.6f}")

    print("训练完成！")

    # 5. ========== 参数验证 ==========
    # 从最终的 TrainState 中获取更新后的 net1 参数
    final_net1_params = train_state.params["net1"]

    # 计算初始参数和最终参数之间的差异
    diff_tree = jax.tree_util.tree_map(
        lambda initial, final: jnp.sum(jnp.abs(initial - final)),
        initial_net1_params,
        final_net1_params,
    )

    # 将差异树的所有叶子节点的值相加，得到总差异
    total_difference = jax.tree_util.tree_reduce(lambda x, y: x + y, diff_tree, 0)

    print("\n========== 验证结果 ==========")
    print(f"训练前后 net1 参数的总绝对差值: {total_difference:.8f}")

    if total_difference > 1e-6:
        print("✅ 成功！net1 的参数在训练过程中被更新了。")
    else:
        print("❌ 失败！net1 的参数没有被训练。")
