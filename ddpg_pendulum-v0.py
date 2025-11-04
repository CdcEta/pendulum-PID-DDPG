"""
DDPG for Pendulum (Gymnasium Compatible)
Includes:
- PID manual controller
- DDPG training with replay buffer and target networks
- Visualization (plots + optional GIF)
- Run:python ddpg_pendulum-v0_2.py --env Pendulum-v1 --episodes n --render_eval --save_gif
"""

import os, math, time, argparse, datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ------------------- 环境兼容层 -------------------
def make_env(env_id: str = 'Pendulum-v1', render_mode: str = None, g: float = 10.0):
    try:
        import gymnasium as gym
        env = gym.make(env_id, g=g, render_mode=render_mode)
        return env, True
    except Exception:
        import gym
        env = gym.make(env_id)
        return env, False

def reset_env(env, is_gymnasium):
    if is_gymnasium:
        obs, info = env.reset()
    else:
        obs = env.reset(); info = {}
    return obs, info

def step_env(env, is_gymnasium, action):
    if is_gymnasium:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    else:
        obs, reward, done, info = env.step(action)
    return obs, reward, done, info

# ------------------- 网络定义 -------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = mlp([obs_dim, 256, 256, act_dim], nn.ReLU, nn.Tanh)
        self.act_limit = act_limit
    def forward(self, obs):
        return self.act_limit * self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, 256, 256, 1])
    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))

# ------------------- Replay Buffer -------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), np.float32)
        self.acts_buf = np.zeros((size, act_dim), np.float32)
        self.rews_buf = np.zeros((size, 1), np.float32)
        self.done_buf = np.zeros((size, 1), np.float32)
        self.max_size, self.ptr, self.size = size, 0, 0

    def store(self, obs, act, rew, obs2, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = obs2
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=128):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            obs2=self.obs2_buf[idxs],
            done=self.done_buf[idxs]
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

# ------------------- OU噪声 -------------------
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size; self.mu = mu; self.theta = theta; self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
    def reset(self): self.state = np.ones(self.size) * self.mu
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx; return self.state

# ------------------- 主训练逻辑 -------------------
def ddpg(env_id='Pendulum-v1', episodes=200, max_steps=200, seed=0, g=10.0,
         gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
         start_steps=1000, update_after=1000, update_every=50,
         act_noise=0.1, batch_size=128, replay_size=200000,
         render_eval=False, save_gif=False, out_root='runs_pendulum'):

    # === 自动创建带时间戳的输出目录 ===
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(out_root, f'run_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] 输出结果将保存至: {out_dir}")

    np.random.seed(seed); torch.manual_seed(seed)

    env, gymnasium_flag = make_env(env_id, render_mode='human', g=g)
    eval_env, _ = make_env(env_id, render_mode='rgb_array', g=g)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = Actor(obs_dim, act_dim, act_limit)
    critic1 = Critic(obs_dim, act_dim)
    critic2 = Critic(obs_dim, act_dim)
    targA = Actor(obs_dim, act_dim, act_limit)
    targC1 = Critic(obs_dim, act_dim)
    targC2 = Critic(obs_dim, act_dim)
    targA.load_state_dict(actor.state_dict())
    targC1.load_state_dict(critic1.state_dict())
    targC2.load_state_dict(critic2.state_dict())

    pi_opt = torch.optim.Adam(actor.parameters(), lr=pi_lr)
    q1_opt = torch.optim.Adam(critic1.parameters(), lr=q_lr)
    q2_opt = torch.optim.Adam(critic2.parameters(), lr=q_lr)
    buf = ReplayBuffer(obs_dim, act_dim, replay_size)
    noise = OUNoise(act_dim, sigma=act_noise)

    returns, pi_losses, q_losses = [], [], []

    for ep in range(episodes):
        o, _ = reset_env(env, gymnasium_flag)
        noise.reset(); ep_ret = 0
        for t in range(max_steps):
            if ep * max_steps + t < start_steps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).cpu().numpy()[0]
                a = np.clip(a + noise.sample(), -act_limit, act_limit)
            o2, r, d, _ = step_env(env, gymnasium_flag, a)
            buf.store(o, a, r, o2, d)
            o = o2; ep_ret += r
            if d: break

            total_t = ep * max_steps + t
            if total_t >= update_after and total_t % update_every == 0:
                for _ in range(update_every):
                    batch = buf.sample_batch(batch_size)
                    obs_b, act_b, rew_b, obs2_b, done_b = batch['obs'], batch['acts'], batch['rews'], batch['obs2'], batch['done']
                    with torch.no_grad():
                        a2 = targA(obs2_b)
                        tq1 = targC1(obs2_b, a2)
                        tq2 = targC2(obs2_b, a2)
                        backup = rew_b + gamma * (1 - done_b) * torch.min(tq1, tq2)
                    q1 = critic1(obs_b, act_b)
                    q2 = critic2(obs_b, act_b)
                    q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
                    q1_opt.zero_grad(); q2_opt.zero_grad(); q_loss.backward(); q1_opt.step(); q2_opt.step()
                    pi = actor(obs_b)
                    pi_loss = -critic1(obs_b, pi).mean()
                    pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()
                    q_losses.append(q_loss.item()); pi_losses.append(pi_loss.item())
                    with torch.no_grad():
                        for p, tp in zip(actor.parameters(), targA.parameters()):
                            tp.data.mul_(polyak); tp.data.add_((1 - polyak) * p.data)
                        for p, tp in zip(critic1.parameters(), targC1.parameters()):
                            tp.data.mul_(polyak); tp.data.add_((1 - polyak) * p.data)
                        for p, tp in zip(critic2.parameters(), targC2.parameters()):
                            tp.data.mul_(polyak); tp.data.add_((1 - polyak) * p.data)
        returns.append(ep_ret)
        print(f"Episode {ep+1}/{episodes} Return: {ep_ret:.1f}")

    # === 保存训练曲线 ===
    plt.figure(); plt.plot(returns); plt.title("Episode Returns"); plt.xlabel("Episode"); plt.ylabel("Return"); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "returns.png"), dpi=150); plt.close()
    plt.figure(); plt.plot(pi_losses); plt.title("Policy Loss"); plt.savefig(os.path.join(out_dir, "pi_loss.png")); plt.close()
    plt.figure(); plt.plot(q_losses); plt.title("Q Loss"); plt.savefig(os.path.join(out_dir, "q_loss.png")); plt.close()

    # === 评估 + GIF 保存 ===
    if render_eval or save_gif:
        obs, _ = reset_env(eval_env, True)
        frames = []
        ret = 0
        for t in range(200):
            with torch.no_grad():
                a = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).cpu().numpy()[0]
            obs, r, done, _ = step_env(eval_env, True, a)
            ret += r
            frame = eval_env.render()
            if save_gif and frame is not None:
                frames.append(frame)
            if done: break
        print(f"Eval Return: {ret:.1f}")
        if save_gif and frames:
            imageio.mimsave(os.path.join(out_dir, "eval.gif"), frames, fps=30)
            print(f"[INFO] 评估GIF已保存: {os.path.join(out_dir, 'eval.gif')}")

    print(f"[INFO] 所有结果已保存至: {out_dir}")

# ------------------- 主入口 -------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Pendulum-v1")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--render_eval", action="store_true")
    p.add_argument("--save_gif", action="store_true")
    args = p.parse_args()
    ddpg(env_id=args.env, episodes=args.episodes, render_eval=args.render_eval, save_gif=args.save_gif)

if __name__ == "__main__":
    main()
