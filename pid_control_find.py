import gym
import pygame
import math
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 环境构建 ====================
env = gym.make('Pendulum-v1', g=1, render_mode="human")
observation, info = env.reset()
max_torque = 10
env.max_torque = max_torque
env.action_space = gym.spaces.Box(-max_torque, max_torque)
dt = env.dt if hasattr(env, "dt") else 0.05

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# ==================== 辅助函数 ====================
def angle_wrap(a):
    """角度环绕到 [-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi

def get_current_state(obs):
    x, y, w = obs
    theta = math.atan2(y, x)
    return theta, w

def run_episode(P, I, D, render=False, max_steps=400, seed=None):
    """运行一局，返回稳定性指标（越小越稳）"""
    local_env = gym.make('Pendulum-v1', g=1)  # 每个线程独立环境
    if seed is not None:
        observation, info = local_env.reset(seed=seed)
    else:
        observation, info = local_env.reset()

    prev_error = 0.0
    integral = 0.0
    I_LIMIT = 20.0
    total_abs_error = 0.0
    dt = local_env.dt if hasattr(local_env, "dt") else 0.05

    for step in range(max_steps):
        theta, w = get_current_state(observation)
        error = angle_wrap(0 - theta)
        integral = np.clip(integral + error * dt, -I_LIMIT, I_LIMIT)
        derivative = (error - prev_error) / dt
        torque = P * error + I * integral + D * derivative
        torque = float(np.clip(torque, -max_torque, max_torque))
        observation, reward, terminated, truncated, info = local_env.step([torque])
        prev_error = error
        total_abs_error += abs(error)
        if terminated or truncated:
            break

    local_env.close()
    return total_abs_error / max_steps

# ==================== 保存结果图函数 ====================
def save_pid_results(results, best_params, save_dir="runs_pid"):
    """
    根据PID调参结果生成可视化图像并保存到指定目录。
    results: [(P, I, D, avg_score), ...]
    """
    os.makedirs(save_dir, exist_ok=True)
    data = np.array(results)
    P_vals, I_vals, D_vals, scores = data[:,0], data[:,1], data[:,2], data[:,3]

    # ---- 绘制误差分布散点图 ----
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(P_vals, I_vals, D_vals, c=scores, cmap='viridis', s=40)
    ax.set_xlabel('K_P')
    ax.set_ylabel('K_I')
    ax.set_zlabel('K_D')
    fig.colorbar(p, ax=ax, label='Average Error')
    ax.set_title('PID Parameter Search Result')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pid_search_3d.png"), dpi=200)
    plt.close()

    # ---- 绘制二维误差平面投影 ----
    fig2, ax2 = plt.subplots(figsize=(7,5))
    scatter = ax2.scatter(P_vals, D_vals, c=scores, cmap='coolwarm', s=60)
    plt.colorbar(scatter, label='Average Error')
    ax2.set_xlabel("K_P")
    ax2.set_ylabel("K_D")
    ax2.set_title("Error Distribution (K_P vs K_D)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pid_error_plane.png"), dpi=200)
    plt.close()

    # ---- 绘制调参收敛趋势 ----
    sorted_scores = np.sort(scores)
    plt.figure(figsize=(6,4))
    plt.plot(sorted_scores, linewidth=1.5)
    plt.xlabel("Parameter Combination Index (sorted)")
    plt.ylabel("Average Error")
    plt.title("PID Tuning Convergence Trend")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pid_convergence.png"), dpi=200)
    plt.close()

    # ---- 输出最优参数到文本文件 ----
    with open(os.path.join(save_dir, "best_pid.txt"), "w") as f:
        f.write(f"Best PID Parameters:\n")
        f.write(f"P={best_params[0]:.3f}, I={best_params[1]:.3f}, D={best_params[2]:.3f}\n")

    print(f"\n[INFO] PID 调参结果图与文本已保存至: {os.path.abspath(save_dir)}")

# ==================== 并行自动调参函数 ====================
def auto_tune_pid_parallel(p_range, i_range, d_range, repeat=3, max_workers=8):
    """并行扫描 PID 参数组合并找出最稳定的一组"""
    best_score = float('inf')
    best_params = (0, 0, 0)
    results = []

    param_combos = list(itertools.product(p_range, i_range, d_range))
    print(f"共有 {len(param_combos)} 组 PID 参数待测试...")

    def evaluate_pid(P, I, D):
        scores = []
        for i in range(repeat):
            score = run_episode(P, I, D, render=False, seed=i)
            scores.append(score)
        return (P, I, D, np.mean(scores))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_pid, P, I, D): (P, I, D) for (P, I, D) in param_combos}
        for future in as_completed(futures):
            P, I, D = futures[future]
            try:
                P, I, D, avg_score = future.result()
                results.append((P, I, D, avg_score))
                print(f"PID ({P:.1f}, {I:.1f}, {D:.1f}) -> 平均误差: {avg_score:.4f}")
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = (P, I, D)
            except Exception as e:
                print(f"参数 ({P},{I},{D}) 测试失败: {e}")

    print("\n=== 自动调参完成 ===")
    print(f"最优 PID: P={best_params[0]:.2f}, I={best_params[1]:.2f}, D={best_params[2]:.2f}")
    print(f"最小平均误差: {best_score:.4f}")

    # === 调参完成后保存结果 ===
    save_pid_results(results, best_params)

    return best_params, results

# ==================== 主程序 ====================
if __name__ == "__main__":
    pygame.init()

    # 参数扫描范围
    p_values = np.linspace(20, 45, 6)   # 20,25,30,35,40,45
    i_values = np.linspace(0.5, 3.0, 6) # 0.5,1.0,1.5,2.0,2.5,3.0
    d_values = np.linspace(6, 14, 5)    # 6,8,10,12,14

    # 自动调参（并行版）
    best_params, all_results = auto_tune_pid_parallel(
        p_values, i_values, d_values,
        repeat=3,
        max_workers=8  # 可根据CPU核心数调整
    )

    # 使用最优参数可视化演示
    P, I, D = best_params
    print("\n使用最优参数运行演示（可视化）...")
    observation, info = env.reset()
    prev_error = 0
    integral = 0
    I_LIMIT = 20.0
    for step in range(600):
        theta, w = get_current_state(observation)
        error = angle_wrap(0 - theta)
        integral = np.clip(integral + error * dt, -I_LIMIT, I_LIMIT)
        derivative = (error - prev_error) / dt
        torque = P * error + I * integral + D * derivative
        torque = float(np.clip(torque, -max_torque, max_torque))
        observation, reward, terminated, truncated, info = env.step([torque])
        prev_error = error
        env.render()
        pygame.display.set_caption(f"θ={theta:.2f}, ω={w:.2f}, τ={torque:.2f}, P={P},I={I},D={D}")
        if terminated or truncated:
            break

    env.close()
    pygame.quit()
