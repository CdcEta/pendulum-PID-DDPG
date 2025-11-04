import gym
import time
import pygame
import math

# Env Construction
env = gym.make('Pendulum-v1', g=1, render_mode="human")


# Env Reset
observation, info = env.reset()
max_torque = 10
env.max_torque = max_torque
env.action_space = gym.spaces.Box(-max_torque,max_torque)

# Env Info
print("observation space: " + str(env.observation_space))
print("action space: " + str(env.action_space))
step = 0
episode = 0

# 初始化PID参数
P = 45.0
I = 0.50
D = 10.0

# 初始化变量
prev_error = 0
integral = 0

def get_current_state(obs):
    x = obs[0]
    y = obs[1]
    w = obs[2]
    theta = math.atan2(y, x)
    return theta, w

while True:
    step += 1
    # action = env.action_space.sample()

    # 获取当前摆杆的角度和角速度
    current_angle, current_angular_velocity = get_current_state(observation)

    # 计算偏差
    # error = 0 - current_angle
    error = 0 - current_angle
    # 计算积分误差
    integral = integral + error

    # 计算微分误差
    derivative = error - prev_error

    # 计算控制输出
    control_output = P * error + I * integral + D * derivative

    # 施加力矩控制倒立摆
    # apply_torque(control_output)
    action = [control_output]
    observation, reward, terminated, truncated, info = env.step(action)

    # 更新前一次的偏差
    prev_error = error

    env.render()
    obs_action_string = "x: {:.2f} y: {:.2f} w: {:.2f} ".format(observation[0], observation[1], observation[2])
    obs_action_string += "Torque: {:.2f} ".format(action[0])
    obs_action_string += "Episode: {:.0f} ".format(episode)
    pygame.display.set_caption(obs_action_string)
    # time.sleep(1)

    if terminated or truncated:
        print("Episode Step: " + str(step))
        print("New episode start!")
        observation, info = env.reset()
        step = 0
        episode += 1
        prev_error = 0
        integral = 0

    if episode > 10:
        break

env.close()
