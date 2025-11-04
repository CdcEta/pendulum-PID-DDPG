import gym
import time
import pygame

# Env Construction
env = gym.make('Pendulum-v1', g=1, render_mode="human")


# Env Reset
observation, info = env.reset()
max_torque = 10
env.max_torque = max_torque
env.action_space = gym.spaces.Box(-max_torque,max_torque)

# Env Infoac
print("observation space: " + str(env.observation_space))
print("action space: " + str(env.action_space))
step = 0
episode = 0

while True:
    step += 1
    # Random Action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

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

    if episode > 10:
        break

env.close()
