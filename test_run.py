import time

import cv2
import numpy as np

from DQN import *
from Env import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from bird_view.lidar_birdeye import *
import matplotlib.pyplot as plt

TRAIN_EPISODES = 1000

agent = DQN()
agent.load_model()
env = CarEnv()

torch.cuda.empty_cache()
destination = [21.0, 28.4]
save_picture = False
reward_list = []
save_model = False
# agent.load_model()

try:
    for episode in tqdm(range(1, TRAIN_EPISODES + 1), ascii=True, unit='episode'):
        v_list = []
        dis_list = []
        collision_num = 0
        total_reward = 0
        env.collision_history = []

        epi_reward = 0
        step = 1

        # reset the environment

        current_state = env.reset()

        # reset the finish flag
        done = False
        # get the start time

        episode_start = time.time()

        episode_dur = 10

        # begin to drive
        while True:
            # get the action based on the current state
            action = agent.select_action(current_state)
            # action = 0
            #0 get the result basde on the action
            new_state, reward, done, collision = env.step(action)
            collision_num += collision

            vel, distance = env.log()
            v_list.append(vel)
            dis_list.append(distance)
            epi_reward += reward
            agent.store_transition(current_state, action, reward, new_state, done)
            current_state = new_state
            step += 1
            if done:
                print("number of collision", collision_num)
                print("epi_reward:", epi_reward / step)
                reward_list.append(epi_reward / step)
                total_reward += epi_reward / step
                x = np.arange(0, len(v_list))
                plt.plot(x, v_list)
                plt.title('velocity')
                plt.xlabel('steps')
                plt.ylabel('velocity [m/s]')
                plt.xlim(0, 100)
                plt.show()
                y = np.arange(0, len(dis_list))
                plt.plot(y, dis_list)
                plt.xlim(0, 100)
                plt.title('distance')
                plt.xlabel('steps')
                plt.ylabel('distance [m]')
                plt.show()
                break

        env.terminal()
        # env.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
finally:
    print(collision_num/1000)
    print(total_reward/1000)
    env.terminal()
