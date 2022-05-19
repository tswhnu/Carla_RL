import time

import cv2
from DDQN import *
from DDPG import *
from Env import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from bird_view.lidar_birdeye import *
import matplotlib.pyplot as plt

TRAIN_EPISODES = 5000

agent = DDQN()
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
            new_state, reward, done, _ = env.step(action)
            vel, distance = env.log()
            v_list.append(vel)
            dis_list.append(distance)
            # if new_state is not None:
            #     image = new_state
            #     cv2.imshow("depth", image[:, :, 3:6])
            #     cv2.waitKey(1)
            #     [lidar_image, camera_image, bird_view, one_hot]  = new_state
            #     if save_picture:
            #         cv2.imwrite('./scenario_1/lidar_image/frame' + str(step) +'.png', lidar_image)
            #         cv2.imwrite('./scenario_1/front_camera/frame' + str(step) + '.png', camera_image)
            #         np.save('./scenario_1/bird_view_onehot/frame'+str(step)+'.npy', one_hot)
            #     Hori = np.concatenate((lidar_image, camera_image, bird_view), axis=1)
            #     cv2.imshow("camera", Hori)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            epi_reward += reward
            agent.store_transition(current_state, action, reward, new_state, done)
            current_state = new_state
            step += 1
            if agent.memory_counter > MEMORY_CAPACITY:
                agent.optimize_model()

            if done:
                print("epi_reward:", epi_reward / step)
                reward_list.append(epi_reward / step)
                break
        if episode % 10 == 0:
            x = np.arange(0, len(v_list))
            plt.plot(x, v_list)
            plt.title('velocity')
            plt.show()
        if episode % 100 == 0 and save_model:
            print('saving data')
            plt.plot(reward_list)
            plt.ylim(-10, 10)
            plt.xlim(0, 900)
            plt.show()
            np.save('average_reward.npy', np.array(reward_list))
            agent.save_model("./policy_net"+str(episode)+'.pt', "./target_net"+str(episode)+'.pt')
        env.terminal()
        # env.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
finally:
    env.terminal()
