#!/usr/bin/env python
import glob
import os
import sys
import cv2
from ideal_sensor import *
import matplotlib.pyplot as plt
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import math
from queue import Queue
from queue import Empty
from scipy.signal import *
import matplotlib.pyplot as plt
from DQN import *

data1 = np.load('model_02/average_reward.npy')
data2 = np.load('model_03/average_reward.npy')
data_smooth1 = savgol_filter(data1,53,3)
data_smooth2 = savgol_filter(data2, 53, 3)
plt.plot(data_smooth1, color = "blue", label="DQN")
plt.plot(data_smooth2, color = "red", label="Double DQN")
plt.legend()
plt.xlim(0,1000)
plt.ylim(0,10)
plt.xlabel("episodes")
plt.ylabel("average reward")
plt.show()
