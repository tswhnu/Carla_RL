#!/usr/bin/env python
import glob
import os
import sys
import cv2
import random

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
from queue import Queue, Empty

from DQN import *

IM_WIDTH = 400
IM_HEIGHT = 400
SHOW_PREVIEW = False
SECONDS_PER_EPI = 10

actor_list = []

def speed_reward(current_speed, speed_limit, ratio1, ratio2):
    # if have collision risk, the vehicle speed need to be smaller than speed limit
    if speed_limit == 0:
        if current_speed == 0.0:
            reward = 0
        else:
            reward = -100
    # elif speed_limit == 60:
    #     reward = - ((current_speed - speed_limit) ** 2 - 25) * 0.05

    elif current_speed > speed_limit:
        reward = - (current_speed - speed_limit) ** 2 * 0.05
    else:
        reward = ratio2 * current_speed / (speed_limit + 0.1)
        # reward = 0

    return reward

def sensor_callback(data, queue):
    queue.put(data)


class CarEnv(object):
    # attributes shared by the object created from same class
    show_cam = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    steer_control = 1.0
    front_camera = None
    """docstring for CarEnv"""

    def __init__(self):

        self.sync_mode = True
        self.client = carla.Client("localhost", 2000)
        # self.client = carla.Client("192.168.1.100", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        # settings = self.world.get_settings()
        # settings.no_rendering_mode = True
        # self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model3 = self.blueprint_library.filter("model3")[0]

        self.depth_camera_bp = self.blueprint_library.find("sensor.camera.depth")
        self.depth_camera_bp.set_attribute("image_size_x", f"{self.im_width}")
        self.depth_camera_bp.set_attribute("image_size_y", f"{self.im_height}")
        self.depth_camera_bp.set_attribute("fov", f"110")

        self.optical_camera_bp = self.blueprint_library.find("sensor.camera.optical_flow")
        self.optical_camera_bp.set_attribute("image_size_x", f"{self.im_width}")
        self.optical_camera_bp.set_attribute("image_size_y", f"{self.im_height}")
        self.optical_camera_bp.set_attribute("fov", f"110")

    def reset(self):
        # initialize the world
        self.collision_history = []
        self.actor_list = []
        self.sensor_list = []
        self.position_hist = []
        self.depth_queue = Queue()
        self.optical_queue = Queue()
        self.last_image = None
        self.last_vel = None

        # walker_position = [10.0 + random.uniform(-15.0, 15.0), 28.4 + random.uniform(-1.0, 1.0)]
        walker_position = [23, 28.4]
        self.destination = [walker_position[0] + 5.0, walker_position[1]]
        if self.sync_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            # run at 10 fps
            settings.fixed_delta_seconds = 0.1
            self.world.apply_settings(settings)

        # vehicle actor
        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.transform = carla.Transform(carla.Location(x=-10, y=28.4, z=0.5), carla.Rotation(yaw=0))
        self.vehicle = self.world.spawn_actor(self.model3, self.transform)

        self.actor_list.append(self.vehicle)
        # # rgb_camera
        # self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        # self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        # self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        # self.rgb_cam.set_attribute("fov", f"110")
        #
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        # self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        # self.sensor.listen(lambda data: self.process_image(data))
        # self.actor_list.append(self.sensor)


        self.depth_camera = self.world.spawn_actor(self.depth_camera_bp,
                                                   carla.Transform(carla.Location(x=2.5, z=0.7)),
                                                   attach_to=self.vehicle)
        self.depth_camera.listen(lambda data: sensor_callback(data, self.depth_queue))
        self.sensor_list.append(self.depth_camera)

        # self.optical_flow_camera = self.world.spawn_actor(self.optical_camera_bp,
        #                                                   carla.Transform(carla.Location(x=2.5, z=0.7)),
        #                                                   attach_to=self.vehicle)
        #
        # self.optical_flow_camera.listen(lambda data: sensor_callback(data, self.optical_queue))
        # self.sensor_list.append(self.optical_flow_camera)


        walker_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        walker_bp = walker_bps[23]
        # walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_transform = carla.Transform(carla.Location(x=walker_position[0], y=walker_position[1], z=0.5), carla.Rotation(yaw=0))
        self.walker = self.world.try_spawn_actor(walker_bp, walker_transform)
        self.actor_list.append(self.walker)

        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, transform, attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        self.sensor_list.append(self.collision_sensor)


        # while self.front_camera is None:
        self.episode_start = time.time()
        # self.vehicle.apply_control(carla.VehicleControl(manual_gear_shift = True, gear=1))

        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        state = self.test_env()
        return state

    def collision_data(self, event):
        self.collision_history.append(event)

    def test_env(self):
        walker_location = [self.walker.get_location().x, self.walker.get_location().y]
        ego_location = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        walker_speed = [self.walker.get_velocity().x, self.walker.get_velocity().y]
        ego_speed = [self.vehicle.get_velocity().x, self.vehicle.get_velocity().y]

        return [walker_location[0] - ego_location[0],
                walker_location[1] - ego_location[1],
                walker_speed[0] - ego_speed[0],
                walker_speed[1] - ego_speed[1]]

    def get_env(self):
        # get the lidar data
        try:
            depth_data = self.depth_queue.get(True, 1.0)
            converter = carla.ColorConverter.LogarithmicDepth
            depth_data.convert(converter)
            img = np.array(depth_data.raw_data)
            img2 = img.reshape((self.im_width, self.im_height, 4))
            image = img2[:, :, :3].astype(np.uint8)
            if self.last_image is None:
                cat_image = np.concatenate((image, image), axis=2)
            else:
                cat_image = np.concatenate((image, self.last_image), axis=2)
            self.last_image = image
        except Empty:
            # image = None
            cat_image = None
            print('cannot get depth data')
        return cat_image

    def process_image(self, image):
        img = np.array(image.raw_data)
        img2 = img.reshape((self.im_height, self.im_width, 4))
        img3 = img2[:, :, :3]

        if self.show_cam:
            cv2.imshow("preview", img3)
            cv2.waitKey(1)

        self.front_camera = img3 / 255

    def step(self, action):
        # action = float(action)
        # if action >= 0:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=action))
        # else:
        #     self.vehicle.apply_control(carla.VehicleControl(brake=-action))
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
        # elif action == 3:
        #     self.vehicle.apply_control(carla.VehicleControl(brake=0.5))
        # elif action == 3:
        #     self.vehicle.apply_control(carla.VehicleControl(brake=1))
        # elif action == 2:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0))
        # elif action == 3:
        #     self.vehicle.apply_control(carla.VehicleControl(brake=1.0))

        vehicle_speed = self.vehicle.get_velocity()
        # relative_speed = math.sqrt((vehicle_speed.x - ped_speed.x) ** 2 + (vehicle_speed.y - ped_speed.y) ** 2)
        vehicle_vel = math.sqrt(vehicle_speed.x ** 2 + vehicle_speed.y ** 2 + vehicle_speed.z ** 2)
        v_kmh = 3.6 * vehicle_vel
        current_location = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        if self.last_vel is None:
            vel_hist = vehicle_vel
        else:
            vel_hist = self.last_vel
        if len(self.position_hist) == 0:
            last_location = current_location
        else:
            last_location = self.position_hist[-1]
        self.position_hist.append(current_location)
        ped_location = [15, 28.4]
        Lt = math.sqrt((last_location[0] - self.destination[0]) ** 2 + (last_location[1] - self.destination[1]) ** 2)
        Lt1 = math.sqrt(
            (current_location[0] - self.destination[0]) ** 2 + (current_location[1] - self.destination[1]) ** 2)
        des_ped = math.sqrt(
            (current_location[0] - ped_location[0]) ** 2 + (current_location[1] - ped_location[1]) ** 2)
        # reward function part
        if len(self.collision_history) != 0:
            print("collision")
            print(self.collision_history[0])
            done = True
            print("last_action:", action)
            rc = -1000
            reward = rc
        else:
            done = False
            if des_ped >= 7:
                speed_limit = 60
            else:
                speed_limit = 6 * (des_ped - 3.5)
                if des_ped <= 3.5:
                    speed_limit = 0  # km/h
            rv = speed_reward(v_kmh, speed_limit, 1, 1)
            # print("des:", des_ped, "curr", v_kmh, "speed_ limit", speed_limit, "rv:", rv)
            rg = (Lt - Lt1) * 10 / (0.1 * (vel_hist + 1))
            rc = 0
            reward = rc + rv + rg
        if self.episode_start + SECONDS_PER_EPI < time.time():
            done = True

        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        state = self.test_env()
        return state, reward, done, len(self.collision_history)

    def log(self):
        vehicle_speed = self.vehicle.get_velocity()
        vehicle_vel = math.sqrt(vehicle_speed.x ** 2 + vehicle_speed.y ** 2 + vehicle_speed.z ** 2)
        v_kmh = 3.6 * vehicle_vel
        vehicle_location = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        ped_location = [self.walker.get_location().x, self.walker.get_location().y]
        ped_distance = math.sqrt((vehicle_location[0] - ped_location[0]) ** 2 + (vehicle_location[1] - ped_location[1]) ** 2)
        return v_kmh, ped_distance

    def terminal(self):
        self.world.apply_settings(self.original_settings)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        for sensor in self.sensor_list:
            sensor.destroy()
        time.sleep(0.5)
