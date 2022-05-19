#!/usr/bin/env python
import glob
import logging
import os
import sys
import cv2
from ideal_sensor import *

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

from DQN import *
from bird_view.lidar_birdeye import *
from bird_view.birdview_semantic import *


def sensor_callback(data, queue):
    queue.put(data)


class CarEnv(object):
    # attributes shared by the object created from same class
    show_cam = False
    im_width = 400
    im_height = 400
    steer_control = 1.0
    front_camera = None

    """docstring for CarEnv"""

    def __init__(self, sync_mode=True):

        self.depth_camera = None
        self.depth_queue = None
        self.manual_mode = False
        self.sync_mode = True
        self.synchronous_master = False
        self.spawn_npc = False

        self.seconds_per_epi = 10
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager(8000)

        self.number_of_vehicles = 100
        self.num_of_pedestrian = 100

        # set the destination of the vehicle
        self.destination = [20.0, 28.4]
        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_bp = self.blueprint_library.filter("model3")[0]
        self.ego_transform = carla.Transform(carla.Location(x=8.3, y=-49.6, z=0.5), carla.Rotation(yaw=270))
        self.ego_vehicle = None
        self.collision_sensor = None
        self.lidar_sensor = None
        self.depth_camera_bp = None
        self.episode_start = None

        # blueprint of lidar
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=-0.5, z=1.8))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', str(64))
        self.lidar_bp.set_attribute('range', str(100))
        self.lidar_bp.set_attribute('points_per_second', str(100000))
        self.lidar_bp.set_attribute('dropoff_general_rate', str(0))

        # rgb_camera
        self.rgb_cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam_bp.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam_bp.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam_bp.set_attribute("fov", f"110")
        self.rgb_cam = None

        self.depth_camera_bp = self.blueprint_library.find("sensor.camera.depth")
        self.depth_camera_bp.set_attribute("image_size_x", f"{self.im_width}")
        self.depth_camera_bp.set_attribute("image_size_y", f"{self.im_height}")
        self.depth_camera_bp.set_attribute("fov", f"110")

        # the list that store the information
        self.collision_history = None
        self.vehicle_list = None
        self.sensor_list = None
        self.walkers_list = None
        self.all_id = None
        self.all_ped = None

        # the queue used to save the data from lidar
        self.lidar_queue = None
        self.image_queue = None
        self.position_hist = None

    def reset(self):
        self.synchronous_master = False

        # initialize the world
        self.collision_history = []
        self.all_id = []
        self.vehicle_list = []
        self.sensor_list = []
        self.walkers_list = []
        self.lidar_queue = Queue()
        self.image_queue = Queue()
        self.depth_queue = Queue()
        self.position_hist = []

        # set the world to sync mode
        if self.sync_mode:
            settings = self.world.get_settings()
            self.traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                self.synchronous_master = True
                settings.synchronous_mode = True
                # run at 10 fps
                settings.fixed_delta_seconds = 0.1
                self.world.apply_settings(settings)
        # generate all the actors (vehicle and pedestrian
        if self.spawn_npc:
            self.spawn_vehicles()
            self.spawn_pedestrian()

        # test mode, spawn a walker for test
        # walker_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        # walker_bp = random.choice(walker_bps)
        # walker_transform = carla.Transform(carla.Location(x=-1, y=28.4, z=0.5), carla.Rotation(yaw=0))
        # walker = self.world.try_spawn_actor(walker_bp, walker_transform)
        # self.walkers_list.append(walker)
        # self.all_id.append(walker.id)
        # vehicle_bp = self.blueprint_library.filter("model3")[0]
        # vehicle_transform = carla.Transform(carla.Location(x=0, y=28.4, z=0.5), carla.Rotation(yaw=0))
        # vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)
        # self.vehicle_list.append(vehicle)
        # spawn the ego vehicle
        while self.ego_vehicle is None:
            ego_point = random.choice(self.world.get_map().get_spawn_points())
            # ego_point = carla.Transform(carla.Location(x=-8.5, y=28.4, z=0.5), carla.Rotation(yaw=0))
            self.ego_vehicle = self.world.try_spawn_actor(self.ego_bp, ego_point)

        if self.manual_mode:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        else:
            self.ego_vehicle.set_autopilot(True, 8000)
        self.vehicle_list.append(self.ego_vehicle.id)
        ####################################################################
        # spawn the collision sensor, lidar sensor, camera
        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor,
                                                       carla.Transform(carla.Location(x=2.5, z=0.7)),
                                                       attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        self.sensor_list.append(self.collision_sensor)
        # lidar sensor generation
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)
        self.lidar_sensor.listen(lambda data: sensor_callback(data, self.lidar_queue))
        self.sensor_list.append(self.lidar_sensor)
        # rgb camera
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.rgb_cam = self.world.spawn_actor(self.rgb_cam_bp, transform, attach_to=self.ego_vehicle)
        self.rgb_cam.listen(lambda data: sensor_callback(data, self.image_queue))
        self.sensor_list.append(self.rgb_cam)
        # depth camera
        self.depth_camera = self.world.spawn_actor(self.depth_camera_bp,
                                              carla.Transform(carla.Location(x=2.5, z=0.7)),
                                              attach_to=self.ego_vehicle)
        self.depth_camera.listen(lambda data: sensor_callback(data, self.depth_queue))
        self.sensor_list.append(self.depth_camera)

        #######################################################################
        # try to get the environment data
        if self.sync_mode and self.synchronous_master:
            self.world.tick()
            state = self.get_env()
        else:
            self.world.wait_for_tick()
            state = self.get_env()
        # while self.front_camera is None:
        #     time.sleep(0.01)
        self.position_hist.append([self.ego_vehicle.get_location().x, self.ego_vehicle.get_location().y])
        return state

    def spawn_vehicles(self):
        # find all blueprints of vehicle
        vehicle_bps = self.world.get_blueprint_library().filter("vehicle.*")
        # sort the bps according to the bp id
        vehicle_bps = sorted(vehicle_bps, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)

        if self.number_of_vehicles < num_spawn_points:
            random.shuffle(spawn_points)
        else:
            print('dont have enough number of spawn points')

        # used the carla command to apply the command on batch of actors
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot

        FutureActor = carla.command.FutureActor

        # the batch to store the command
        batch = []

        for n, transform in enumerate(spawn_points):
            if n >= self.number_of_vehicles:
                break

            vehicle_bp = random.choice(vehicle_bps)
            batch.append(SpawnActor(vehicle_bp, transform)
                         .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))

        # execute the command
        for (i, response) in enumerate(self.client.apply_batch_sync(batch, self.synchronous_master)):
            if response.error:
                print(response.error)
            else:
                print("Future Actor", response.actor_id)
                self.vehicle_list.append(response.actor_id)

        # tick once let the actor spawn
        if not self.sync_mode or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()

    def spawn_pedestrian(self):
        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        SpawnActor = carla.command.SpawnActor

        percentagePedestriansRunning = 0.1  # how many pedestrians will run
        percentagePedestriansCrossing = 0.1  # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.num_of_pedestrian):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        walker_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_bps)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                print("spawn_error")
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_ped = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not self.sync_mode or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_ped[i].start()
            # set walk to random point
            self.all_ped[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_ped[i].set_max_speed(float(walker_speed[int(i / 2)]))

    def process_image(self, image):
        img = np.array(image.raw_data)
        img2 = img.reshape((self.im_height, self.im_width, 4))
        img3 = img2[:, :, :3]

        self.front_camera = img3

    def get_lidar_data(self, data):
        self.lidar_data = data

    def collision_data(self, event):
        self.collision_history.append(event)

    def test_env(self):
        # get the lidar data
        try:
            depth_data = self.depth_queue.get(True, 1.0)
            converter = carla.ColorConverter.Depth
            depth_data.convert(converter)
            img = np.array(depth_data.raw_data)
            img2 = img.reshape((self.im_width, self.im_height, 4))
            image = img2[:, :, :3].astype(np.uint8)
        except Empty:
            image = None
            print('cannot get lidar data')
        return image

    def get_env(self):

        # get the lidar data
        try:
            lidar_data = self.lidar_queue.get(True, 1.0)
            p_cloud_size = len(lidar_data)
            p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
            lidar_image = birds_eye_point_cloud(p_cloud, agent_vehicle=self.ego_vehicle)
        except Empty:
            lidar_image = None
            print('cannot get lidar data')

        # get the camera data
        try:
            image_data = self.image_queue.get(True, 1.0)
            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3]
        except Empty:
            im_array = None
            print('cannot get camera data')

        bird_view, one_hot = bird_seg(self.client, agent_vehicle=self.ego_vehicle)
        return [lidar_image, im_array, bird_view, one_hot]

    def step(self, action):\

        if action == 0:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
        elif action == 1:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
        elif action == 2:
            self.ego_vehicle.apply_control(carla.VehicleControl(brake=1.0))

        vehicle_speed = self.ego_vehicle.get_velocity()
        vehicle_vel = math.sqrt(vehicle_speed.x ** 2 + vehicle_speed.y ** 2 + vehicle_speed.z ** 2)
        v_kmh = int(3.6 * vehicle_vel)
        last_location = self.position_hist[-1]
        current_location = [self.ego_vehicle.get_location().x, self.ego_vehicle.get_location().y]
        self.position_hist.append(current_location)
        # reward function part
        if len(self.collision_history) != 0:
            done = True
            rc = -1000
        else:
            done = False
            rc = 1

        # we want the vehicle maintain the goal speed
        rv = - math.sqrt((v_kmh - 30) ** 2) * 10

        Lt = math.sqrt((last_location[0] - self.destination[0]) ** 2 + (last_location[1] - self.destination[1]) ** 2)

        Lt1 = math.sqrt(
            (current_location[0] - self.destination[0]) ** 2 + (current_location[1] - self.destination[1]) ** 2)

        if Lt1 < 0.5:
            # in this situation we think the vehicle reach the destination
            done = True
            rg = 200

        else:
            rg = (Lt - Lt1) * 100 / (0.1 * (vehicle_vel + 1))
        reward = rc + rv + rg

        if self.sync_mode:
            self.world.tick()
            state = self.get_env()
        else:
            self.world.wait_for_tick()
            state = self.get_env()
        return state, reward, done, None

    def terminal(self):
        self.world.apply_settings(self.original_settings)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        if self.spawn_npc:
            for i in range(0, len(self.all_id), 2):
                self.all_ped[i].stop()
        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
        for sensor in self.sensor_list:
            sensor.destroy()
        time.sleep(0.5)
