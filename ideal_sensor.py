#!/usr/bin/env python
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
import numpy as np

from DQN import *


class SENSOR(object):
    def __init__(self, box_extent=None, sensor_location=None, destination = [21.0, 28.4],attach_object=None):
        # half value of the box size
        if sensor_location is None:
            self.sensor_location = [4.5, 0, 0]
        if box_extent is None:
            self.sensor_range = [16, 8, 1]
        if attach_object is None:
            print("please assign the object that the sensor attach to")
        else:
            self.ego_actor = attach_object
        self.destination = destination
        self.bounding_box = carla.BoundingBox(
            carla.Location(
                self.sensor_location[0] * math.cos(self.ego_actor.get_transform().rotation.yaw * math.pi / 180) -
                self.sensor_location[1] * math.sin(self.ego_actor.get_transform().rotation.yaw * math.pi / 180),
                self.sensor_location[0] * math.sin(self.ego_actor.get_transform().rotation.yaw * math.pi / 180) +
                self.sensor_location[1] * math.cos(self.ego_actor.get_transform().rotation.yaw * math.pi / 180),
                0), carla.Vector3D(self.sensor_range[0], self.sensor_range[1], self.sensor_range[2]))

    def info_output(self, local_actor):
        # here the information of the actor are all the
        # relative information to the aro actor which the sensor is attached with
        vehicle_location = [self.ego_actor.get_location().x, self.ego_actor.get_location().y]
        vehicle_velocity = [self.ego_actor.get_velocity().x, self.ego_actor.get_velocity().y]
        actor_location = [local_actor.get_location().x - vehicle_location[0],
                          local_actor.get_location().y - vehicle_location[1]]
        actor_velocity = [local_actor.get_velocity().x - vehicle_velocity[0],
                          local_actor.get_velocity().y - vehicle_velocity[1]]
        actor_heading = local_actor.get_transform().rotation.yaw

        return actor_location, actor_velocity, actor_heading

    # the function is used to detect whether the actors in the scenario is in the range of bounding box
    def listen(self, actor_list):

        # creat the initial empty matrix
        state = np.zeros((self.sensor_range[0] * 2, self.sensor_range[1] * 2, 3))

        # the transform of the vehicle, use it to know the global location of the bounding box
        vehicle_transform = self.ego_actor.get_transform()

        for actor in actor_list:
            actor_loc = actor.get_transform().location
            # if the actor go into the bounding box
            if self.bounding_box.contains(actor_loc, vehicle_transform):
                location, velocity, heading_direction = self.info_output(actor)
                position_x = math.ceil(location[0])
                position_y = math.ceil(location[1])
                state[position_x, position_y, 0] = velocity[0]
                state[position_x, position_y, 1] = velocity[1]
                state[position_x, position_y, 2] = heading_direction
                # print(state[position_x, position_y, :])
                # state = torch.tensor(state, dtype=float)
        return state



#
#
# actor_list = []
# box_location = [4.5, 0, 0]
# box_extent = [8, 3, 1]
#
# try:
#     client = carla.Client("localhost", 2000)
#     client.set_timeout(2.0)
#     world = client.get_world()
#     blueprint_lib = world.get_blueprint_library()
#
#     # generate the ego vehicle
#     ego_bp = blueprint_lib.filter("model3")[0]
#     # transform = Transform(Location(x=20, y=195, z=40), Rotation(yaw=180))
#     transform = carla.Transform(carla.Location(x=-8.5, y=28.4, z=1), carla.Rotation(yaw=0))
#     ego_vehicle = world.try_spawn_actor(ego_bp, transform)
#
#     # generate a walker
#     walker_bp = blueprint_lib.filter("walker")[0]
#     walker_transform = carla.Transform(carla.Location(20, 28.4, 0.7), carla.Rotation(yaw=0))
#     walker = world.try_spawn_actor(walker_bp, walker_transform)
#     actor_list.append(walker)
#     start_time = time.time()
#     # generate a sensor
#     sensor_box = SENSOR(attach_object=ego_vehicle)
#     ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5))
#     while True:
#         state = sensor_box.listen(actor_list)
#         if time.time() - start_time == 20:
#             break

    # # remember to wait a moment until we can get the information of the actors
    # # time.sleep(0.1)
    #
    # debug = world.debug
    # # here the location of the bounding box is a relative position when it attach to the vehicle, we need to know it
    # # global position by using the transform thats why we need transform in the contains function
    # bounding_box = detecting_box(ego_vehicle, box_location, box_extent)
    # ego_vehicle.apply_control(carla.VehicleControl(throttle=0.3))
    # ## test if the pedestrian is in the range of bounding box
    # while True:
    #     # print(walker.get_transform().location)
    #     # print(ego_vehicle.get_transform().location)
    #     # print(actor_detection(walker, bounding_box, ego_vehicle))
    #     if actor_detection(walker, bounding_box, ego_vehicle):
    #         print('contacting ped')
    #         # print(walker.get_location())
    #         # print(walker.get_velocity())
    #         # print(walker.get_transform())
    #     if time.time() - start_time == 15:
    #         break
    #
    # here the location of the bounding box is relevant to the location of the vehicle, so here we should add the
    # location fo the vehicle
    # bounding_box.location += transform.location
    # debug.draw_box(bounding_box,
    #                ego_vehicle.get_transform().rotation, 0.05, carla.Color(255, 0, 0, 0), 2)

    # check whether the pedestrian is in the range of detection:

#     time.sleep(20)
# #
# finally:
#     for actor in actor_list:
#         ego_vehicle.destroy()
#         actor.destroy()
