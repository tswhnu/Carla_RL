import math
import numpy as np
import cv2


def draw_route(image, route_list=None, agent_vehicle=None, image_range=None, resolution=0.1):
    upper_left_corner = [agent_vehicle.get_location().x - image_range[0] / 2,
                         agent_vehicle.get_location().y - image_range[1] / 2]
    yaw = -agent_vehicle.get_transform().rotation.yaw
    if route_list is not None:
        pts_list = []
        for i in range(len(route_list)):
            wp = route_list[i][0]
 
            wp_location = [wp.transform.location.x, wp.transform.location.y]
            relative_location = [(wp_location[0] - upper_left_corner[0]), (wp_location[1] - upper_left_corner[1])]
            image_location = [relative_location[0] * math.cos(yaw * math.pi / 180) -
                              relative_location[1] * math.sin(yaw * math.pi / 180),
                              relative_location[0] * math.sin(yaw * math.pi / 180) +
                              relative_location[1] * math.cos(yaw * math.pi / 180)
                              ]
            pixel_position = [int(image_location[1] / resolution), -int(image_location[0] / resolution)]
            pts_list.append(pixel_position)
            # rgb[pixel_position[0], pixel_position[1], :] = [0, 255, 255]
        pts_list = np.array(pts_list, np.int32)
        pts = pts_list.reshape((-1, 1, 2))
        image = cv2.polylines(image, [pts], False, (0, 0, 255), 3)
    return image
