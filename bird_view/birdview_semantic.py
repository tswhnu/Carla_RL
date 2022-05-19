import numpy as np
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
import math
import cv2
from bird_view.draw_routes import draw_route


def bird_seg(client, agent_vehicle, route_list=None, image_range=None, resolution=0.1):
    if image_range is None:
        image_range = [40.0, 40.0]
    birdview_producer = BirdViewProducer(
        client,
        target_size=PixelDimensions(width=int(image_range[0] / resolution), height=int(image_range[1] / resolution)),
        pixels_per_meter=int(1.0 / resolution),
        crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
    )

    birdview = birdview_producer.produce(
        agent_vehicle=agent_vehicle
    )

    rgb = BirdViewProducer.as_rgb(birdview)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if route_list is not None:
        bgr = draw_route(bgr, route_list, agent_vehicle, image_range, resolution)

    return bgr, birdview
