import carla
import random
import time
import open3d as o3d
import numpy as np
from matplotlib import cm
import cv2

IM_WIDTH = 640
IM_HEIGHT = 480

def preprocess_img(image):
    i = np.array(image.raw_data)
    i = i.reshape((IM_HEIGHT,IM_WIDTH,4))
    R, G, B, _ = cv2.split(i)
    img = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    return img

def get_cam(bp_lib):
    cam_bp = bp_lib.find('sensor.camera.depth')
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")
    cam_bp.set_attribute('sensor_tick', '.1')
    return cam_bp

actor_list = []


try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('cybertruck')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)
    actor_list.append(vehicle)

    cam = get_cam(blueprint_library)
    spawn_point = carla.Transform(carla.Location(x=0.8, z=2.5))
    sensor = world.spawn_actor(cam, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda image: image.save_to_disk(f"test{time.time()}.jpg",carla.ColorConverter().LogarithmicDepth))
    time.sleep(10)


finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
