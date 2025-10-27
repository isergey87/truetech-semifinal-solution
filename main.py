import time

from Lidar import Lidar
import numpy as np

from Visualizer import Visualizer
from constants import MAP_SIZE, MAP_ORIGIN
from save_lidar_to_map import add_points_to_map


def main():
    lidar = Lidar(mode = 'socat', HOST = 'localhost', PORT = 54321)
    log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    visualizer = Visualizer(log_odds)
    last_lidar_time = None


    try:
        while True:
            lidar_points, lidar_updated_time = lidar.get_last_revo_points()
            if lidar_points is not None and (last_lidar_time is None or lidar_updated_time > last_lidar_time):
                a = {"data": lidar_points}
                print(a)
                print("-------")
                last_lidar_time = lidar_updated_time
                add_points_to_map(log_odds, lidar_points, 0, 0, MAP_ORIGIN, MAP_ORIGIN)
            visualizer.render()


    except KeyboardInterrupt:
        print("[client] stop")

    finally:
        lidar.cleanup()


if __name__ == "__main__":
    main()
