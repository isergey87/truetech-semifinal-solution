import time

from Lidar import Lidar
from Controler import Controler
from Odometry import Odometry
from IMU import IMU
import numpy as np

from Visualizer import Visualizer
from constants import MAP_SIZE, MAP_ORIGIN
from save_lidar_to_map import add_points_to_map


def main():
    lidar = Lidar(mode = 'socat', HOST = 'localhost', PORT = 54321)
    imu = IMU()
    controler = Controler(imu=imu)
    odometry = Odometry(lidar, imu)
    log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    visualizer = Visualizer(log_odds)
    last_lidar_time = None
    last_imu_time = None


    try:
        while True:
            lidar_points, lidar_updated_time = lidar.get_last_revo_points()
            if lidar_points and (not last_lidar_time or (lidar_updated_time > last_lidar_time)):
                a = {"data": lidar_points}
                print(a)
                print("-------")
                last_lidar_time = lidar_updated_time
                add_points_to_map(log_odds, lidar_points, 0, 0, MAP_ORIGIN, MAP_ORIGIN, 0)

            imu_data, imu_updated_time = imu.get_latest_data()
            if imu_data and (not imu_updated_time or (imu_updated_time > last_imu_time)):
                imu_packet = {
                    "acc": imu_data[:3],  # [ax, ay, az] in m/s²
                    "gyr": imu_data[3:],  # [gx, gy, gz] in °/s
                   }
                last_imu_time = imu_updated_time
                print(imu_packet)
                print("-------")

            odometry.update(lidar_points, lidar_updated_time, imu_data, imu_updated_time)
            od_data, od_time = odometry.get_pose()
            [x,y,theta] = od_data

            # controler.send(v,w)
                
            visualizer.render()


    except KeyboardInterrupt:
        print("[client] stop")

    finally:
        controler.cleanup()
        imu.cleanup()
        lidar.cleanup()


if __name__ == "__main__":
    main()
