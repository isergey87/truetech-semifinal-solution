import numpy as np
import time
from typing import Tuple, Optional


class Odometry:
    def __init__(
        self,
        lidar: None,
        imu: None,
        wheel_radius: float = 0.05,
        track_width: float = 0.2,
    ):
        """
        Initialize odometry system using LIDAR and IMU data.

        Args:
            lidar: Lidar object providing point cloud data.
            imu: IMU object providing acceleration and angular velocity.
            wheel_radius: Robot wheel radius in meters (for velocity scaling).
            track_width: Distance between wheels in meters (for kinematics).
        """
        self.lidar = lidar
        self.imu = imu
        self.wheel_radius = wheel_radius
        self.track_width = track_width

        # Robot pose: [x, y, theta] (meters, meters, radians)
        self.pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pose_updated_time = time.time()

        # IMU-based velocity estimate
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)  # [vx, vy] in m/s
        self.last_imu_time = None

        # LIDAR previous scan for scan matching
        self.last_lidar_points = None
        self.last_lidar_time = None

    def _polar_to_cartesian(self, points: np.ndarray) -> np.ndarray:
        """
        Convert LIDAR points from polar [angle, distance] to Cartesian [x, y].

        Args:
            points: Nx2 array of [angle (degrees), distance (meters)].

        Returns:
            Nx2 array of [x, y] in meters.
        """
        angles = np.radians(points[:, 0])
        distances = points[:, 1]
        return np.column_stack((distances * np.cos(angles), distances * np.sin(angles)))

    def _icp(
        self, prev_points: np.ndarray, curr_points: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Simplified Iterative Closest Point (ICP) for 2D point cloud alignment.

        Args:
            prev_points: Nx2 array of previous scan [x, y] in meters.
            curr_points: Mx2 array of current scan [x, y] in meters.

        Returns:
            Tuple of (transformation [dx, dy, dtheta], error).
        """
        if prev_points.shape[0] < 10 or curr_points.shape[0] < 10:
            return np.array([0.0, 0.0, 0.0]), np.inf

        # Initialize transformation: [dx, dy, dtheta]
        transform = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        max_iterations = 20
        error_threshold = 0.01  # meters

        curr_points_transformed = curr_points.copy()
        for _ in range(max_iterations):
            # Find nearest neighbors (brute force for simplicity)
            distances = np.sqrt(
                ((prev_points[:, None] - curr_points_transformed[None, :]) ** 2).sum(
                    axis=2
                )
            )
            correspondences = np.argmin(distances, axis=1)
            matched_points = curr_points_transformed[correspondences]

            # Compute centroid alignment
            centroid_prev = np.mean(prev_points, axis=0)
            centroid_curr = np.mean(matched_points, axis=0)
            dx, dy = centroid_prev - centroid_curr

            # Compute rotation using point correspondences
            centered_prev = prev_points - centroid_prev
            centered_curr = matched_points - centroid_curr
            cov = centered_curr.T @ centered_prev
            U, _, Vt = np.linalg.svd(cov)
            R = U @ Vt
            dtheta = np.arctan2(R[1, 0], R[0, 0])

            # Apply transformation
            cos_theta = np.cos(dtheta)
            sin_theta = np.sin(dtheta)
            rotation_matrix = np.array(
                [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
            )
            curr_points_transformed = (curr_points_transformed @ rotation_matrix.T) + [
                dx,
                dy,
            ]

            # Update transform
            transform[0] += dx
            transform[1] += dy
            transform[2] += dtheta

            # Compute error
            error = np.mean(
                np.sqrt(
                    ((prev_points - curr_points_transformed[correspondences]) ** 2).sum(
                        axis=1
                    )
                )
            )
            if error < error_threshold:
                break

        return transform, error

    def update(imu_data, imu_time, lidar_points, lidar_time):
        current_time = time.time()

        # --- IMU Update ---
        imu_data, imu_time = self.imu.get_latest_data()
        if imu_data and ((not self.last_imu_time) or imu_time > self.last_imu_time):
            dt = imu_time - (self.last_imu_time or imu_time)
            self.last_imu_time = imu_time

            # Extract IMU data: [ax, ay, az, gx, gy, gz]
            ax, ay, _, _, _, gz = imu_data
            gz_rad = np.radians(gz)  # Convert °/s to rad/s

            # Update orientation (yaw) using angular velocity
            self.pose[2] += gz_rad * dt

            # Normalize theta to [-pi, pi]
            self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi

            # Update velocity estimate (simple integration, prone to drift)
            self.velocity += np.array([ax, ay]) * dt
            # Transform velocity to global frame
            cos_theta = np.cos(self.pose[2])
            sin_theta = np.sin(self.pose[2])
            global_velocity = np.array(
                [
                    self.velocity[0] * cos_theta - self.velocity[1] * sin_theta,
                    self.velocity[0] * sin_theta + self.velocity[1] * cos_theta,
                ]
            )
            # Update position (temporary, to be corrected by LIDAR)
            self.pose[:2] += global_velocity * dt
            self.pose_updated_time = current_time

        # --- LIDAR Update ---
        if lidar_points and (
            (not self.last_lidar_time) or lidar_time > self.last_lidar_time
        ):
            if self.last_lidar_points is not None:
                # Convert points to Cartesian
                prev_points = self._polar_to_cartesian(self.last_lidar_points)
                curr_points = self._polar_to_cartesian(lidar_points)

                # Run ICP to estimate relative transform [dx, dy, dtheta]
                transform, error = self._icp(prev_points, curr_points)
                if error < 0.5:  # Accept transform if error is reasonable
                    dx, dy, dtheta = transform
                    # Transform to global frame
                    cos_theta = np.cos(self.pose[2])
                    sin_theta = np.sin(self.pose[2])
                    global_dx = dx * cos_theta - dy * sin_theta
                    global_dy = dx * sin_theta + dy * cos_theta
                    # Update pose with LIDAR estimate
                    self.pose[0] += global_dx
                    self.pose[1] += global_dy
                    self.pose[2] += dtheta
                    self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi
                    self.pose_updated_time = current_time

                    # Reset IMU velocity to reduce drift
                    self.velocity = np.array([0.0, 0.0])

            self.last_lidar_points = lidar_points
            self.last_lidar_time = lidar_time

    def get_pose(self) -> Tuple[np.ndarray, float]:
        """Return the current pose [x, y, theta] and timestamp."""
        return self.pose.copy(), self.pose_updated_time
