import serial
import json
import threading
import time
import queue
import numpy as np

class Controller:
    def __init__(self, port='/dev/ttyACM1', baud=115200, timeout=0.2, wheel_radius=0.05, track_width=0.2, max_motor_cmd=3000, imu=None):
        self.stop_event = threading.Event()
        self.timeout = timeout
        self.latest_response = None
        self.response_updated_time = None
        self.response_queue = queue.Queue()  # Queue for storing incoming responses
        self.wheel_radius = wheel_radius  # Wheel radius in meters
        self.track_width = track_width    # Distance between wheels in meters
        self.max_motor_cmd = max_motor_cmd  # Maximum motor command value (e.g., 3000)
        self.max_linear_speed = 1.0       # Max linear speed in m/s (configurable)
        self.max_angular_speed = 2.0      # Max angular speed in rad/s (configurable)
        self.imu = imu                    # IMU instance for feedback
        self.kp_angular = 0.5             # Proportional gain for angular velocity control

        # Initialize serial connection
        self.serial = serial.Serial(port, baud, timeout=timeout)
        print(f"[Controller] Opened {port} @ {baud} baud")

        # Start response reading thread
        self.read_thread = threading.Thread(target=self._read_response_thread, args=())
        self.read_thread.daemon = True
        self.read_thread.start()

    def _read_response_thread(self):
        """Internal thread for reading and parsing JSON responses."""
        while not self.stop_event.is_set():
            try:
                line = self.serial.readline().decode('utf-8').strip()
                if not line:
                    continue
                print(f"{time.time()} ← Received: {line}")
                try:
                    response = json.loads(line)
                    self.latest_response = response
                    self.response_updated_time = time.time()
                    self.response_queue.put((response, self.response_updated_time))
                except json.JSONDecodeError:
                    print(f"{time.time()} ⚠️ JSON parsing error")
            except Exception as e:
                print(f"{time.time()} ⚠️ Error reading response: {e}")
            time.sleep(0.01)  # Prevent tight loop

    def send_json(self, cmd: dict):
        """Send a JSON command to the device."""
        try:
            data = json.dumps(cmd) + "\n"
            self.serial.write(data.encode('utf-8'))
            print(f"{time.time()} → Sent: {data.strip()}")
        except Exception as e:
            print(f"{time.time()} ⚠️ Error sending command: {e}")

    def send_json_raw(self, data: str):
        """Send a raw JSON string to the device."""
        try:
            data = data + "\n"
            self.serial.write(data.encode('utf-8'))
            print(f"{time.time()} → Sent: {data.strip()}")
        except Exception as e:
            print(f"{time.time()} ⚠️ Error sending raw command: {e}")

    def send(self, velocity: float, angular_velocity: float):
        """
        Send motor commands based on desired linear velocity (m/s) and angular velocity (rad/s),
        using IMU feedback for angular velocity control.

        Args:
            velocity (float): Linear velocity in m/s (positive = forward, negative = backward).
            angular_velocity (float): Angular velocity in rad/s (positive = counterclockwise).
        """
        try:
            # Get actual angular velocity from IMU (if available)
            actual_angular_velocity = 0.0
            if self.imu is not None:
                imu_data, _ = self.imu.get_latest_data()
                if imu_data is not None:
                    gz = imu_data[5]  # g_z in °/s
                    actual_angular_velocity = np.radians(gz)  # Convert to rad/s

            # Simple P-controller for angular velocity
            angular_error = angular_velocity - actual_angular_velocity
            angular_correction = self.kp_angular * angular_error

            # Adjust desired angular velocity with correction
            corrected_angular_velocity = angular_velocity + angular_correction

            # Differential drive kinematics:
            # v = (v_r + v_l) / 2, ω = (v_r - v_l) / track_width
            v_l = velocity - (self.track_width * corrected_angular_velocity) / 2
            v_r = velocity + (self.track_width * corrected_angular_velocity) / 2

            # Convert wheel velocities to motor commands
            max_wheel_angular_speed = self.max_linear_speed / self.wheel_radius
            left_cmd = (v_l / self.wheel_radius) / max_wheel_angular_speed * self.max_motor_cmd
            right_cmd = (v_r / self.wheel_radius) / max_wheel_angular_speed * self.max_motor_cmd

            # Clamp motor commands to [-max_motor_cmd, max_motor_cmd]
            left_cmd = max(min(left_cmd, self.max_motor_cmd), -self.max_motor_cmd)
            right_cmd = max(min(right_cmd, self.max_motor_cmd), -self.max_motor_cmd)

            # Round to integers for motor commands
            left_cmd = int(left_cmd)
            right_cmd = int(right_cmd)

            # Send JSON command
            cmd = {"T": 1, "L": left_cmd, "R": right_cmd}
            self.send_json(cmd)
        except Exception as e:
            print(f"{time.time()} ⚠️ Error in send: {e}")

    def get_latest_response(self):
        """Return the latest response and its timestamp."""
        return self.latest_response, self.response_updated_time

    def get_queued_response(self, block=False, timeout=None):
        """Get a response from the queue, optionally blocking."""
        try:
            return self.response_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None, None

    def stopped(self):
        """Return True if the read thread
