import socket
import threading
import time

import numpy as np
import serial

from parse_lidar_data import sync_and_read_frame, parse_frame


class Lidar:
    def __init__(self, mode='serial', port='/dev/ttyUSB0', baud=230400, timeout=0.2, HOST='localhost', PORT=54321):
        self.stop_event = threading.Event()
        self.timeout = timeout
        self.current_revo = []
        self.last_angle_deg = None
        self.last_revo_points = None

        if mode == 'serial':
            self.channel = serial.Serial(port, baud, timeout=timeout)
            self.get_data = self.channel.read
        else:
            self.channel = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.channel.connect((HOST, PORT))
            self.get_data = self.channel.recv

        # Start telemetry receiving thread
        self.tel_thread = threading.Thread(
            target=self._receive_data_thread,
            args=()
        )
        self.tel_thread.daemon = True
        self.tel_thread.start()

    def add_points(self, points: np.ndarray):
        angles = points[:, 0]
        ranges = points[:, 1]

        for angle, dist in zip(angles, ranges):
            # Проверка перепрыгивания угла (360 → 0)
            if (
                    (self.last_angle_deg is not None
                     and self.last_angle_deg > 270.0
                     and angle < 90.0)
                    or (len(self.current_revo) > 100)
            ):
                # wrap detected: finalize revolution
                if self.current_revo:
                    self.last_revo_points = np.array(self.current_revo, dtype=np.float32)
                self.current_revo = []  # начинаем новый оборот
                print(f"{time.time()} full scan")

            # добавляем текущую точку
            self.current_revo.append((angle, dist))
            self.last_angle_deg = angle

    def _receive_data_thread(self):
        """Internal thread for receiving and processing telemetry."""

        # Stats
        frame_count = 0
        fps_ema = 0.0
        last_t = time.time()

        while not self.stop_event.is_set():
            frame = sync_and_read_frame(self.get_data, self.timeout)
            if frame is None:
                continue
            parsed = parse_frame(frame)
            if not parsed:
                continue
            self.add_points(parsed["points"])
            now = time.time()
            dt = max(1e-6, now - last_t)
            fps_inst = 1.0 / dt
            fps_ema = 0.9 * fps_ema + 0.1 * fps_inst if fps_ema > 0 else fps_inst
            frame_count += 1
            last_t = now

            print(f"LD19, fps={fps_ema:.1f} fps_inst={fps_inst:.1f} start={parsed["start_deg"]} end={parsed["end_deg"]}")

    def get_last_revo_points(self):
        return self.last_revo_points

    def stopped(self):
        """Return True if the telemetry thread has stopped."""
        return self.stop_event.is_set()

    def cleanup(self):
        self.stop_event.set()
        self.tel_thread.join(timeout=1.0)
        try:
            self.channel.close()
        except Exception:
            pass
