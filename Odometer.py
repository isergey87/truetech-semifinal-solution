import socket
import threading
import time
import json
import numpy as np
import serial

# === Constants for Cobra Flex (adjust based on your calibration) ===
WHEEL_BASE = 0.105      # м (distance between wheels for 4WD differential)
WHEEL_RADIUS = 0.050    # м (hub motor wheel radius)
TIME_STEP_S = 0.032     # с (32ms step, matching Webots for consistency)

class Odometer:
    def __init__(self, mode='serial', serial_port='/dev/ttyUSB0', baud=115200, timeout=0.2, 
                 HOST='192.168.4.1', PORT=8080):  # ESP32 AP IP/port for TCP fallback
        self.stop_event = threading.Event()
        self.timeout = timeout

        # Odometry state
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_th = 0.0
        self.current_v = 0.0
        self.current_w = 0.0

        # Thread stats
        self.updated_time = time.time()
        self.fps_ema = 0.0

        # Connection
        self.channel = None
        self.send_cmd = None
        self.recv_data = None
        if mode == 'serial':
            self.channel = serial.Serial(serial_port, baud, timeout=timeout)
            self.send_cmd = self._send_serial
            self.recv_data = self._recv_serial
            print(f"[Odometer] Connected via serial: {serial_port} @ {baud} (Cobra Flex USB)")
        else:
            self.channel = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.channel.settimeout(timeout)
            while True:
                try:
                    self.channel.connect((HOST, PORT))
                    break
                except Exception as e:
                    print(f"[Odometer] TCP connect retry to {HOST}:{PORT} ({e})")
                    time.sleep(0.5)
            self.channel.settimeout(None)
            self.send_cmd = self._send_tcp
            self.recv_data = self._recv_tcp
            print(f"[Odometer] Connected via TCP: {HOST}:{PORT} (Cobra Flex WiFi)")

        # Start thread
        self.thread = threading.Thread(target=self._odometry_thread, daemon=True)
        self.thread.start()

    # ================================================================
    # Send/Recv Methods
    # ================================================================
    def _send_serial(self, data: bytes):
        try:
            self.channel.write(data)
            self.channel.flush()
        except Exception as e:
            print(f"[Odometer] Serial send error: {e}")

    def _send_tcp(self, data: bytes):
        try:
            self.channel.sendall(data)
        except Exception as e:
            print(f"[Odometer] TCP send error: {e}")

    def _recv_serial(self, size: int) -> bytes:
        try:
            return self.channel.read(size) or b""
        except Exception:
            return b""

    def _recv_tcp(self, size: int) -> bytes:
        try:
            return self.channel.recv(size)
        except Exception:
            return b""

    # ================================================================
    # Request Odometry (JSON Protocol for Cobra Flex)
    # ================================================================
    def _request_odometry_data(self):
        # Send JSON command
        cmd = json.dumps({"cmd": "get_odom"}).encode('utf-8') + b'\n'
        self.send_cmd(cmd)

        # Receive response (up to 256 bytes, typical JSON size)
        data = b""
        start_time = time.time()
        while len(data) < 256 and (time.time() - start_time) < self.timeout:
            chunk = self.recv_data(256 - len(data))
            if not chunk:
                break
            data += chunk
            if b'\n' in data:
                data = data[:data.index(b'\n') + 1]  # Line-based for JSON
                break

        if not data:
            return None

        try:
            resp = json.loads(data.decode('utf-8').strip())
            if resp.get("cmd") == "odom_resp":
                return resp.get("odom", {})
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Odometer] JSON parse error: {e} (raw: {data.decode('utf-8', errors='ignore')})")
            return None

        return None

    # ================================================================
    # Update Odometry (from v_l, v_r or direct x,y,th)
    # ================================================================
    def _update_odometry(self, odom_data):
        # Flexible: use direct x,y,th if available, else compute from v_l/v_r
        if "x" in odom_data and "y" in odom_data and "th" in odom_data:
            self.odom_x = odom_data["x"]
            self.odom_y = odom_data["y"]
            self.odom_th = odom_data["th"]
        else:
            # Fallback differential computation (if only velocities)
            v_l = odom_data.get("v_l", 0.0)
            v_r = odom_data.get("v_r", 0.0)
            ds = (v_l + v_r) * TIME_STEP_S / 2
            dth = (v_r - v_l) * TIME_STEP_S / WHEEL_BASE
            th_mid = self.odom_th + dth / 2
            self.odom_x += ds * np.cos(th_mid)
            self.odom_y += ds * np.sin(th_mid)
            self.odom_th += dth

        # Update velocities
        if "v_l" in odom_data and "v_r" in odom_data:
            self.current_v = (odom_data["v_l"] + odom_data["v_r"]) / 2
            self.current_w = (odom_data["v_r"] - odom_data["v_l"]) / WHEEL_BASE
        else:
            self.current_v = 0.0
            self.current_w = 0.0

        self.updated_time = time.time()

    # ================================================================
    # Background Thread
    # ================================================================
    def _odometry_thread(self):
        print("[Odometer] Odometry thread started (Cobra Flex mode)")
        frame_count = 0
        last_t = time.time()
        while not self.stop_event.is_set():
            odom_data = self._request_odometry_data()
            if odom_data:
                self._update_odometry(odom_data)
                now = time.time()
                dt = max(1e-6, now - last_t)
                fps_inst = 1.0 / dt
                self.fps_ema = 0.9 * self.fps_ema + 0.1 * fps_inst if self.fps_ema > 0 else fps_inst
                frame_count += 1
                last_t = now
                if frame_count % 50 == 0:  # Periodic debug
                    print(f"[Odometer] FPS: {self.fps_ema:.1f}, Odom: x={self.odom_x:.3f}, y={self.odom_y:.3f}")
            else:
                time.sleep(TIME_STEP_S * 0.5)  # Backoff on failure

            # Reconnect on error (simple check)
            if not self.channel or self.channel.closed:
                print("[Odometer] Reconnecting...")
                time.sleep(1.0)
                # Re-init channel (simplified; extend if needed)

        print("[Odometer] Odometry thread stopped")

    # ================================================================
    # Public API
    # ================================================================
    def get_latest_odometry(self):
        """
        Returns tuple: (x, y, θ, v, 0.0, w, timestamp)
        Matches udp_diff format (gyro placeholder as 0.0).
        """
        return (
            self.odom_x,
            self.odom_y,
            self.odom_th,
            self.current_v,
            0.0,  # gyro_x placeholder
            self.current_w,
            self.updated_time
        )

    def stopped(self):
        return self.stop_event.is_set()

    def cleanup(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        if self.channel:
            try:
                self.channel.close()
            except Exception:
                pass
        print("[Odometer] Cleaned up (Cobra Flex)")
