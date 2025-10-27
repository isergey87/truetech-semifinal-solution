import spidev
import time
import struct
import threading
import numpy as np

class IMU:
    def __init__(self, spi=0, cs=0, max_speed_hz=1000000, mode=0):
        self.stop_event = threading.Event()
        self.updated_time = time.time()
        self.current_data = None
        self.acc_filtered = [0, 0, 0]
        self.gyr_filtered = [0, 0, 0]
        self.alpha = 0.2  # Exponential moving average coefficient

        # SPI setup
        self.spi = spidev.SpiDev()
        self.spi.open(spi, cs)
        self.spi.max_speed_hz = max_speed_hz
        self.spi.mode = mode

        # Sensor constants
        self.ACC_SENS = 0.244 / 1000 * 9.80665  # 0.244 mg/LSB at ±8g → m/s²
        self.GYR_SENS = 70.0 / 1000              # 70 mdps/LSB at ±2000dps → °/s

        # Verify sensor connection
        whoami = self._read_reg(0x0F)
        if whoami != 0x69:
            raise RuntimeError(f"LSM6DS3 not responding, WHO_AM_I = 0x{whoami:02X}, check connection!")

        # Configure sensor
        self._write_reg(0x10, 0b10011111)  # CTRL1_XL: 3.33kHz, ±8g, 100Hz filter
        self._write_reg(0x11, 0b10011100)  # CTRL2_G: 3.33kHz, ±2000 dps
        self._write_reg(0x12, 0b00000100)  # CTRL3_C: BDU=1, auto-increment=1
        time.sleep(0.1)  # Allow sensor to stabilize

        # Start data receiving thread
        self.data_thread = threading.Thread(target=self._receive_data_thread, args=())
        self.data_thread.daemon = True
        self.data_thread.start()

    def _read_reg(self, addr):
        """Read a single byte from the specified register."""
        addr = 0x80 | addr  # Set read bit
        resp = self.spi.xfer2([addr, 0x00])
        return resp[1]

    def _write_reg(self, addr, value):
        """Write a single byte to the specified register."""
        self.spi.xfer2([addr & 0x7F, value])

    def _read_16bit(self, addr):
        """Read a 16-bit value from the specified register."""
        addr = 0x80 | addr
        resp = self.spi.xfer2([addr, 0x00, 0x00])
        val = struct.unpack("<h", bytes(resp[1:]))[0]
        return val

    def _receive_data_thread(self):
        """Internal thread for receiving and processing IMU data."""
        frame_count = 0
        fps_ema = 0.0
        last_t = time.time()

        while not self.stop_event.is_set():
            # Read raw data
            ax = self._read_16bit(0x28)
            ay = self._read_16bit(0x2A)
            az = self._read_16bit(0x2C)
            gx = self._read_16bit(0x22)
            gy = self._read_16bit(0x24)
            gz = self._read_16bit(0x26)

            # Convert to physical units
            ax_m = ax * self.ACC_SENS
            ay_m = ay * self.ACC_SENS
            az_m = az * self.ACC_SENS
            gx_d = gx * self.GYR_SENS
            gy_d = gy * self.GYR_SENS
            gz_d = gz * self.GYR_SENS

            # Apply exponential moving average filter
            self.acc_filtered = [
                (1 - self.alpha) * self.acc_filtered[0] + self.alpha * ax_m,
                (1 - self.alpha) * self.acc_filtered[1] + self.alpha * ay_m,
                (1 - self.alpha) * self.acc_filtered[2] + self.alpha * az_m,
            ]
            self.gyr_filtered = [
                (1 - self.alpha) * self.gyr_filtered[0] + self.alpha * gx_d,
                (1 - self.alpha) * self.gyr_filtered[1] + self.alpha * gy_d,
                (1 - self.alpha) * self.gyr_filtered[2] + self.alpha * gz_d,
            ]

            # Store data as numpy array: [ax, ay, az, gx, gy, gz]
            self.current_data = np.array(
                self.acc_filtered + self.gyr_filtered,
                dtype=np.float32
            )
            self.updated_time = time.time()

            # Update FPS stats
            now = time.time()
            dt = max(1e-6, now - last_t)
            fps_inst = 1.0 / dt
            fps_ema = 0.9 * fps_ema + 0.1 * fps_inst if fps_ema > 0 else fps_inst
            frame_count += 1
            last_t = now

            time.sleep(0.01)  # Control read frequency (~100Hz)

    def get_latest_data(self):
        """Return the latest filtered IMU data and timestamp."""
        return self.current_data, self.updated_time

    def stopped(self):
        """Return True if the data thread has stopped."""
        return self.stop_event.is_set()

    def cleanup(self):
        """Stop the data thread and close the SPI connection."""
        self.stop_event.set()
        self.data_thread.join(timeout=1.0)
        try:
            self.spi.close()
        except Exception:
            pass
