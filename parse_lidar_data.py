import numpy as np
import struct
import time

HEADER = 0x54
VERLEN = 0x2C
FRAME_LEN = 47
POINTS_PER_PACK = 12
BYTES_PER_POINT = 3

def sum8(data: bytes) -> int:
    return sum(data) & 0xFF

def crc8_maxim(data: bytes) -> int:
    # poly 0x31, init 0x00, refin/refout true, xorout 0x00
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0x8C  # reversed 0x31
            else:
                crc >>= 1
    return crc & 0xFF

def crc8_itu(data: bytes) -> int:
    # poly 0x07, init 0x00, no refin/refout, xorout 0x00
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ 0x07
            else:
                crc = (crc << 1) & 0xFF
    return crc & 0xFF

def crc8_j1850(data: bytes) -> int:
    # poly 0x1D, init 0xFF, xorout 0xFF, no refin/refout
    crc = 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ 0x1D
            else:
                crc = (crc << 1) & 0xFF
    crc ^= 0xFF
    return crc & 0xFF

def checksum_ok(frame: bytes) -> bool:
    body = frame[:-1]
    tail = frame[-1]
    return (
        sum8(body) == tail or
        crc8_maxim(body) == tail or
        crc8_itu(body) == tail or
        crc8_j1850(body) == tail
    )


def sync_and_read_frame(get_data, timeout_s: float, ) -> bytes | None:
    """Find 0x54 0x2C and return a full CRC-validated 47-byte frame or None on timeout."""
    start = time.time()
    buf = bytearray()
    while time.time() - start < timeout_s:
        chunk = get_data(128)
        if not chunk:
            continue
        buf.extend(chunk)

        i = 0
        while i <= len(buf) - 2:
            if buf[i] == HEADER and buf[i+1] == VERLEN:
                if len(buf) - i < FRAME_LEN:
                    break
                frame = bytes(buf[i:i+FRAME_LEN])
                if checksum_ok(frame):
                    del buf[:i+FRAME_LEN]
                    return frame
                i += 1
                continue
            i += 1

        if len(buf) > 1024:
            del buf[:512]
    return None

def parse_frame(frame: bytes):
    """Парсит один 47-байтный фрейм и возвращает словарь с данными и points в np.ndarray."""
    if len(frame) != FRAME_LEN or frame[0] != HEADER or frame[1] != VERLEN:
        return None

    # --- 1. Заголовок ---
    # '<BBHH' = header, verlen, speed, start_angle
    header, verlen, speed, start_angle = struct.unpack_from('<BBHH', frame, 0)

    # --- 2. Сырые точки (векторно) ---
    # каждая точка = 3 байта: dist_mm (uint16), intensity (uint8)
    raw_points = np.frombuffer(frame, dtype=np.uint8, offset=6, count=POINTS_PER_PACK * BYTES_PER_POINT)
    raw_points = raw_points.reshape(POINTS_PER_PACK, BYTES_PER_POINT)

    # дистанции
    dist_mm = raw_points[:, 0].astype(np.uint16) | (raw_points[:, 1].astype(np.uint16) << 8)
    intensity = raw_points[:, 2]

    # --- 3. Углы ---
    end_angle, ts_ms = struct.unpack_from('<HH', frame, 42)
    start_deg = (start_angle % 36000) / 100.0
    end_deg = (end_angle % 36000) / 100.0
    angle_diff = (end_deg - start_deg) % 360.0
    step = angle_diff / (POINTS_PER_PACK - 1) if POINTS_PER_PACK > 1 else 0.0
    angles_deg = (start_deg + np.arange(POINTS_PER_PACK) * step) % 360.0

    # --- 4. Фильтрация невалидных ---
    mask = (dist_mm != 0) & (dist_mm != 0xFFFF) & (intensity > 0)
    if not np.any(mask):
        return None

    dist_m = dist_mm[mask].astype(np.float32) / 1000.0
    angles_deg = angles_deg[mask]

    # --- 5. Формируем points для add_points_to_map() ---
    # Возвращаем массив (N, 2): [угол°, дистанция м]
    points = np.column_stack((angles_deg, dist_m))

    return {
        "speed": speed,
        "start_deg": start_deg,
        "end_deg": end_deg,
        "timestamp_ms": ts_ms,
        "points": points,
    }
