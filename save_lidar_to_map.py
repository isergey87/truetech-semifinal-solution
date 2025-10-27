import numpy as np
from numba import njit

from constants import MAP_ORIGIN, MAP_RESOLUTION, MAP_SIZE

lo_occ = 0.85
lo_free = -0.4
lo_min, lo_max = -5, 5

occupied_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)


def to_index(x, y):
    gx = MAP_ORIGIN + np.floor(x / MAP_RESOLUTION).astype(np.int32)
    gy = MAP_ORIGIN + np.floor(y / MAP_RESOLUTION).astype(np.int32)
    return gx, gy


@njit(cache=True)
def update_log_odds_numba(
        gx_idx, gy_idx,
        rx_idx, ry_idx,
        log_odds,
        lo_occ, lo_free, lo_min, lo_max
):
    local_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)

    # === Добавляем занятые клетки ===
    for i in range(gx_idx.shape[0]):
        gx = gx_idx[i]
        gy = gy_idx[i]
        if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
            log_odds[gx, gy] = min(lo_max, max(lo_min, log_odds[gx, gy] + lo_occ))
            local_mask[gx, gy] = True

    # === Рисуем свободные клетки ===
    for i in range(gx_idx.shape[0]):
        gx = gx_idx[i]
        gy = gy_idx[i]
        dx = gx - rx_idx
        dy = gy - ry_idx
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            continue

        for s in range(steps):
            fx = rx_idx + (dx * s) // steps
            fy = ry_idx + (dy * s) // steps
            if 0 <= fx < MAP_SIZE and 0 <= fy < MAP_SIZE:
                if local_mask[fx, fy]:
                    continue
                log_odds[fx, fy] = min(lo_max, max(lo_min, log_odds[fx, fy] + lo_free))


def add_points_to_map(log_odds, points, x, y, rx_idx, ry_idx, theta_rad):
    """
    Добавляет lidar-точки в карту log_odds с учётом угла поворота робота.

    points: np.ndarray (N, 2) [угол°, дистанция м]
    x, y: позиция робота в глобальных координатах
    theta_rad: ориентация робота в радианах (0 вдоль оси X)
    """

    if points.size == 0:
        return

    # === 1. Переводим в локальные координаты лидара ===
    angles = np.deg2rad(points[:, 0])
    ranges = points[:, 1]
    lx = ranges * np.cos(angles)
    ly = ranges * np.sin(angles)

    # === 2. Поворот относительно угла робота ===
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    gx_world = x + lx * cos_t - ly * sin_t
    gy_world = y + lx * sin_t + ly * cos_t

    # === 3. В индексы карты ===
    gx_idx, gy_idx = to_index(gx_world, gy_world)

    # === 4. Обновление карты ===
    update_log_odds_numba(
        gx_idx.astype(np.int32),
        gy_idx.astype(np.int32),
        np.int32(rx_idx),
        np.int32(ry_idx),
        log_odds,
        lo_occ, lo_free, lo_min, lo_max
    )
