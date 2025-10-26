import numpy as np
from numba import njit

from constants import MAP_ORIGIN, MAP_RESOLUTION, MAP_SIZE

lo_occ = 0.85
lo_free = -0.4
lo_min, lo_max = -5, 5

occupied_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)


def to_index(x, y):
    gx = MAP_ORIGIN + np.floor(x / MAP_RESOLUTION).astype(int)
    gy = MAP_ORIGIN + np.floor(y / MAP_RESOLUTION).astype(int)
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


def add_points_to_map(log_odds, points, x, y, rx_idx, ry_idx):
    angles = np.deg2rad(points[:, 0])
    ranges = points[:, 1]
    lx = ranges * np.cos(angles)
    ly = ranges * np.sin(angles)
    gx_idx, gy_idx = to_index(x + lx, y + ly)

    update_log_odds_numba(
        gx_idx.astype(np.int32),
        gy_idx.astype(np.int32),
        np.int32(rx_idx),
        np.int32(ry_idx),
        log_odds,
        lo_occ, lo_free, lo_min, lo_max
    )
