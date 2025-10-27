import numpy as np


SPLIT_THRESHOLD = 0.015
MIN_SEGMENT_POINTS = 8
MERGE_ANGLE_THRESHOLD = np.deg2rad(5)
DETECT_ANGLE_THRESHOLD = np.deg2rad(15)
MERGE_DIST_THRESHOLD = 0.04
GAP_FACTOR = 20
MIN_SEGMENT_LENGTH_RATIO = 0.2


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def split_and_merge(points):
    threshold = SPLIT_THRESHOLD
    min_pts = MIN_SEGMENT_POINTS

    def fit_line(p1, p2, pts):
        # p1, p2 — это [x, y, idx]
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-8:
            return np.zeros(len(pts))
        # расстояние от каждой точки до линии
        return np.abs(dy * pts[:, 0] - dx * pts[:, 1] + x2 * y1 - y2 * x1) / length

    def recursive_split(pts):
        if len(pts) < min_pts:
            return []
        dists = fit_line(pts[0], pts[-1], pts)
        idx = np.argmax(dists)
        if dists[idx] > threshold:
            # рекурсивное разбиение
            return recursive_split(pts[: idx + 1]) + recursive_split(pts[idx:])
        else:
            # сегмент с сохранением индексов
            return [{
                "start_point": pts[0, :2],
                "end_point": pts[-1, :2],
            }]

    return recursive_split(points)


def merge_segments(segs):
    angle_thresh = MERGE_ANGLE_THRESHOLD
    dist_thresh = MERGE_DIST_THRESHOLD

    segs_info = []
    for s in segs:
        p1 = np.array(s["start_point"])
        p2 = np.array(s["end_point"])
        vec = p2 - p1
        angle = np.arctan2(vec[1], vec[0])
        segs_info.append({
            "p1": p1,
            "p2": p2,
            "angle": angle,
        })

    used = [False] * len(segs_info)
    merged_final = []

    for i, seg in enumerate(segs_info):
        if used[i]:
            continue

        group_pts = [seg["p1"], seg["p2"]]
        used[i] = True

        for j, other in enumerate(segs_info):
            if used[j]:
                continue

            # Проверка на близость углов (учёт периодичности π)
            if abs((seg["angle"] - other["angle"] + np.pi / 2) % np.pi - np.pi / 2) < angle_thresh:
                # Проверка на близость концов
                dists = [
                    np.linalg.norm(seg["p1"] - other["p1"]),
                    np.linalg.norm(seg["p1"] - other["p2"]),
                    np.linalg.norm(seg["p2"] - other["p1"]),
                    np.linalg.norm(seg["p2"] - other["p2"]),
                ]
                if min(dists) < dist_thresh:
                    group_pts.extend([other["p1"], other["p2"]])
                    used[j] = True

        # Объединяем точки и находим крайние
        pts = np.vstack(group_pts)
        dir_vec = seg["p2"] - seg["p1"]
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        projections = pts @ dir_vec
        start = pts[np.argmin(projections)]
        end = pts[np.argmax(projections)]


        merged_final.append({
            "start_point": start,
            "end_point": end,
            "angle": np.arctan2(end[1] - start[1], end[0] - start[0]),
            "length": np.linalg.norm(end - start),
        })

    return merged_final


def split_by_gaps(points):
    factor = GAP_FACTOR
    dists = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    if len(dists) == 0:
        return [points]
    mean_dist = np.mean(dists)
    gaps = np.where(dists > mean_dist * factor)[0]
    if len(gaps) == 0:
        return [points]
    sections = []
    start = 0
    for g in gaps:
        sections.append(points[start:g + 1])
        start = g + 1
    sections.append(points[start:])
    return [s for s in sections if len(s) >= 3]



def detect_walls(range_points):
    angles_deg = range_points[:, 0]
    ranges = range_points[:, 1]

    # TODO: проверить как приходят углы, может не нужно дважды конвертировать
    angles_rad = np.deg2rad(angles_deg)
    x = ranges * np.cos(angles_rad)
    y = ranges * np.sin(angles_rad)

    points = np.stack((x, y), axis=1)
    groups = split_by_gaps(points)
    segments_all = []
    for g in groups:
        segs = split_and_merge(g)
        segments_all.extend(segs)

    return merge_segments(segments_all)

