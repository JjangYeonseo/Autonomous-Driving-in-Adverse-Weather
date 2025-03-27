import open3d as o3d
import numpy as np

def load_pcd_as_bev(pcd_path, x_range=(-20, 20), y_range=(-10, 10), res=0.1):
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)
    except:
        return np.zeros((400, 200))  # fallback

    # 필터링
    x_min, x_max = x_range
    y_min, y_max = y_range
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
    )
    pts = pts[mask]

    # BEV 맵 크기
    H = int((x_max - x_min) / res)
    W = int((y_max - y_min) / res)
    bev = np.zeros((H, W))

    for x, y, z in pts:
        i = int((x - x_min) / res)
        j = int((y - y_min) / res)
        if 0 <= i < H and 0 <= j < W:
            bev[i, j] += 1

    bev = np.clip(bev / np.max(bev + 1e-5), 0, 1)
    return bev
