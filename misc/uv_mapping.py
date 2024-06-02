import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import cv2
import torch
import numpy as np
import pandas as pd
from misc.io import create_parent
from misc import CWD


dp_uv_lookup_256_np = np.load(f"{CWD}/misc/dp_uv_lookup_256.npy")


def mapper(iuv, resolution=256):
    H, W, _ = iuv.shape
    valid_mask = iuv[:, :, 0] > 0  # Pixels where there are body parts
    iuv_valid = iuv[valid_mask]  # H x W
    x = np.linspace(0, W - 1, W).astype(np.uint16)
    y = np.linspace(0, H - 1, H).astype(np.uint16)
    xx, yy = np.meshgrid(x, y)  # Horizontal and vertical gradients
    xx_rgb = xx[valid_mask]  # col incides of iuv pixels with values
    yy_rgb = yy[valid_mask]  # row incides of iuv pixels with values
    # modify i to start from 0... 0-23
    i = iuv_valid[:, 0] - 1  # [0:23]
    u = iuv_valid[:, 1]  # [0:256]
    v = iuv_valid[:, 2]  # [0:256]
    uv_smpl = dp_uv_lookup_256_np[i.astype(np.uint16), v.astype(np.uint16), u.astype(np.uint16)]
    u_f = uv_smpl[:, 0] * (resolution - 1)
    v_f = (1 - uv_smpl[:, 1]) * (resolution - 1)
    return xx_rgb, yy_rgb, u_f, v_f


def crop_downsample(dp, new_res, orig_res=512):
    factor = orig_res / new_res
    valid_idx = np.argwhere(dp[..., 0] > 0)
    x1 = valid_idx[:, 0].min()
    x2 = valid_idx[:, 0].max()
    y1 = valid_idx[:, 1].min()
    y2 = valid_idx[:, 1].max()
    dp_crop = cv2.resize(dp[x1:x2, y1:y2], dsize=(0, 0), fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_NEAREST)
    hh, ww, ch = dp_crop.shape
    dp_low = np.zeros((new_res, new_res, ch))
    xx1, yy1 = map(lambda k: (k / factor).astype(np.uint8), [x1, y1])
    dp_low[xx1 : xx1 + hh, yy1 : yy1 + ww] = dp_crop
    return dp_low


def get_unique_mapping(ui, vi, x, y):
    uv = [(uu, vv) for uu, vv in zip(ui, vi)]
    uv_unique = pd.Series(uv).drop_duplicates()
    uv_unique_idx = pd.Series(uv).drop_duplicates().index
    un = np.asarray([p[0] for p in uv_unique])
    vn = np.asarray([p[1] for p in uv_unique])
    xn = np.asarray([x[i] for i in uv_unique_idx])
    yn = np.asarray([y[i] for i in uv_unique_idx])
    return un, vn, xn, yn


def dp_to_uv_map(data_image, dp_image, resolution=512, mode="linear"):
    x, y, u, v = mapper(dp_image, resolution)

    # Sparse
    uv_img_sparse = np.zeros_like(data_image) * np.nan
    ui = u.astype(np.uint16)
    vi = v.astype(np.uint16)

    uv_img_sparse[vi, ui, :] = data_image[y, x, :]

    # un, vn, xn, yn = get_unique_mapping(ui, vi, x, y)
    # uv_img_sparse[vn, un, :] = data_image[yn, xn, :]

    # A meshgrid of pixel coordinates
    nx, ny = resolution, resolution
    ch = data_image.shape[-1]
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))

    ##### mask for where the map is
    uv_mask = np.zeros((ny, nx))  # * np.nan
    uv_mask[np.ceil(v).astype(int), np.ceil(u).astype(int)] = 1
    uv_mask[np.floor(v).astype(int), np.floor(u).astype(int)] = 1
    uv_mask[np.ceil(v).astype(int), np.floor(u).astype(int)] = 1
    uv_mask[np.floor(v).astype(int), np.ceil(u).astype(int)] = 1
    kernel = np.ones((3, 3), np.uint8)
    uv_mask_d = uv_mask  # cv2.dilate(uv_mask, kernel, iterations=1)

    ###### dense
    img_values = data_image[y, x, :]
    uv_img_dense = griddata((v, u), img_values, (Y, X), method=mode)
    # uv_img_dense_ = griddata((v, u), img_values, (Y, X), method="nearest")
    # uv_img_dense[np.isnan(uv_img_dense)] = uv_img_dense_[np.isnan(uv_img_dense)]
    uv_img_dense = uv_img_dense * uv_mask_d[..., None].repeat(ch, -1)
    # uv_img_dense = uv_img_dense.astype(np.uint8)

    # ## get x,y coordinates   # Not needed for now
    # uv_y = griddata((v, u), y, (Y, X), method="linear")
    # uv_y_ = griddata((v, u), y, (Y, X), method="nearest")
    # uv_y[np.isnan(uv_y)] = uv_y_[np.isnan(uv_y)]
    # uv_x = griddata((v, u), x, (Y, X), method="linear")
    # uv_x_ = griddata((v, u), x, (Y, X), method="nearest")
    # uv_x[np.isnan(uv_x)] = uv_x_[np.isnan(uv_x)]

    # # update
    # coor_x = uv_x * uv_mask_d
    # coor_y = uv_y * uv_mask_d
    # coor_xy = np.stack([coor_x, coor_y], 2)
    return uv_img_sparse, uv_img_dense  # , uv_img_dense  # , coor_xy, uv_mask_d


def uv_map_to_dp(dp_image, uv_map, resolution=512, mode="linear"):
    # Get a maping from XY-domain of the original image with shape H,W,C
    # to the UV-domain with shape resolution,resolution,C
    x, y, u, v = mapper(dp_image, resolution)
    ui = u.astype(np.uint16)
    vi = v.astype(np.uint16)

    dp_sparse = np.zeros_like(uv_map) * np.nan

    dp_sparse[y, x, :] = uv_map[vi, ui, :]

    # un, vn, xn, yn = get_unique_mapping(ui, vi, x, y)
    # dp_sparse[yn, xn, :] = uv_map[vn, un, :]

    # Mask
    nx, ny = resolution, resolution
    nch = uv_map.shape[-1]
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    dp_mask = np.zeros((ny, nx)) * np.nan
    dp_mask[y, x] = 1
    kernel = np.ones((3, 3), np.uint8)
    dp_mask_d = dp_mask  ##cv2.dilate(dp_mask, kernel, iterations=1)
    # dense
    uv_values = uv_map[vi, ui, :]
    # yy, xx = np.where(dp_mask != 0)
    dp_dense = griddata((y, x), uv_values, (Y, X), method=mode)
    # dp_dense_ = griddata((y, x), uv_values, (Y, X), method="nearest")
    # dp_dense[np.isnan(dp_dense)] = dp_dense_[np.isnan(dp_dense)]
    dp_dense = dp_dense * dp_mask_d[..., None].repeat(nch, -1)
    # dp_dense = dp_dense.astype(np.uint16)
    return dp_sparse, dp_dense


def update_uv_map(orig_map, new_img, new_iuv, resolution=512, use_dense=False):
    if use_dense:
        uv_new = dp_to_uv_map(new_img, new_iuv, resolution)[1]
    else:
        uv_new = dp_to_uv_map(new_img, new_iuv, resolution)[0]

    mask = np.isnan(orig_map)
    orig_map[mask] = uv_new[mask]
    return orig_map


def dp_pred_to_img(data, img, idx):
    uv = data["pred_densepose"][idx].uv.permute(1, 2, 0).cpu().numpy() * 255
    i = data["pred_densepose"][idx].labels.cpu().numpy()[..., None]
    iuv_bbox = np.concatenate((i, uv), -1).astype(np.uint8)

    # Map iuv to the image coordinates
    iuv_img = np.zeros_like(img)
    bbox = data["pred_boxes_XYXY"][idx].numpy().astype(np.uint16)
    h, w, _ = iuv_bbox.shape
    iuv_img[bbox[1] : bbox[1] + h, bbox[0] : bbox[0] + w] = iuv_bbox
    return iuv_img


def generate_uv_maps(path):
    path = str(path)
    with open(path, "rb") as f:
        data = torch.load(f)

    scores = data["scores"].numpy()
    idxs = np.where(scores >= 0.99)[0].astype(np.uint16)
    if len(idxs) == 0:
        return
    img_path = path.replace("/densepose/pkl/", "/").replace("pkl", "jpg")
    guidance_path = path.replace(".pkl", ".png").replace("pkl", "guidance")
    dp_path = guidance_path.replace("guidance", "dp")
    create_parent(guidance_path)
    create_parent(dp_path)

    img = plt.imread(img_path)
    if len(img.shape) < 3:
        img = img[..., None].repeat(3, -1)

    iuv_img_total = np.zeros_like(img)
    guidance_total = np.zeros_like(img)
    # try:
    for idx in idxs:
        iuv_img = dp_pred_to_img(data, img, idx)

        if np.count_nonzero(iuv_img[..., 0]) < 4:
            continue
        uv_map_sparse, uv_map_dense = dp_to_uv_map(img, iuv_img)
        guidance = uv_map_to_dp(iuv_img, uv_map_sparse)
        guidance_total += guidance
        iuv_img_total += iuv_img

    if np.count_nonzero(iuv_img_total[..., 0]) < 4:
        return
    plt.imsave(guidance_path, guidance_total.astype(np.uint8))
    plt.imsave(dp_path, iuv_img_total.astype(np.uint8))
    # except:
    # print("failed for", path)
    # return


def pt_get_unique_mapping(ui, vi, x, y, device):
    uv = [(uu, vv) for uu, vv in zip(ui, vi)]
    uv_unique = pd.Series(uv).drop_duplicates()
    uv_unique_idx = pd.Series(uv).drop_duplicates().index
    un = torch.asarray([p[0] for p in uv_unique]).int().to(device)
    vn = torch.asarray([p[1] for p in uv_unique]).int().to(device)
    xn = torch.asarray([x[i] for i in uv_unique_idx]).int().to(device)
    yn = torch.asarray([y[i] for i in uv_unique_idx]).int().to(device)
    return un, vn, xn, yn


def pt_dp_to_uv_map(data_image: torch.Tensor, dp_image: np.ndarray, resolution: int = 512):
    """_summary_

    Args:
        data_image (torch.Tensor): C x H x W
        dp_image (np.ndarray): H x W x 3
        resolution (int, optional): _description_. Defaults to 512.

    Returns:
        _type_: _description_
    """
    x, y, u, v = mapper(dp_image, resolution)
    ui = u.astype(np.int16)
    vi = v.astype(np.int16)
    x = x.astype(int)
    y = y.astype(int)

    # Sparse
    uv_img_sparse = torch.zeros_like(data_image) * torch.nan

    uv_img_sparse[:, vi, ui] = data_image[:, y, x]

    # The mapping is one-to-many, so we need to find unique correpondences
    # to make the mapping invertible
    # device = data_image.device
    # un, vn, xn, yn = pt_get_unique_mapping(ui, vi, x, y, device)
    # uv_img_sparse[:, vn, un] = data_image[:, yn, xn]
    return uv_img_sparse


def pt_uv_map_to_dp(dp_image: np.ndarray, uv_map: torch.Tensor, resolution: int = 512):
    # Get a maping from XY-domain of the original image with shape H,W,C
    # to the UV-domain with shape resolution,resolution,C
    x, y, u, v = mapper(dp_image, resolution)
    ui = u.astype(np.int16)
    vi = v.astype(np.int16)
    x = x.astype(int)
    y = y.astype(int)
    dp_sparse = torch.zeros_like(uv_map) * torch.nan

    # dp_sparse[:, y, x] = uv_map[:, vi, ui]

    device = uv_map.device
    un, vn, xn, yn = pt_get_unique_mapping(ui, vi, x, y, device)
    dp_sparse[:, yn, xn] = uv_map[:, vn, un]
    return dp_sparse


def pt_update_uv_map(out_map: torch.Tensor, new_data: torch.Tensor, dp_img: np.ndarray, resolution: int = 512):
    new_map = pt_dp_to_uv_map(new_data, dp_img, resolution)
    mask = out_map == 0
    out_map[mask] = new_map[mask]
    return out_map, new_map


from scipy.spatial.distance import cdist
from lapsolver import solve_dense
import cv2
import numpy as np


def dp_downsample(dp, new_res, method="nearest"):
    if method == "nearest":
        dp_ds = cv2.resize(dp, (new_res, new_res), interpolation=cv2.INTER_NEAREST)
    elif method == "linear":
        dp_ds = cv2.resize(dp, (new_res, new_res), interpolation=cv2.INTER_LINEAR)
    elif method == "cubic":
        dp_ds = cv2.resize(dp, (new_res, new_res), interpolation=cv2.INTER_CUBIC)
        dp_ds[..., 0] = dp_ds[..., 0].clip(0, 23)
        dp_ds[..., [1, 2]] = dp_ds[..., [1, 2]].clip(0, 254)
    return dp_ds


def downsample(img, dp, new_res=512, method="nearest"):
    img_ds = cv2.resize(img, (new_res, new_res), interpolation=cv2.INTER_NEAREST)
    dp_ds = dp_downsample(dp, new_res, method)
    return img_ds, dp_ds


def get_centeroid(dp, part_idx):
    ys, xs = np.where(dp[..., 0] == part_idx)
    cy = ys.mean()
    cx = xs.mean()
    return cy, cx


def get_meshgrid(res):
    x = np.arange(res, dtype=int)
    y = np.arange(res, dtype=int)
    xx, yy = np.meshgrid(x, y)
    pos = np.stack([yy, xx], -1)
    return pos


def get_pos_encoding(dp, part_idx):
    mask = dp[..., 0] == part_idx
    yy, xx = np.where(mask)
    x1, x2 = xx.min(), xx.max() + 1
    y1, y2 = yy.min(), yy.max() + 1
    sx = x2 - x1
    sy = y2 - y1
    x = np.linspace(-1, 1, sx)
    y = np.linspace(-1, 1, sy)
    xm, ym = np.meshgrid(x, y)
    pos = np.ones_like(dp[..., :2]).astype(float)
    pos[y1:y2, x1:x2, 1] = xm * (sx / 2)
    pos[y1:y2, x1:x2, 0] = ym * (sy / 2)
    pos[np.bitwise_not(mask)] = np.nan
    return pos


def get_dist_from_center(dp, part_idx):
    part_pos = get_pos_encoding(dp, part_idx)  # [yy,xx]
    center = np.asarray([0, 0])[None]
    mask = np.bitwise_not(np.isnan(part_pos[..., 0]))
    part_pos_masked = part_pos[mask]
    dist = cdist(part_pos_masked, center, metric="minkowski", p=1)[..., 0]
    new_part_pos = np.zeros_like(part_pos[..., 0])
    new_part_pos[mask] = dist
    scale = 127.5 / new_part_pos.max()
    return new_part_pos * (scale / 2)


def get_part(dp, part_idx):
    y, x = np.where(dp[..., 0] == part_idx)
    part_dp = np.zeros_like(dp)
    part_dp[y, x] = dp[y, x]
    return part_dp


def get_xy_mapping(dp1, dp2, mode="first"):
    res = dp1.shape[0]
    pos = get_meshgrid(res)  # [yy, xx]

    def get_dp_vec(dp, part_idx):
        part_dp = get_part(dp, part_idx)
        pos_local = get_dist_from_center(dp, part_idx)[..., None]
        dp_mat = np.concatenate([part_dp, pos, pos_local], -1)
        mask = dp_mat[..., 0] == part_idx
        dp_masked = dp_mat[mask]
        dp_masked[:, [1, 2]] -= dp_masked[:, [1, 2]].mean(0)  # zero-mean u,v
        return dp_masked

    if mode == "first":
        new_dp2 = np.zeros_like(dp2)
    else:
        new_dp2 = dp2.copy()

    xs1, ys1, xs2, ys2 = [], [], [], []
    for part in range(1, 25):
        if not part in dp1[..., 0] or not part in dp2[..., 0]:
            continue

        dp1_masked = get_dp_vec(dp1, part_idx=part)
        dp2_masked = get_dp_vec(dp2, part_idx=part)

        cost = cdist(dp1_masked[:, [1, 2, 5]], dp2_masked[:, [1, 2, 5]], metric="minkowski", p=2)

        dp1_match, dp2_match = solve_dense((cost))

        y1 = dp1_masked[dp1_match, 3].astype(int)
        x1 = dp1_masked[dp1_match, 4].astype(int)

        y2 = dp2_masked[dp2_match, 3].astype(int)
        x2 = dp2_masked[dp2_match, 4].astype(int)
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)
        new_dp2[y2, x2] = dp1[y1, x1]

    xs1 = np.concatenate(xs1)
    ys1 = np.concatenate(ys1)
    xs2 = np.concatenate(xs2)
    ys2 = np.concatenate(ys2)

    return new_dp2, xs1, ys1, xs2, ys2
