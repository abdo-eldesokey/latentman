from pathlib import Path
import math

import cv2
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip

from external.config import get_motion_dir

added_prompt = "best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, HDR"
# added_prompt = "van gough style, oil painting, detailed,"

negative_prompt = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"
)


def parse_exp_name(exp_path):
    assert Path(exp_path).exists(), "This experiment does not exist!"
    motion = exp_path.split("/")[-2]
    tmp = exp_path.split("/")[-1]
    prompt = " ".join((tmp.rsplit("_")[0:-4]))
    sd_seed = int(tmp.split("_s")[-1].split("_")[0])
    shift_x = int(tmp.split("_x")[-1].split("_")[0])
    shift_y = int(tmp.split("_y")[-1].split("_")[0])
    start_from = int(tmp.split("_f")[-1].split("_")[0])
    return motion, prompt, sd_seed, start_from, shift_x, shift_y


def load_motion(motion, prompt, seed, start_from, shift_x, shift_y, skip=2, batch_size=8):
    motion = motion
    start_from = start_from
    shift_x = shift_x  # +ve: right
    shift_y = shift_y  # +ve: down
    skip = skip

    print(f"==> Motion: {motion}, \t start_from: {start_from}, \t shift_x: {shift_x}, \t shift_y: {shift_y}")

    motion_dir = get_motion_dir(motion)

    ###### Create Output Directory
    ident = prompt.replace(" ", "_") + f"_s{seed}_x{shift_x}_y{shift_y}_f{start_from}"
    out_dir = Path(motion_dir) / ident
    for subdir in ["mp4", "png", "npy"]:
        if not (out_dir / subdir).exists():
            (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    ### Load DensePose
    dp_dir = Path(f"{motion_dir}/densepose")
    dps = []
    for idx, img_path in enumerate(sorted(dp_dir.glob("*.png"))):
        if idx < start_from:
            continue
        if idx % skip == 0:
            dp = cv2.imread(str(img_path))[..., ::-1]
            dp = np.roll(dp, shift=[shift_y, shift_x], axis=[0, 1])
            dps.append(dp)

    guidance_imgs = []

    ### Rendered Depth
    depth_dir = Path(f"{motion_dir}/depth")
    for idx, img_path in enumerate(sorted(depth_dir.glob("*.png"))):
        if idx < start_from:
            continue
        if idx % skip == 0:
            dep = cv2.imread(str(img_path))
            dep = np.roll(dep, shift=[shift_y, shift_x], axis=[0, 1])
            guidance_imgs.append(dep)

    print(len(guidance_imgs), len(dps))

    ### Make batches
    num_batches = math.ceil(len(guidance_imgs) / batch_size)
    batches = []
    for b in range(num_batches):
        batch = guidance_imgs[b * batch_size : (b + 1) * batch_size]
        batches.append(torch.from_numpy(np.stack(batch)).permute((0, 3, 1, 2)))
        break

    return dps, guidance_imgs, batches, motion_dir, out_dir


import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from misc.uv_mapping import get_xy_mapping, dp_downsample
from misc.io import create_dir


def get_dense_correspond(dps, new_res, reference_mode, motion_dir, num_dps, ds_method="nearest"):
    # Match each densepose with previous or first frame
    if "prev" in reference_mode:
        ref_dp_idx = list(range(num_dps - 1))  # Previous frame is reference
    else:
        ref_dp_idx = [0] * (num_dps - 1)  # First frame is reference

    print(f"Using `{reference_mode}` reference: {ref_dp_idx}")

    dp_dir = Path(f"{motion_dir}/densepose")

    # First frame
    dp1 = dp_downsample(dps[0].copy(), new_res, ds_method)

    new_dps = [dp1]
    xy_xy = [None]
    for idx in tqdm(range(1, num_dps)):
        ref_idx = ref_dp_idx[idx - 1]
        dp2 = dp_downsample(dps[idx], new_res, ds_method)
        new_dp2, x1, y1, x2, y2 = get_xy_mapping(dp1, dp2, mode=reference_mode)
        new_dps.append(new_dp2)
        xy_xy.append((x1, y1, x2, y2))

        if reference_mode == "prev":
            dp1 = dp2.copy()
        elif reference_mode == "prev_new":
            dp1 = new_dp2.copy()

    print(len(new_dps), len(xy_xy))
    return new_dps, xy_xy


from misc.uv_mapping import dp_to_uv_map


def uv_mse_error(images, dps_256, txt_path):
    images_256 = [cv2.resize(np.asarray(img), (256, 256), interpolation=cv2.INTER_LINEAR) for img in images]
    uv_maps_256 = np.stack([dp_to_uv_map(img, dp, resolution=256)[1].astype(np.uint8) for img, dp in zip(images_256, dps_256)])
    uv_maps_256_shift = np.roll(uv_maps_256, 1, 0)
    diff = np.power(uv_maps_256 - uv_maps_256_shift, 2)
    mask = np.bitwise_and(uv_maps_256 != 0, uv_maps_256_shift != 0)

    with open(txt_path, "w") as file:
        file.write(f"{diff.mean()}\n")
        file.write(f"{diff.sum() / np.count_nonzero(uv_maps_256)}\n")
        file.write(f"{(diff * mask).sum() / mask.sum()}\n")
    file.close()
    print(diff.mean())
    print(diff.sum() / np.count_nonzero(uv_maps_256))
    print((diff * mask).sum() / mask.sum())
    return (diff * mask).sum() / mask.sum()
