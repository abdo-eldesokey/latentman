from config import DENSEPOSE_PATH
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import imageio

from config import get_motion_dir
from misc.io import create_dir

from apply_net_new import InferenceAction


def load_densepose_model(model_backbone="resnet50"):
    print(f"Loading DensePose with Backbone `{model_backbone}`")
    if model_backbone == "resnet50":
        dp_args = SimpleNamespace(
            cfg=f"{DENSEPOSE_PATH}/configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml",
            model="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/model_final_0ed407.pkl",
            opts=[],
        )
    else:
        dp_args = SimpleNamespace(
            cfg=f"{DENSEPOSE_PATH}/configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml",
            model="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl",
            opts=[],
        )
    densepose = InferenceAction()
    densepose.init(dp_args)
    return densepose


def detect_densepose(densepose_model, images_list, action=None, mdm_seed=None):
    ##### Process
    print("Detecting DensePose ..")
    dp_out = densepose_model.execute(images_list)

    if action is not None:
        action_dir = get_motion_dir(action, mdm_seed)
        dp_dir = action_dir + "/densepose"
        create_dir(dp_dir)
    else:
        dp_dir = action_dir = None

    # dp_dir = str(obj_dir).replace("obj", "densepose")
    res = images_list[0].shape[0]
    # Parse the output and put it in an image
    dp_preds = []
    for idx, out in enumerate(dp_out):
        xyxy = out["pred_boxes_XYXY"][0].int().cpu().numpy()
        uv = out["pred_densepose"][0].uv.permute(1, 2, 0).cpu().numpy() * 255
        l = out["pred_densepose"][0].labels.cpu().numpy()[..., None]
        luv = np.concatenate((l, uv), -1)
        h, w, _ = l.shape
        dp_pred = np.zeros((res, res, 3))
        dp_pred[xyxy[1] : xyxy[1] + h, xyxy[0] : xyxy[0] + w] = luv
        dp_preds.append(dp_pred.astype(np.uint8))
        if dp_dir is not None:
            plt.imsave(f"{dp_dir}/frame_{idx:02}.png", dp_pred.astype(np.uint8))

    if dp_dir and action_dir:
        imageio.mimsave(action_dir + "/densepose.gif", dp_preds, duration=150, loop=100)

    print("Done DensePose!")

    return dp_preds
