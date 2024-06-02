import io
from PIL import Image
import imageio
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# import cv2
from textwrap import wrap


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(
    save_path,
    kinematic_tree,
    joints,
    title,
    dataset,
    figsize=(3, 3),
    fps=120,
    radius=3,
    vis_mode="default",
    gt_frames=[],
):
    matplotlib.use("Agg")

    title = "\n".join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints

    # preparation related to specific datasets
    if dataset == "kit":
        data *= 0.003  # scale for visualization
    elif dataset == "humanml":
        data *= 1.3  # scale for visualization
    elif dataset in ["humanact12", "uestc"]:
        data *= -1.5  # reverse axes, scale for visualization

    fig = plt.figure(figsize=(3, 3))
    plt.tight_layout()

    ax = plt.figure().add_subplot(projection="3d")
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    imgs = []
    for index in range(frame_number):
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot(
                data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color
            )
            # plt.scatter(data[index, chain, 0], data[index, chain, 1], linewidth=linewidth, color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=72)
        buf.seek(0)
        im = Image.open(buf)
        imgs.append(np.asarray(im))
        buf.close()

    imageio.mimwrite(save_path, imgs, duration=50, loop=100)
