import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import time
from pathlib import Path

import trimesh
import pyrender
import imageio
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from pyrender import (
    PerspectiveCamera,
    OrthographicCamera,
    DirectionalLight,
    SpotLight,
    Mesh,
    Scene,
    OffscreenRenderer,
    RenderFlags,
    constants,
)
from IPython.display import Image, display

from config import get_motion_dir
from misc.io import create_dir
from misc import CWD


def make_dir_light(intensity=1.0):
    return DirectionalLight(color=np.ones(3), intensity=intensity)


def make_spot_light(intensity=10.0):
    return SpotLight(color=np.ones(3), intensity=intensity, innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)


def create_camera(scene, obj_scale, use_persp=False, cam_dist=2):
    """Compute camera pose given a pymesh scene that includes the objects to render
    Returns a pymesh camera object and the camera pose as a 4x4 numpy array
    """
    if scene.scale == 0.0:
        scene.scale = constants.DEFAULT_SCENE_SCALE
    s2 = 1.0 / np.sqrt(2.0)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    hfov = np.pi / 6.0
    dist = scene.scale / (cam_dist * np.tan(hfov))
    cam_pose[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + scene.centroid

    # cam_pose = create_uniform_camera_pose()

    # Set defaults as needed
    zfar = max(scene.scale * 10.0, constants.DEFAULT_Z_FAR)
    if scene.scale == 0:
        znear = constants.DEFAULT_Z_NEAR
    else:
        znear = min(scene.scale / 10.0, constants.DEFAULT_Z_NEAR)

    if use_persp:
        cam = PerspectiveCamera(yfov=np.pi / 3.0, znear=znear, zfar=zfar)
    else:
        xmag = ymag = obj_scale * scene.scale
        if scene.scale == 0:
            xmag = ymag = 1.0
        cam = OrthographicCamera(xmag=xmag, ymag=ymag, znear=znear, zfar=zfar)
    return cam, cam_pose


def rotation_transform(degrees):
    """
    Computes the rotation matrix given a list of 3 numbers in degrees (X Y Z)
    :param degrees:
    :return:
    """
    x_axis, y_axis, z_axis = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    r_x = trimesh.transformations.rotation_matrix(np.radians(degrees[0]), x_axis)
    r_y = trimesh.transformations.rotation_matrix(np.radians(degrees[1]), y_axis)
    r_z = trimesh.transformations.rotation_matrix(np.radians(degrees[2]), z_axis)
    transformation = trimesh.transformations.concatenate_matrices(r_x, r_y, r_z)
    return transformation


def init_renderer(resolution):
    # Render 3d object
    width = resolution[0]
    height = resolution[1]
    r = OffscreenRenderer(viewport_width=width, viewport_height=height)
    return r


def render_motion(renderer, action, scale=0.5, normalize="mean", use_persp=False, cam_dist=2, show_render=False):
    action_dir = get_motion_dir(action)
    obj_dir = action_dir + "/obj"

    rend_dir = str(obj_dir).replace("obj", "rendered_images")
    create_dir(rend_dir)
    depth_dir = str(obj_dir).replace("obj", "depth")
    create_dir(depth_dir)

    # Preload meshes
    objs = list(enumerate(sorted(Path(obj_dir).glob("*.obj"))))
    meshes = []
    model_pose = rotation_transform([0, 45, 90])

    smallest_y = 1000
    for idx, obj_path in objs:
        mesh = trimesh.load(str(obj_path))
        # mesh.apply_transform(model_pose)
        if normalize == "mean":
            mesh.vertices = mesh.vertices - mesh.vertices.mean(0)
        elif normalize == "min":
            mesh.vertices = mesh.vertices - mesh.vertices.min(0)

        min_y = mesh.vertices.min(0)[1]
        if min_y < smallest_y:
            smallest_y = min_y

        meshes.append(mesh)

    rendered_images = []
    depth_images = []
    for idx, mesh in tqdm(enumerate(meshes), total=len(objs)):
        # mesh = trimesh.load(str(obj_path))
        mesh.vertices[:, 1] = mesh.vertices[:, 1] - smallest_y

        # Convert mesh to pymesh
        model_mesh = Mesh.from_trimesh(mesh, smooth=False)

        # Light creation
        direc_l = make_dir_light(intensity=2.0)

        # Scene creation and initialization
        scene = Scene(ambient_light=np.array([0.2, 0.2, 0.2, 1.0]))
        scene.add(model_mesh, pose=model_pose)

        # Create a pyrender Node from the mesh
        # node = pyrender.Node(mesh=model_mesh)
        # scene.add_node(node)

        # Compute Camera
        if idx == 0:
            # Scene creation and initialization
            scene.add(model_mesh, pose=model_pose)
            cam, cam_pose = create_camera(scene, scale, use_persp=use_persp, cam_dist=cam_dist)

        # Add scene elements
        scene.add(cam, pose=cam_pose)
        scene.add(direc_l, pose=cam_pose)
        color, depth = renderer.render(scene)  # SKIP_CULL_FACES = 1024
        # face_ids, depth, p_image, normal_map = correct_normals(mesh, cam_pose)
        if use_persp:
            depth = ((depth / 5) * 255).astype(np.uint8)
        else:
            depth = ((depth * 10) * 255).astype(np.uint8)

        rendered_images.append(color)
        depth_images.append(depth)

        imageio.imwrite(f"{depth_dir}/frame_{idx:02}.png", depth)
        imageio.imwrite(f"{rend_dir}/frame_{idx:02}.png", color)

    imageio.mimsave(action_dir + "/render.gif", rendered_images, loop=100)
    imageio.mimsave(action_dir + "/depth.gif", depth_images, loop=100)

    if show_render:
        display(Image(action_dir + "/depth.gif"))
    return rendered_images, depth_images
