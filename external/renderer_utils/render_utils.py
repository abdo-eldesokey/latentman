# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from renderer_utils.point_utils import rotate_rotation_point_cloud

from open3d import utility, geometry
from matplotlib import pyplot as plt
import pyrr
from pyrender import (
    DirectionalLight,
    SpotLight,
    PointLight,
)
from sklearn.neighbors import KDTree
import trimesh
import pyrender
import numpy as np
from PIL import Image
from renderer_utils.vis_utils import (
    convert_o3d_to_trimesh,
)
import time
import os

CWD = os.getcwd()

SIZE = None
Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector
# draw_geometries = o3d.visualization.draw_geometries


class NormalShader:
    def __init__(self):
        """
        Custom shader function for rendering normal maps
        """
        self.program = None

    def get_program(self, vertex_shader=None, fragment_shader=None, geometry_shader=None, defines=None):
        # overwrite the shader files
        self.program = pyrender.shader_program.ShaderProgram(
            vertex_shader=f"{CWD}/external/renderer_utils/mesh_to_sdf/shader/mesh.vert",
            fragment_shader=f"{CWD}/external/renderer_utils/mesh_to_sdf/shader/mesh.frag",
            defines=defines,
        )
        return self.program


class Render:
    def __init__(self, size, camera_poses):
        self.size = size
        global SIZE
        SIZE = size

        if not isinstance(camera_poses, np.ndarray):
            print("preparing camera!")
            self.camera_poses = create_uniform_camera_poses(2.0)
        else:
            self.camera_poses = camera_poses

    def render(self, path, clean=True, intensity=6.0, mesh=None, only_render_images=False, color=False, correct_n=True):
        # segmentation label transfer
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = prepare_mesh(path, color=False, clean=clean)
            if not color:
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        try:
            if mesh.visual.defined:
                mesh.visual.material.kwargs["Ns"] = 1.0
        except:
            print("Error loading material!")

        mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        t1 = time.time()
        triangle_ids, normal_maps, depth_images, p_images = None, None, None, None
        if not only_render_images:
            # Normals are not normalized.
            triangle_ids, normal_maps, depth_images, p_images = correct_normals(
                mesh, self.camera_poses, correct=correct_n
            )
        rendered_images, _ = pyrender_rendering(
            mesh1,
            viz=False,
            light=True,
            camera_poses=self.camera_poses,
            intensity=intensity,
        )
        print(time.time() - t1)
        return triangle_ids, rendered_images, normal_maps, depth_images, p_images


def correct_normals(mesh, camera_poses, correct=True):
    rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    triangle_images = []
    normalmaps = []
    depth_maps = []
    p_images = []
    for i in range(camera_poses.shape[0]):
        a, b, index_tri, sign, p_image = trimesh_ray_tracing(
            mesh, camera_poses[i], resolution=SIZE, rayintersector=rayintersector
        )
        if correct:
            mesh.faces[index_tri[sign > 0]] = np.fliplr(mesh.faces[index_tri[sign > 0]])

        normalmap = render_normal_map(
            pyrender.Mesh.from_trimesh(mesh, smooth=False),
            camera_poses[i],
            SIZE,
            viz=False,
        )

        triangle_images.append(b)
        normalmaps.append(normalmap)
        depth_maps.append(a)
        p_images.append(p_image)
    return triangle_images, normalmaps, depth_maps, p_images


def all_rendering(mesh, camera_poses, light=False, viz=False, correct=True):
    rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    scene.add(mesh1)
    # renderer
    r = pyrender.OffscreenRenderer(SIZE, SIZE)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    # light
    if light:
        lights = init_light(scene, camera_poses[0])

    triangle_images = []
    normalmaps = []
    depth_maps = []
    color_images = []

    for i in range(camera_poses.shape[0]):
        a, b, index_tri, sign = trimesh_ray_tracing(
            mesh, camera_poses[i], resolution=SIZE, rayintersector=rayintersector
        )
        if correct:
            mesh.faces[index_tri[sign > 0]] = np.fliplr(mesh.faces[index_tri[sign > 0]])

        normalmap = render_normal_map(
            pyrender.Mesh.from_trimesh(mesh, smooth=False),
            camera_poses[i],
            SIZE,
            viz=False,
        )

        if light:
            update_light(scene, lights, camera_poses[i])

        if light:
            color, _ = r.render(scene)  # , flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES
        else:
            color, _ = r.render(scene)  # | pyrender.constants.RenderFlags.SKIP_CULL_FACES
            # , flags=pyrender.constants.RenderFlags.FLAT
        triangle_images.append(b)
        normalmaps.append(normalmap)
        depth_maps.append(a)
        color_images.append(color)
    return color_images, triangle_images, normalmaps, depth_maps


def normalize_mesh(mesh, mode="sphere", max_norm=None):
    if mode == "sphere":
        mesh.vertices = mesh.vertices - mesh.vertices.mean(0)
        if max_norm:
            scale = max_norm
        else:
            scale = np.linalg.norm(mesh.vertices, axis=1, ord=2).max()
        mesh.vertices = mesh.vertices / scale
    elif mode == "com":
        box = mesh.bounding_box_oriented
        mesh.vertices = mesh.vertices - box.vertices.mean(0)
        scale = np.linalg.norm(mesh.vertices, axis=1, ord=2).max()
        mesh.vertices = mesh.vertices / scale


def prepare_mesh(model_name, color=False, clean=False):
    mesh = trimesh.load(model_name, force="mesh")
    if clean:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()

    # normalize_mesh(mesh, "sphere")
    return mesh


def clean_using_o3d(mesh):
    mesh = convert_trimesh_to_o3d(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    p = mesh.sample_points_poisson_disk(10000, 1)
    # o3d.visualization.draw_geometries([mesh, p])
    return convert_o3d_to_trimesh(mesh)


def init_light(scene, camera_pose, intensity=6.0):
    direc_l = DirectionalLight(color=np.ones(3), intensity=intensity)
    spot_l = SpotLight(
        color=np.ones(3),
        intensity=1.0,
        innerConeAngle=np.pi / 16,
        outerConeAngle=np.pi / 6,
    )
    point_l = PointLight(color=np.ones(3), intensity=1)

    direc_l_node = scene.add(direc_l, pose=camera_pose)
    point_l_node = scene.add(point_l, pose=camera_pose)
    spot_l_node = scene.add(spot_l, pose=camera_pose)
    return spot_l_node, direc_l_node, point_l_node


def update_light(scene, lights, pose):
    for l in lights:
        scene.set_pose(l, pose)


def render_normal_map(mesh, camera_pose, size, viz=False):
    scene = pyrender.Scene(bg_color=(255, 255, 255))
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(size, size)
    renderer._renderer._program_cache = NormalShader()

    normals, depth = renderer.render(scene)  # flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES
    world_space_normals = normals / 255 * 2 - 1

    if viz:
        image = Image.fromarray(normals, "RGB")
        image.show()

    return world_space_normals


def pyrender_rendering(mesh, camera_poses, viz=False, light=False, intensity=6.0):
    # renderer
    r = pyrender.OffscreenRenderer(SIZE, SIZE)

    scene = pyrender.Scene()
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera = scene.add(camera, pose=camera_poses[0])
    # light
    if light:
        lights = init_light(scene, camera_poses[0], intensity=intensity)

    images = []
    depth_images = []
    for i in range(camera_poses.shape[0]):
        # scene.set_pose(camera, camera_poses[i])
        scene.set_pose(camera, camera_poses[0])
        if light:
            # update_light(scene, lights, camera_poses[i])
            update_light(scene, lights, camera_poses[0])

        if light:
            color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES)
        else:
            color, depth = r.render(
                scene,
                # flags=pyrender.constants.RenderFlags.FLAT
            )  # | pyrender.constants.RenderFlags.SKIP_CULL_FACES

        if viz:
            plt.figure()
            plt.imshow(color)
        images.append(color)
        depth_images.append(depth)
    return images, depth_images


def trimesh_ray_tracing(mesh, M, resolution=225, fov=60, rayintersector=None):
    # this is done to correct the mistake in way trimesh raycasting works.
    # in general this cannot be done.
    extra = np.eye(4)
    extra[0, 0] = 0
    extra[0, 1] = 1
    extra[1, 0] = -1
    extra[1, 1] = 0
    scene = mesh.scene()

    scene.camera_transform = M @ extra  # @ np.diag([1, -1,-1, 1]

    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = [resolution, resolution]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = fov, fov

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    # for each hit, find the distance along its vector
    index_tri, index_ray, points = rayintersector.intersects_id(
        origins, vectors, multiple_hits=False, return_locations=True
    )
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    sign = trimesh.util.diagonal_dot(mesh.face_normals[index_tri], vectors[index_ray])

    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]
    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)
    b = np.ones(scene.camera.resolution, dtype=np.int32) * -1
    p_image = (
        np.ones(
            [scene.camera.resolution[0], scene.camera.resolution[1], 3],
            dtype=np.float32,
        )
        * -1
    )
    depth_float = (depth - depth.min()) / depth.ptp()

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)

    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    b[pixel_ray[:, 0], pixel_ray[:, 1]] = index_tri
    p_image[pixel_ray[:, 0], pixel_ray[:, 1]] = points

    # show the resulting image
    return a, b, index_tri, sign, p_image


def create_uniform_camera_poses(distance=2):
    mesh = geometry.TriangleMesh()
    frontvectors = np.array(mesh.create_sphere(distance, 7).vertices)
    camera_poses = []
    for i in range(frontvectors.shape[0]):
        camera_pose = np.array(
            pyrr.Matrix44.look_at(eye=frontvectors[i], target=np.zeros(3), up=np.array([0.0, 1.0, 0])).T
        )
        camera_pose = np.linalg.inv(np.array(camera_pose))
        camera_poses.append(camera_pose)
    return np.stack(camera_poses, 0)


def create_random_uniform_camera_poses(distance=2, low_scale=0.51):
    mesh = geometry.TriangleMesh()
    frontvectors = np.array(mesh.create_sphere(distance, 7).vertices)
    frontvectors = rotate_rotation_point_cloud(np.expand_dims(frontvectors, 1), angle_range=0.25)

    # randomly scale the point clouds
    scales = np.random.uniform(low_scale, 1.0, frontvectors.shape[0])
    for b in range(frontvectors.shape[0]):
        frontvectors[b, :, :] = frontvectors[b, :, :] * scales[b]
    frontvectors = frontvectors[:, 0, :]

    camera_poses = []
    for i in range(frontvectors.shape[0]):
        camera_pose = np.array(
            pyrr.Matrix44.look_at(eye=frontvectors[i], target=np.zeros(3), up=np.array([0.0, 1.0, 0])).T
        )
        camera_pose = np.linalg.inv(np.array(camera_pose))
        camera_poses.append(camera_pose)
    return np.stack(camera_poses, 0)


def transfer_labels_shapenet_points_to_mesh(points, labels, mesh):
    points = points - points.mean(0)
    scale = np.linalg.norm(points, axis=1, ord=2).max()
    points = points / scale
    points = points @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T

    points_ = mesh.sample(5000)
    mesh.vertices = mesh.vertices - points_.mean(0)

    points_ = points_ - points_.mean(0)
    scale = np.linalg.norm(points_, axis=1, ord=2).max()
    mesh.vertices = mesh.vertices / scale
    _, points, _ = trimesh.registration.icp(points, mesh)

    _, indices = find_match(np.array(points), mesh.triangles_center)
    return labels[indices], points, mesh


def transfer_labels_shapenet_points_to_mesh_partnet(points, labels, mesh):
    points = points - points.mean(0)
    scale = np.linalg.norm(points, axis=1, ord=2).max()
    points = points / scale

    points_ = mesh.sample(5000)
    mesh.vertices = mesh.vertices - points_.mean(0)

    points_ = points_ - points_.mean(0)
    scale = np.linalg.norm(points_, axis=1, ord=2).max()
    mesh.vertices = mesh.vertices / scale
    _, points, _ = trimesh.registration.icp(points, mesh)

    _, indices = find_match(np.array(points), mesh.triangles_center)
    return labels[indices], points, mesh


def find_match(source, target, k=1):
    tree = KDTree(source)
    d, indices = tree.query(target, k=k)
    return d[:, 0], indices[:, 0]
