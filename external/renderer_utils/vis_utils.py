# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import trimesh
import numpy as np
from open3d import geometry, utility


def convert_trimesh_to_o3d(mesh):
    o3d_mesh = geometry.TriangleMesh()
    o3d_mesh.vertices = utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def convert_o3d_to_trimesh(mesh):
    t_mesh = trimesh.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.triangles))
    t_mesh.vertex_normals = np.array(mesh.vertex_normals)
    t_mesh.face_normals = np.array(mesh.triangle_normals)
    return t_mesh
