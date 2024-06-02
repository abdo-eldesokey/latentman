# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np


def rotation_matrix_x(angle):
    rotate_x = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    return rotate_x


def rotation_matrix_y(angle):
    rotate_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    return rotate_y


def rotation_matrix_z(angle):
    rotate_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    return rotate_z


def rotate_rotation_point_cloud(original_pc_batch, angle_sigma=0.06, angle_range=0.18):
    """
    Randomly rotates the point cloud in a mini batch in all three directions.
    :param original_pc_batch: mini batch data of the point cloud
    :param angle_sigma: std of the noise to be added
    :param angle_range: max angle to perturb
    :return: rotated point cloud
    """
    B = original_pc_batch.shape[0]
    rotated_points = np.zeros(original_pc_batch.shape, dtype=np.float32)
    for k in range(B):
        angles = angle_sigma * np.random.randn(3)
        angles = np.clip(angles, a_min=-angle_range, a_max=angle_range)
        rotate_x = rotation_matrix_x(angles[0])
        rotate_y = rotation_matrix_y(angles[1])
        rotate_z = rotation_matrix_z(angles[2])
        R = rotate_z @ rotate_y @ rotate_x
        rotated_points[k] = original_pc_batch[k].reshape((-1, 3)) @ R
    return rotated_points
