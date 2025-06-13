#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
from pytorch3d.ops import knn_points

def _copysign(a, b):
    return torch.where((a < 0) != (b < 0), -a, a)

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step<100:
            return 0.0
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling(s):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    return L

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
import torch

def inverse_rodrigues(rot_mats, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the axis-angle vectors for a batch of rotation matrices
        Parameters
        ----------
        rot_mats: torch.tensor Nx3x3
            array of N rotation matrices
        Returns
        -------
        rot_vecs: torch.tensor Nx3
            The axis-angle vectors for the given rotation matrices
    '''

    batch_size = rot_mats.shape[0]
    device = rot_mats.device

    # Ensure the matrices are close to valid rotation matrices by symmetrizing them
    rot_mats = (rot_mats + rot_mats.transpose(1, 2)) / 2

    # Trace of the rotation matrix, Tr(R) = 1 + 2 * cos(theta)
    trace = torch.einsum('bii->b', rot_mats)  # Trace of each 3x3 matrix in the batch
    cos = (trace - 1) / 2

    # Clamp cos values to avoid numerical issues (cosine values should be in [-1, 1])
    cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)

    # Calculate the angle (theta)
    angle = torch.acos(cos).unsqueeze(-1)

    # Calculate sin(theta) to compute the axis
    sin = torch.sin(angle)

    # If sin(theta) is too small, return zero vectors to avoid division by zero
    near_zero = sin.abs() < epsilon
    sin[near_zero] = 1.0  # Prevent division by zero in the following steps

    # Calculate the axis of rotation using the skew-symmetric part of the rotation matrix
    rx = (rot_mats[:, 2, 1] - rot_mats[:, 1, 2]) / (2 * sin).squeeze(-1)
    ry = (rot_mats[:, 0, 2] - rot_mats[:, 2, 0]) / (2 * sin).squeeze(-1)
    rz = (rot_mats[:, 1, 0] - rot_mats[:, 0, 1]) / (2 * sin).squeeze(-1)

    rot_dir = torch.stack([rx, ry, rz], dim=1)

    # For angles near zero, set rotation direction to zero
    rot_dir[near_zero.squeeze(-1)] = 0.0
    # print(rot_dir.shape,angle.shape)
    # Return axis-angle vectors (angle * axis)
    rot_vecs = angle * rot_dir

    return rot_vecs
def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def transform_obj(obj,r,trans,s):
    query_center = np.mean(obj,axis=0)
    # print(query_center)
    obj= obj - query_center
    obj= obj * s
    obj = np.matmul(obj,r)
    obj= obj + trans+query_center
    return obj

def knn_opacity_filter(points,scores):
    K = 3
    points=points.unsqueeze(0)
    knn_output = knn_points(points, points, K=K)

    # Get the indices of the 5 nearest neighbors for each point
    neighbor_indices = knn_output.idx  # Shape: (N, K)

    # Gather the scores of the nearest neighbors
    scores = scores[neighbor_indices]  # Shape: (N, K)

    # Compute the average score for each point (including itself if desired)
    scores = scores.mean(dim=1)  # Shape: (N,)

    # return average_scores

def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.split(quaternions, 1, dim=-1)
    two_s = 2.0 / torch.sum(quaternions * quaternions, dim=-1, keepdim=True)

    o = torch.stack([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j)
    ], dim=-1)
    return o.view(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def axis_angle_to_quaternion(axis_angle):
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions

def matrix_to_quaternion(matrix):
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), dim=-1)

def quaternion_to_axis_angle(quaternions):
    norms = torch.linalg.norm(quaternions[..., 1:], dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 0:1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

