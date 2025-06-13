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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from sdf import SDF
import torch.nn as nn

from pytorch3d.ops import knn_points

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def chamfer_distance(pcl1, pcl2):
    dist1, idx1, _ = knn_points(pcl1, pcl2, K=1)  # KNN for A -> B
    dist2, idx2, _ = knn_points(pcl2, pcl1, K=1)  # KNN for B -> A
    return dist1, dist2

def normalize(x:torch.Tensor):
    return (x- x.min()) / (x.max() - x.min())

def contact_compute(cf_distance,opacity,distance_threshold=0.001,opacity_threshold=0.05):
    cf_distance=cf_distance**2
    # print(cf_distance.shape)
    h_distance=cf_distance[:,:10475]
    o_distance=cf_distance[:,10475:]
    distance_score_h=normalize(h_distance)
    distance_score_o=normalize(o_distance)
    distance_score=torch.cat((distance_score_h,distance_score_o),dim=1)
    min_distance=distance_score.min(dim=1)[0]
    distance_threshold=distance_threshold*2


    opacity_h=opacity[:10475]
    opacity_o=opacity[10475:]
    opacity_score_h=normalize(opacity_h)
    opacity_score_o=normalize(opacity_o)
    opacity_score=torch.cat((opacity_score_h,opacity_score_o),dim=0)

    #save opacity
    # opacity_save=opacity_score.cpu().detach().numpy()[:10475]
    # np.save('opacity_save.npy',opacity_save)

    contact_region=(distance_score<distance_threshold) & (opacity_score<opacity_threshold)
    contact_region=contact_region[0]
    # contact_region=(opacity_score>=opacity_threshold)
    # print(contact_region.shape)
    return contact_region


def calculate_chamfer_distance(points_A, points_B):
    # Convert points to PyTorch tensors
    points_A_tensor = torch.tensor(points_A).to(torch.float32).unsqueeze(0)  # Shape [1, N, 3]
    points_B_tensor = torch.tensor(points_B).to(torch.float32).unsqueeze(0)  # Shape [1, M, 3]

    # Use pytorch3d's chamfer_distance function
    dist1, idx1, _ = knn_points(points_A_tensor, points_B_tensor, K=1)  # KNN for A -> B
    dist2, idx2, _ = knn_points(points_B_tensor, points_A_tensor, K=1)  # KNN for B -> A

    return idx1, idx2, dist1, dist2


def compute_vertex_normals(meshes):
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 1],
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 2],
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 0],
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
            dim=1,
        ),
    )

    return torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=1)

class HOCollisionLoss(nn.Module):
# adapted from multiperson (links, multiperson.sdf.sdf_loss.py)

    def __init__(self, smpl_faces, grid_size=32, robustifier=None,):
        super().__init__()
        self.sdf = SDF()
        self.register_buffer('faces', torch.tensor(smpl_faces.astype(np.int32)))
        self.grid_size = grid_size
        self.robustifier = robustifier


    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        # vertices: (n, 3)
        boxes = torch.zeros(2, 3, device=vertices.device)
        boxes[0, :] = vertices.min(dim=0)[0]
        boxes[1, :] = vertices.max(dim=0)[0]
        return boxes


    @torch.no_grad()
    def check_overlap(self, bbox1, bbox2):
        # check x
        if bbox1[0,0] > bbox2[1,0] or bbox2[0,0] > bbox1[1,0]:
            return False
        #check y
        if bbox1[0,1] > bbox2[1,1] or bbox2[0,1] > bbox1[1,1]:
            return False
        #check z
        if bbox1[0,2] > bbox2[1,2] or bbox2[0,2] > bbox1[1,2]:
            return False
        return True


    def forward(self, hoi_dict):
        # assume one person and one object
        # person_vertices: (n, 3), object_vertices: (m, 3)
        person_vertices, object_vertices = hoi_dict['smplx_v_centered'], hoi_dict['object_v_centered']
        object_vertices.retain_grad()
        b = person_vertices.shape[0]
        scale_factor = 0.2
        loss = torch.zeros(1).float().to(object_vertices.device)

        for b_idx in range(b):
            person_bbox = self.get_bounding_boxes(person_vertices[b_idx])
            object_bbox = self.get_bounding_boxes(object_vertices[b_idx])
            # print(person_bbox, object_bbox)
            if not self.check_overlap(person_bbox, object_bbox):
                return loss

            person_bbox_center = person_bbox.mean(dim=0).unsqueeze(0)
            person_bbox_scale = (1 + scale_factor) * 0.5 * (person_bbox[1] - person_bbox[0]).max()

            with torch.no_grad():
                person_vertices_centered = person_vertices[b_idx] - person_bbox_center
                person_vertices_centered = person_vertices_centered / person_bbox_scale
                assert(person_vertices_centered.min() >= -1)
                assert(person_vertices_centered.max() <= 1)
                phi = self.sdf(self.faces, person_vertices_centered.unsqueeze(0))
                assert(phi.min() >= 0)

            object_vertices_centered = (object_vertices[b_idx] - person_bbox_center) / person_bbox_scale
            object_vertices_grid = object_vertices_centered.view(1, -1, 1, 1, 3)
            phi_val = nn.functional.grid_sample(phi.unsqueeze(1), object_vertices_grid).view(-1)
            phi_val.retain_grad()
            cur_loss = phi_val
            if self.robustifier:
                frac = (cur_loss / self.robustifier) ** 2
                cur_loss = frac / (frac + 1)

            loss += cur_loss.sum()
            out_loss = loss.sum() / b
            # out_loss.backward()

            #print(object_vertices.grad, phi_val.grad, out_loss.grad,79879426951,torch.sum(object_vertices.grad))

        return out_loss