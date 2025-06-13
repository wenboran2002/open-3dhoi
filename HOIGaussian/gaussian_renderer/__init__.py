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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from hoi_scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh,SH2RGB,RGB2SH
from utils.loss_utils import chamfer_distance, contact_compute
from utils.visualize_utils import *
from utils.general_utils import inverse_rodrigues, batch_rodrigues,transform_obj,knn_opacity_filter,inverse_sigmoid
import open3d as o3d
import numpy as np
import smplx

model_type='smplx'
model_folder="./data/SMPLX_NEUTRAL.npz"
layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False,
             'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False,
             'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
smplx_model = smplx.create(model_folder, model_type=model_type,
                     gender='neutral',
                     num_betas=10,
                     num_expression_coeffs=10, use_pca=False, use_face_contour=True, **layer_arg)
zero_pose = torch.zeros((1, 3)).float().repeat(1, 1).cuda()
import cv2

model = smplx_model.cuda()


def render(iter, viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, return_smpl_rot=False, transforms=None, translation=None):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    means3D = pc.get_xyz
    means3D_o = means3D[10475:, :]
    means3D_h = means3D[0:10475, :]
    h_xyz = None

    ## wbr: transform
    if not pc.motion_offset_flag:
        _, means3D, _, transforms, _ = pc.coarse_deform_c2source(means3D_h[None], viewpoint_camera.smpl_param,
                                                                 viewpoint_camera.big_pose_smpl_param,
                                                 viewpoint_camera.big_pose_world_vertex[None])
    else:
        if transforms is None:
            # pose offset
            dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:]
            pose_out = pc.pose_decoder(dst_posevec)
            correct_Rs = pose_out['Rs']


            ## wbr: save smpl parameters after translation
            pose_org = viewpoint_camera.smpl_param['poses'].clone()
            batch_size = pose_org.shape[0]
            rot_mats = batch_rodrigues(pose_org.view(-1, 3)).view([batch_size, -1, 3, 3])
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(
                54, 3, 3)
            rev_list = []
            for mat in rot_mats_no_root:
                rev = cv2.Rodrigues(mat.detach().cpu().numpy().reshape(3, 3))
                rev_list.append(rev[0])
            save_body_pose = np.concatenate(rev_list, axis=0)
            save_body_pose = torch.tensor(save_body_pose).reshape(1, -1).cuda()

            # SMPL lbs weights
            lbs_weights = pc.lweight_offset_decoder(means3D_h[None].detach())
            lbs_weights = lbs_weights.permute(0, 2, 1)
            # transform points
            _, means3D_h, _, transforms, translation = pc.coarse_deform_c2source(means3D_h[None],
                                                                                 viewpoint_camera.smpl_param,
                                                                                 viewpoint_camera.big_pose_smpl_param,
                                                                                 viewpoint_camera.big_pose_world_vertex[
                                                                                     None], lbs_weights=lbs_weights,
                                                                                 correct_Rs=correct_Rs,
                                                                                 return_transl=return_smpl_rot)
            # print(h_xyz,means3D_h)
        else:
            correct_Rs = None
            means3D_h = torch.matmul(transforms, means3D_h[..., None]).squeeze(-1) + translation

    ## wbr: transform obj gaussian:
    ro = pc.get_transform_obj  # rotation
    transform_o = ro.unsqueeze(0).repeat(means3D_o.shape[0], 1, 1).unsqueeze(0)  # transform: for gaussian renderer
    transo = pc.get_transl_obj # translation
    scaleo = pc.get_scale_obj # scale
    means3D_o = pc.obj_transform_gs(means3D_o, ro, transo, scaleo)  # transform
    means3D_o = means3D_o.unsqueeze(0)

    ## wbr: save object transform
    pose={}
    ro_save= ro.detach().cpu().numpy()
    transl_save = transo.detach().cpu().numpy()
    scale_save = scaleo.detach().cpu().numpy()
    pose['ro'] = ro_save.tolist()
    pose['transl'] = transl_save.tolist()
    pose['scale'] = scale_save.tolist()
    means3D = torch.cat([means3D_h, means3D_o], dim=1)
    # print(means3D.shape)
    means3D = means3D.squeeze()
    means3D_o = means3D_o.squeeze()
    means3D_h = means3D_h.squeeze()
    means2D = screenspace_points
    opacity = pc.get_opacity
    transforms = torch.cat([transforms, transform_o], dim=1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features  # (10475,16,3)
    else:
        colors_precomp = override_color


    means2D_o = means2D[10475:, :]
    means2D_h = means2D[:10475, :]
    shs_o = shs[10475:, :]
    shs_h = shs[:10475, :]

    opacity_o = opacity[10475:]
    opacity_h = opacity[:10475]
    scales_o = scales[10475:]
    scales_h = scales[:10475]
    rotations_o = rotations[10475:]
    rotations_h = rotations[:10475]

    ## wbr contact region:
    knn_opacity_filter(means3D.clone(), opacity)
    h_distance, o_distance = chamfer_distance(means3D_h.unsqueeze(0), means3D_o.unsqueeze(0))
    cf_distance = torch.concatenate([h_distance, o_distance], dim=1).cuda()
    contact_region = contact_compute(cf_distance, opacity)
    pc.contact=contact_region

    ## render:
    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    rendered_o, _, depth_o, alpha_o = rasterizer(
        means3D=means3D_o,
        means2D=means2D_o,
        shs=shs_o,
        colors_precomp=colors_precomp,
        opacities=opacity_o,
        scales=scales_o,
        rotations=rotations_o,
        cov3D_precomp=cov3D_precomp)

    rendered_h, _, depth_h, alpha_h = rasterizer(
        means3D=means3D_h,
        means2D=means2D_h,
        shs=shs_h,
        colors_precomp=colors_precomp,
        opacities=opacity_h,
        scales=scales_h,
        rotations=rotations_h,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_o": rendered_o,
            "render_h": rendered_h,
            "render_alpha_o": alpha_o,
            "render_alpha_h": alpha_h,
            "render_depth": depth,
            "render_alpha": alpha,
            "depth_h":depth_h, "depth_o":depth_o,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "transforms": transforms,
            "translation": translation,
            "correct_Rs": correct_Rs,
            "obj_pose":pose,
            "h_param":save_body_pose
            }
