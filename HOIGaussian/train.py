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
import cv2
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render
import sys
from hoi_scene import HOIDataset, GaussianModel
from hoi_optimizer import HOIOptimizer
from utils.general_utils import safe_state
from utils.visualize_utils import visualize_imgs
import uuid
import imageio
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import fov2focal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from smpl.smpl_numpy import SMPL
import smplx
import open3d as o3d
import random
import json
from scipy.ndimage.morphology import distance_transform_edt

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips

loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

def transform_obj(obj,r,trans,s):
    query_center = np.mean(obj,axis=0)
    # print(query_center)
    obj= obj - query_center
    obj= obj * s
    obj = np.matmul(obj,r)
    obj= obj + trans+query_center
    return obj

def save_result_hoi(save_dir, h_pose, o_pose,obj_path):
    h_pose = h_pose.detach().cpu().numpy().tolist()

    param=json.load(open(obj_path+'/smplx_parameters.json'))
    obj_mesh= o3d.io.read_triangle_mesh(obj_path+'/obj_pcd_h_align.obj')
    h_mesh=o3d.io.read_triangle_mesh(obj_path+'/h_mesh.obj')
    obj_verts_org=np.asarray(obj_mesh.vertices)
    cam_trans=param['cam_trans']
    ro = o_pose['ro']
    transl = o_pose['transl']
    scale = o_pose['scale']



    obj_verts = obj_verts_org + cam_trans

    # cam_transl=np.asarray(cam_trans)
    exp_verts = transform_obj(obj_verts, ro, transl, scale)
    exp_verts -= cam_trans

    obj_mesh.vertices = o3d.utility.Vector3dVector(exp_verts)
    o3d.io.write_triangle_mesh(save_dir + '/obj_mesh.obj', obj_mesh)
    o3d.io.write_triangle_mesh(save_dir + '/h_mesh.obj', h_mesh)


    with open(save_dir + '/h_pose.json', 'w') as f:
        json.dump(h_pose, f)
    with open(save_dir + '/o_pose.json', 'w') as f:
        json.dump(o_pose, f)


def get_img_from_cam(cam, world_points):
    Tr = cam.T
    R = cam.R
    K = cam.K

    cam_points = np.matmul(world_points, R)
    cam_points += Tr.T
    imgvec = np.dot(cam_points, K.T)
    img_points = np.zeros((imgvec.shape[0], 2))

    for index in range(imgvec.shape[0]):
        vec = imgvec[index]
        z = vec[2]

        img_points[index][0] = vec[0] / z
        img_points[index][1] = vec[1] / z

    return img_points.astype(np.int32)


def calculate_center(mask):
    num_valid_pixels = mask.sum()
    if num_valid_pixels > 0:
        y_indices, x_indices = torch.meshgrid(torch.arange(mask.size(0)), torch.arange(mask.size(1)), indexing='ij')
        x_indices = x_indices.cuda()
        y_indices = y_indices.cuda()
        center_y = (y_indices * mask).sum() / num_valid_pixels
        center_x = (x_indices * mask).sum() / num_valid_pixels
        return center_y, center_x
    else:
        return None, None


def calculate_size(mask):
    return mask.sum()


def size_centre_loss(mask1, mask2):
    centery_1, centerx_1 = calculate_center(mask1)
    centery_2, centerx_2 = calculate_center(mask2)
    # print(centery_1, centerx_1,centery_2, centerx_2)
    if centery_2 is None or centerx_2 is None or centery_1 is None or centerx_1 is None:
        return 0, 0
    size1 = calculate_size(mask1)
    size2 = calculate_size(mask2)
    center_loss = ((centery_1 - centery_2) ** 2 + (centerx_1 - centerx_2) ** 2) ** 0.5
    size_loss = torch.abs(size1 - size2)
    return center_loss, size_loss


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
              save_dir):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)

    scene = HOIDataset(dataset, gaussians)

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    elapsed_time = 0
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Start timer
        start_time = time.time()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        ## zero grad
        gaussians.optimizer.zero_grad(set_to_none=True)

        viewpoint_cam = viewpoint_stack[0]

        pipe.debug = True
        render_pkg = render(iteration, viewpoint_cam, gaussians, pipe, background)

        image, alpha, viewspace_point_tensor, visibility_filter, radii, image_o, image_h, alpha_o, alpha_h, \
            depth_h, depth_o, obj_pose, h_pose = (
            render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"],
            render_pkg["render_o"], render_pkg["render_h"],
            render_pkg["render_alpha_o"], render_pkg["render_alpha_h"], render_pkg['depth_h'],
            render_pkg['depth_o'], render_pkg['obj_pose'], render_pkg['h_param'])

        Ll1,Ll1_h,Ll1_o, mask_loss,mask_loss_h, mask_loss_o, ssim_loss, ssim_loss_h, ssim_loss_o, lpips_loss, lpips_loss_h, lpips_loss_o = (
            None, None, None, None, None, None, None, None, None, None, None, None)
        if iteration < 100:

            # gaussian Loss
            gt_image = viewpoint_cam.original_image.cuda()
            gt_image_h = viewpoint_cam.original_image_h.cuda()
            gt_image_o = viewpoint_cam.original_image_o.cuda()

            bkgd_mask = viewpoint_cam.bkgd_mask.cuda()

            bkgd_mask_o = viewpoint_cam.bkgd_mask_o.cuda()

            bkgd_mask_h = viewpoint_cam.bkgd_mask_h.cuda()

            centre_loss, size_loss = size_centre_loss(alpha_o.squeeze(0), bkgd_mask_o.squeeze(0))
            alpha_o = alpha_o.masked_fill(~bkgd_mask_o.bool(), 0)
            image_o = image_o.masked_fill(~bkgd_mask_o.bool(), 0)

            bound_mask = viewpoint_cam.bound_mask.cuda()
            Ll1 = l1_loss(image.permute(1, 2, 0)[bound_mask[0] == 1], gt_image.permute(1, 2, 0)[bound_mask[0] == 1])
            Ll1_o = l1_loss(image_o.permute(1, 2, 0)[bound_mask[0] == 1], gt_image_o.permute(1, 2, 0)[bound_mask[0] == 1])
            Ll1_h = l1_loss(image_h.permute(1, 2, 0)[bound_mask[0] == 1], gt_image_h.permute(1, 2, 0)[bound_mask[0] == 1])
            mask_loss = l2_loss(alpha[bound_mask == 1], bkgd_mask[bound_mask == 1])
            mask_loss_o = l2_loss(alpha_o[bound_mask == 1], bkgd_mask_o[bound_mask == 1])
            mask_loss_h = l2_loss(alpha_h[bound_mask == 1], bkgd_mask_h[bound_mask == 1])

            # crop the object region
            x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
            img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
            img_pred_o = image_o[:, y:y + h, x:x + w].unsqueeze(0)
            img_pred_h = image_h[:, y:y + h, x:x + w].unsqueeze(0)
            img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
            img_gt_o = gt_image_o[:, y:y + h, x:x + w].unsqueeze(0)
            img_gt_h = gt_image_h[:, y:y + h, x:x + w].unsqueeze(0)

            #ssim loss
            ssim_loss = ssim(img_pred, img_gt)
            ssim_loss_o = ssim(img_pred_o, img_gt_o)
            ssim_loss_h = ssim(img_pred_h, img_gt_h)
            # lipis loss
            lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)
            lpips_loss_o = loss_fn_vgg(img_pred_o, img_gt_o).reshape(-1)
            lpips_loss_h = loss_fn_vgg(img_pred_h, img_gt_h).reshape(-1)


            gaussian_loss = (Ll1 * 0.3 + Ll1_o + Ll1_h +
                             0.05 * mask_loss + 0.1 * mask_loss_o + 0.1 * mask_loss_h
                             + 0.005 * (1.0 - ssim_loss) + 0.01 * (1.0 - ssim_loss_o) +
                             0.01 * (1.0 - ssim_loss_h) + 0.005 * lpips_loss +
                             0.01 * lpips_loss_o + 0.01 * lpips_loss_h)

            loss = gaussian_loss + size_loss * 0.01 + 0.01 * centre_loss

            loss.backward()

        ## inite hoi loss
        contact_loss, depth_loss, collision_loss = None, None, None
        if iteration >= 100 and iteration < 160:

            hoi_optim = HOIOptimizer(gaussians, viewpoint_cam, render_pkg)
            hoi_loss_dict, hoi_loss_weights = hoi_optim(opt)

            if hoi_loss_dict['loss_contact'] is not None:
                contact_loss = hoi_loss_dict['loss_contact'] * hoi_loss_weights['lw_contact']
            else :
                contact_loss = None

            if hoi_loss_dict['loss_depth'] is not None:
                depth_loss = hoi_loss_dict['loss_depth'] * hoi_loss_weights['lw_depth']
            else:
                depth_loss = None

            if hoi_loss_dict['loss_collision'] is not None:
                collision_loss = hoi_loss_dict['loss_collision'] * hoi_loss_weights['lw_collision']
            else:
                collision_loss = None

            if contact_loss != 0 and contact_loss is None:
                contact_loss.backward(retain_graph=True)

            if depth_loss != 0 and depth_loss != None:
                depth_loss.backward(retain_graph=True)

            if collision_loss != 0 and collision_loss is not None:
                collision_loss.backward(retain_graph=True)

            for name, param in gaussians.get_named_parameters().items():
                if ((name == 'scale_obj') or (name == 'x_angle') or (name == 'y_angle') or (name == 'z_angle')):
                    param.grad = None
            if iteration == 159:
                object_path=f'{args.data_path}/{dataset.file_name}/'
                save_result_hoi(save_dir, h_pose, obj_pose, object_path)
                return

        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        if (iteration in testing_iterations):
            print("[Elapsed time]: ", elapsed_time)

        iter_end.record()

        with torch.no_grad():
            if iteration < 100:
                Ll1_loss_for_log = Ll1.item()
                mask_loss_for_log = mask_loss.item()
                ssim_loss_for_log = ssim_loss.item()
                lpips_loss_for_log = lpips_loss.item()
                contact_loss_for_log=None
                depth_loss_for_log=None
                collision_loss_for_log=None
            if iteration >= 100 and iteration < 160:
                contact_loss_for_log = contact_loss.item()
                depth_loss_for_log = depth_loss.item()
                collision_loss_for_log = collision_loss.item()
                Ll1_loss_for_log=0
                mask_loss_for_log=0
                ssim_loss_for_log=0
                lpips_loss_for_log=0

            if iteration % 5 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}",
                                          "mask Loss": f"{mask_loss_for_log:.{2}f}",
                                          "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}",
                                          "contact": contact_loss_for_log,
                                          "depth": depth_loss_for_log, "collision": collision_loss_for_log
                                          })
                progress_bar.update(5)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, Ll1_o, Ll1_h, mask_loss, mask_loss_o, mask_loss_h, ssim_loss,
                            ssim_loss_o, ssim_loss_h, lpips_loss, lpips_loss_o, lpips_loss_h, contact_loss, depth_loss, collision_loss,
                            iter_start.elapsed_time(iter_end))

            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
        # Start timer
        start_time = time.time()
        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", args.exp_name)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, Ll1_o, Ll1_h, mask, mask_o, mask_h, ssim, ssim_o, ssim_h, lpips, lpips_o,
                    lpips_h, contact_loss, depth_loss, collision_loss, elapsed):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item() if Ll1 is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', ssim.item() if ssim is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/lpips_loss', lpips.item() if lpips is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_o', Ll1_o.item() if Ll1_o is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss_o', ssim_o.item() if ssim_o is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/lpips_loss_o', lpips_o.item() if lpips_o is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_h', Ll1_h.item() if Ll1_h is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss_h', ssim_h.item() if ssim_h is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/lpips_loss_h', lpips_h.item() if lpips_h is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/mask', mask.item() if mask is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/mask_o', mask_o.item() if mask_o is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/mask_h', mask_h.item() if mask_h is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/contact_loss', contact_loss.item() if contact_loss is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item() if depth_loss is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/collision_loss', collision_loss.item() if collision_loss is not None else 0, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_200, 2_000, 3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_200, 2_000, 3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--base_dir", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    save_dir = os.path.join("output/",  args.exp_name, args.file_name)
    os.makedirs(save_dir, exist_ok=True)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
              save_dir)

    # All done
    print("\nTraining complete.")

