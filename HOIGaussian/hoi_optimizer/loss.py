import numpy as np
import torch
import torch.nn as nn
from utils.loss_utils import chamfer_distance, calculate_chamfer_distance, compute_vertex_normals, l2_loss, HOCollisionLoss
# import smplx
import open3d as o3d
from utils.general_utils import transform_obj
from pytorch3d.ops import knn_points
from pytorch3d.structures.meshes import Meshes
from utils.general_utils import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, PerspectiveCameras,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)
import cv2


class Losses(object):
    def __init__(self):
        pass

    def compute_contact_loss(self, hverts, overts,h_contact,o_contact):
        h_v_contact = hverts[h_contact]
        h_v_contact=h_v_contact.unsqueeze(0)
        o_v_contact = overts[o_contact]
        o_v_contact=o_v_contact.unsqueeze(0)
        hdist,odist=chamfer_distance(h_v_contact,o_v_contact)
        ho_distance = hdist.mean() + odist.mean()

        return {"loss_contact": ho_distance}


    def compute_ordinal_depth_loss(self, mask_h, mask_o, h_verts, o_verts, o_faces, h_faces,cam):
        loss = torch.tensor(0.0).float().cuda()
        num_pairs = 2
    
        human_faces = torch.from_numpy(h_faces.astype(np.int64)).float().cuda()
        hmesh = Meshes(verts=h_verts.unsqueeze(0), faces=human_faces.unsqueeze(0))

        obj_faces = torch.from_numpy(o_faces.astype(np.int64)).float().cuda()
        omesh = Meshes(verts=o_verts.unsqueeze(0), faces=obj_faces.unsqueeze(0))

        width = cam.image_width
        height = cam.image_height
        P = cam.projection_matrix.t().unsqueeze(0)
        R = torch.FloatTensor(cam.R).cuda().unsqueeze(0)
        T = torch.FloatTensor(cam.T).cuda().t()
        focal = torch.FloatTensor(cam.cam_param['focal'] / 2).cuda().unsqueeze(0)
        princpt = torch.FloatTensor(cam.cam_param['princpt'] / 2).cuda().unsqueeze(0)
        cameras = PerspectiveCameras(R=R, T=T, focal_length=-focal, 
                                     principal_point=princpt, image_size=((height, width),) ,
                                    in_ndc=False, device='cuda')

        raster_settings = RasterizationSettings(
                            image_size=(height, width),
                            blur_radius=0.0,
                            faces_per_pixel=1,
                            # max_faces_per_bin=30000,
                            # bin_size = 8
        )

        rasterizer = MeshRasterizer(
                        cameras=cameras,
                        raster_settings=raster_settings
                        )

        depth_h = rasterizer(hmesh).zbuf.squeeze(0)
        depth_h[depth_h == -1] = 10000
        depth_o = rasterizer(omesh).zbuf.squeeze(0)
        depth_o[depth_o == -1] = 10000

        sil_h_pred = (depth_h < 10000)
        sil_o_pred = (depth_o < 10000)
        # print(sil_o)
        # sil_o_img = np.array(sil_o_pred.cpu().detach().numpy()* 255, dtype=np.uint8)

        has_pred = sil_h_pred & sil_o_pred
        front_h_gt = mask_h & (~mask_o)
        front_h_gt = front_h_gt.permute(1,2,0)
        front_o_pred = depth_o < depth_h
        mh = front_h_gt & front_o_pred & has_pred

        front_o_gt = mask_o & (~mask_h)
        front_o_gt = front_o_gt.permute(1,2,0)
        front_h_pred = depth_h < depth_o

        mo = front_o_gt & front_h_pred & has_pred
        # print(mh.sum(),mo.sum())
        if mh.sum() == 0 and mo.sum() == 0:
            loss = torch.Tensor([0.0]).cuda()
            # print('1')
        elif mh.sum() == 0 and mo.sum() != 0:
            dists = torch.clamp(depth_h - depth_o, min=0.0, max=2.0)
            loss += torch.sum(torch.log(1 + torch.exp(dists))[mo])
            loss /= num_pairs
            # print('2')
        elif mo.sum() == 0 and mh.sum() != 0:
            dists = torch.clamp(depth_o - depth_h, min=0.0, max=2.0)
            loss += torch.sum(torch.log(1 + torch.exp(dists))[mh])
            loss /= num_pairs
            # print('3')
        elif mh.sum() != 0 and mo.sum() != 0:
            distsh = torch.clamp(depth_h - depth_o, min=0.0, max=2.0)
            distso = torch.clamp(depth_o - depth_h, min=0.0, max=2.0)
            loss += torch.sum(torch.log(1 + torch.exp(distsh))[mh]) + torch.sum(torch.log(1 + torch.exp(distso))[mo])
            loss /= num_pairs
            # print('4')

        return {"loss_depth": loss}
    

    def compute_l1_depth_loss(self, mask_h, mask_o, h_verts, o_verts, o_faces, cam):
        loss = torch.tensor(0.0).float().cuda()
        num_pairs = 2
    
        human_faces = torch.from_numpy(h_faces.astype(np.int64)).float().cuda()
        hmesh = Meshes(verts=h_verts.unsqueeze(0), faces=human_faces.unsqueeze(0))

        obj_faces = torch.from_numpy(o_faces.astype(np.int64)).float().cuda()
        omesh = Meshes(verts=o_verts.unsqueeze(0), faces=obj_faces.unsqueeze(0))

        width = cam.image_width
        height = cam.image_height
        P = cam.projection_matrix.t().unsqueeze(0)
        R = torch.FloatTensor(cam.R).cuda().unsqueeze(0)
        T = torch.FloatTensor(cam.T).cuda().t()
        focal = torch.FloatTensor(cam.cam_param['focal'] / 2).cuda().unsqueeze(0)
        princpt = torch.FloatTensor(cam.cam_param['princpt'] / 2).cuda().unsqueeze(0)
        cameras = PerspectiveCameras(R=R, T=T, focal_length=-focal, 
                                     principal_point=princpt, image_size=((height, width),) ,
                                    in_ndc=False, device='cuda')
        # cameras = PerspectiveCameras(R=R, T=T, K=P, device='cuda')

        raster_settings = RasterizationSettings(
                            image_size=(height, width),
                            blur_radius=0.0,
                            faces_per_pixel=1,
                            # max_faces_per_bin=20000
                            )

        rasterizer = MeshRasterizer(
                        cameras=cameras,
                        raster_settings=raster_settings
                        )

        depth_h = rasterizer(hmesh).zbuf.squeeze(0)
        # depth_h[depth_h == -1] = 10000
        depth_o = rasterizer(omesh).zbuf.squeeze(0)
        # depth_o[depth_o == -1] = 10000
    
        gt_depth = np.load('/hdd/boran/contact_dataset_selection/3dhoi_Dataset_s1/car/HICO_train2015_00004083/depth.npy')
        gt_depth = cv2.resize(gt_depth, (width, height), interpolation=cv2.INTER_AREA)
        gt_depth = torch.from_numpy(gt_depth).cuda()

        gt_nums = gt_depth[~(mask_h.squeeze(0) | mask_o.squeeze(0))]
        gt_nums = gt_nums.cpu().numpy()
        dmin = gt_nums.min()
        dmax = gt_nums.max()

        # gt_depth[~(mask_h.squeeze(0) | mask_o.squeeze(0))] = -1
        # gt_depth = (depth - np.min(depth)) / np.ptp(depth)

        pass

    def compute_ho_distance_loss(self, hverts, overts):

        hpoints = hverts.unsqueeze(0)
        opoints = overts.unsqueeze(0)
        hdist, odist = chamfer_distance(hpoints, opoints)

        ho_distance = torch.min(hdist)

        return {"loss_ho_dist": ho_distance}
    
    def compute_collision_loss(self, smplxmodel, viewpoint_camera, correct_Rs, overts):

        overts = overts.unsqueeze(0)
        pose_org = viewpoint_camera.smpl_param['poses'].clone()
        #print(pose_org.shape)
        batch_size = pose_org.shape[0]
        rot_mats = axis_angle_to_matrix(pose_org.view(-1, 3)).view([batch_size, -1, 3, 3])
        #print('rot_mats',rot_mats.shape)
        rot_mats_no_root = rot_mats[:, 1:]

        rot_mats_no_root_reshaped = rot_mats_no_root.reshape(-1, 3, 3)
        correct_Rs_reshaped = correct_Rs.reshape(-1, 3, 3)
        rot_mats_no_root = torch.matmul(rot_mats_no_root_reshaped, correct_Rs_reshaped).reshape(-1,3,3)

        save_body_pose = matrix_to_axis_angle(rot_mats_no_root)
        save_body_pose = save_body_pose.reshape(1, -1)
        save_root = pose_org[:, :3].clone().view(1, -1)
        save_shape = viewpoint_camera.smpl_param['shapes'].clone().view(1, -1)
        # for k,v in viewpoint_camera.smpl_param.items():
        #     print(k)
        # print(save_body_pose.shape,save_root.shape,save_shape.shape,111111111111111111111111)
        smplxmodel_output = smplxmodel(betas=save_shape, body_pose=save_body_pose[:, :63], global_orient=save_root,
                                        return_verts=True,return_full_pose=True,
                                        right_hand_pose = save_body_pose[:, 63:108],
                                        left_hand_pose = save_body_pose[:, 108:153],
                                        jaw_pose = zero_pose,
                                        leye_pose = zero_pose,
                                        reye_pose = zero_pose,
                                        expression = torch.zeros((1, 10)).float().repeat(1, 1).cuda())
        overts=overts-torch.tensor(viewpoint_camera.cam_trans).view(1,3).to(torch.float32).cuda()
        collision_loss = smplxmodel.coap.collision_loss(overts, smplxmodel_output)[0] if overts is not None else 0.0
        # print('collision_loss',collision_loss)
        return {"loss_collision": collision_loss}

    def torch_compute_normal_loss(self, h_verts, h_contact, o_verts, o_faces):
        human_faces = torch.from_numpy(h_faces.astype(np.int64)).float().cuda()
        hmesh = Meshes(verts=h_verts.unsqueeze(0), faces=human_faces.unsqueeze(0))
        human_normals = compute_vertex_normals(hmesh)

        obj_faces = torch.from_numpy(o_faces.astype(np.int64)).float().cuda()

        omesh = Meshes(verts=o_verts.unsqueeze(0), faces=obj_faces.unsqueeze(0))
        obj_normals = compute_vertex_normals(omesh)

        dist1, idx1, _ = knn_points(h_verts.unsqueeze(0), o_verts.unsqueeze(0), K=1)  # KNN for A -> B
        dist2, idx2, _ = knn_points(o_verts.unsqueeze(0), h_verts.unsqueeze(0), K=1)  # KNN for B -> A

        normals_B_a = obj_normals[idx1.squeeze(0)].squeeze(1)

        human2obj_cos = torch.sum(human_normals * normals_B_a, dim=1, keepdim=True)

        # normal_loss = torch.sum(human2obj_cos, dim=0)

        nd_dist = dist1.squeeze(0).squeeze(1).cpu().detach().numpy()
        sort_A_list = sorted(range(len(nd_dist)), key=lambda k: nd_dist[k], reverse=False)
        select_human = sort_A_list[:200]
        
        normal_select_loss = torch.sum(human2obj_cos[select_human])
        exp_loss = torch.sigmoid(normal_select_loss / 10)
        
        return {"loss_normal": exp_loss}

    def compute_collision_loss2(self, smplxmodel, viewpoint_camera, correct_Rs, overts, ofaces):
        overts = overts.unsqueeze(0)
        pose_org = viewpoint_camera.smpl_param['poses'].clone()
        #print(pose_org.shape)
        batch_size = pose_org.shape[0]
        rot_mats = axis_angle_to_matrix(pose_org.view(-1, 3)).view([batch_size, -1, 3, 3])
        #print('rot_mats',rot_mats.shape)
        rot_mats_no_root = rot_mats[:, 1:]

        rot_mats_no_root_reshaped = rot_mats_no_root.reshape(-1, 3, 3)
        correct_Rs_reshaped = correct_Rs.reshape(-1, 3, 3)
        rot_mats_no_root = torch.matmul(rot_mats_no_root_reshaped, correct_Rs_reshaped).reshape(-1,3,3)

        save_body_pose = matrix_to_axis_angle(rot_mats_no_root)
        save_body_pose = save_body_pose.reshape(1, -1)
        save_root = pose_org[:, :3].clone().view(1, -1)
        save_shape = viewpoint_camera.smpl_param['shapes'].clone().view(1, -1)
        zero_pose = torch.zeros((1, 3)).float().repeat(1, 1).cuda()
        # for k,v in viewpoint_camera.smpl_param.items():
        #     print(k)
        # print(save_body_pose.shape,save_root.shape,save_shape.shape,111111111111111111111111)
        smplxmodel_output = smplxmodel(betas=save_shape, body_pose=save_body_pose[:, :63], global_orient=save_root,
                                        return_verts=True,return_full_pose=True,
                                        right_hand_pose = save_body_pose[:, 63:108],
                                        left_hand_pose = save_body_pose[:, 108:153],
                                        jaw_pose = zero_pose,
                                        leye_pose = zero_pose,
                                        reye_pose = zero_pose,
                                        expression = torch.zeros((1, 10)).float().repeat(1, 1).cuda())
        # print(smplxmodel_output)
        overts=overts-torch.tensor(viewpoint_camera.cam_trans).view(1,3).to(torch.float32).cuda()
        #hverts=smplxmodel_output.vertices-torch.tensor(viewpoint_camera.cam_trans).view(1,3).to(torch.float32).cuda()
        hverts=smplxmodel_output.vertices
        hfaces=np.asarray(smplxmodel.faces)

        hoi_dict = {
            'smplx_v_centered': overts-torch.mean(overts,dim=1,keepdim=True),  # Add batch dimension
            'object_v_centered': hverts-torch.mean(overts,dim=1,keepdim=True),  # Add batch dimension
        }

        h_in_o_collision_loss = HOCollisionLoss(ofaces).to('cuda')
        h_in_o_collision_loss_dict = h_in_o_collision_loss(hoi_dict)

        hoi_dict = {
            'smplx_v_centered': hverts-torch.mean(hverts,dim=1,keepdim=True),  # Add batch dimension
            'object_v_centered': overts-torch.mean(hverts,dim=1,keepdim=True),  # Add batch dimension
        }
        o_in_h_collision_loss = HOCollisionLoss(hfaces).to('cuda')
        o_in_h_collision_loss_dict = o_in_h_collision_loss(hoi_dict)

        out_loss = 10 * h_in_o_collision_loss_dict + o_in_h_collision_loss_dict

        if out_loss < 50:
            return {"loss_collision": torch.zeros(1, requires_grad=True).float().to('cuda')}
        return {"loss_collision": out_loss}

