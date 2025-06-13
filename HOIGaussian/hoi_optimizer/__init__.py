import numpy as np
import torch.nn as nn
import torch
from .loss import Losses
import open3d as o3d
import cv2
import smplx
# from hoi_optimizer.coma.imports.coap import attach_coap
model_type='smplx'
model_folder="./data/SMPLX_NEUTRAL.npz"
layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False,
             'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False,
             'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}

class HOIOptimizer(nn.Module):

    def __init__(self,
            gaussian,viewpoint_camera,render_pkg
                 ):
        super(HOIOptimizer,self).__init__()
        self.pc= gaussian
        self.losses=Losses()
        self.camera=viewpoint_camera
        self.render_pkg=render_pkg
        self.smplxmodel = smplx.create(model_folder, model_type=model_type,
                                   gender='neutral',
                                   num_betas=10,
                                   num_expression_coeffs=10, use_pca=False, use_face_contour=True, **layer_arg)
        self.smplxmodel=self.smplxmodel.cuda()
        # self.smplxmodel = attach_coap(smplxmodel, pretrained=True, device="cuda")

    def forward(self,opt):
        means3D = self.pc.get_xyz
        # print(means3D.shape)
        means3D_o =means3D[10475:, :]
        ro = self.pc.get_transform_obj  # rotation
        transo = self.pc.get_transl_obj  # translation
        scaleo = self.pc.get_scale_obj  # scale
        means3D_o = self.pc.obj_transform_gs(means3D_o, ro, transo, scaleo)  # transform
        obj_faces = self.camera.face_sim
        # means3D_o = means3D[10475:, :]
        # means3D_h = means3D[0:10475, :]

        dst_posevec = self.camera.smpl_param['poses'][:, 3:]
        pose_out = self.pc.pose_decoder(dst_posevec)
        correct_Rs = pose_out['Rs']
        _, means3D_h, _,_,_ = self.pc.coarse_deform_c2source(self.camera.big_pose_world_vertex[None],
                                                                             self.camera.smpl_param,
                                                                             self.camera.big_pose_smpl_param,
                                                                             self.camera.big_pose_world_vertex[
                                                                                 None], lbs_weights=None,
                                                                             correct_Rs=correct_Rs,
                                                                             return_transl=False)

        # means3D_h=means3D_h.squeeze(0)-torch.as_tensor(self.camera.cam_trans).cuda().view(1,3).to(torch.float32)
        # means3D_o=means3D_o.squeeze(0)
        means3D_h=means3D_h.squeeze(0)

        contact=self.pc.contact
        # print('contact',contact.shape)
        human_contact = contact[0:10475, :].squeeze(1)
        object_contact = contact[10475:, :].squeeze(1)


        hoi_loss_weights={}

        mask_h=self.camera.bkgd_mask_h.to(torch.bool).cuda()
        mask_o=self.camera.bkgd_mask_o.to(torch.bool).cuda()

        loss_dict= {}

        if torch.any(human_contact):
            loss_dict.update(self.losses.compute_contact_loss(hverts=means3D_h, overts=means3D_o,
                                                               h_contact=human_contact, o_contact=object_contact))
            hoi_loss_weights.update({'lw_contact':opt.contact})
        else:
            loss_dict.update({"loss_contact": torch.zeros(1, requires_grad=True).float().to('cuda')})
            hoi_loss_weights.update({'lw_contact':opt.contact})
        
        loss_dict.update(self.losses.compute_ordinal_depth_loss(
            mask_h=mask_h,mask_o=mask_o, h_verts=means3D_h, o_verts=means3D_o, o_faces=self.camera.face_sim, h_faces=np.asarray(self.smplxmodel.faces),cam=self.camera
        ))
        hoi_loss_weights.update({'lw_depth':opt.depth})

        # loss_dict.update(self.losses.compute_ho_distance_loss(
        #     hverts=means3D_h, overts=means3D_o
        # ))
        # hoi_loss_weights.update({'lw_ho_dist':opt.ho_dist})

        loss_dict.update(
              self.losses.compute_collision_loss2(smplxmodel=self.smplxmodel, viewpoint_camera=self.camera,
                                                 correct_Rs=correct_Rs, overts=means3D_o, ofaces=obj_faces))
        hoi_loss_weights.update({'lw_collision': opt.collision})


        return loss_dict, hoi_loss_weights
