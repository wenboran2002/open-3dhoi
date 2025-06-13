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

import os
import random
import json
import math
from utils.system_utils import searchForMaxIteration
from hoi_scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from utils.system_utils import mkdir_p
from hoi_scene.gaussian_model import BasicPointCloud
from hoi_scene.hoi import load_hoi
from utils.sh_utils import SH2RGB
import numpy as np
import copy
import scipy.linalg as linalg

def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian * math.pi/180))
    return rot_matrix

def get_R_from2vec(origin_vector, location_vector):
    # get Rotate matrix from origin_vector and location_vector
    c = np.dot(origin_vector, location_vector)
    n_vector = np.cross(origin_vector, location_vector)
    s = np.linalg.norm(n_vector)
    print(c, s)
    
    n_vector_invert = np.array((
        [0,-n_vector[2],n_vector[1]],
        [n_vector[2],0,-n_vector[0]],
        [-n_vector[1],n_vector[0],0]
        ))
    I = np.eye(3)

    R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert)/(1+c)
    
    return R_w2c

class HOIDataset:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """ wbr
        dataset initialization
        """
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.loaded_iter = None
        self.gaussians = gaussians
        nerf_radius=1

        self.camera_extent=nerf_radius
        self.train_cameras = {}
        # load hoi info
        self.hoi_infos=[]

        img_name = args.file_name

        # obj_name='horse'

        # hoi_info = load_hoi(os.path.join(self.data_path, hoi_dir), args.white_background, dataset_type=args.dataset_type)
        # hoi_info = load_hoi('/hdd/boran/3dhoi_Dataset/elephant/HICO_train2015_00006488/', args.white_background, dataset_type=args.dataset_type)
        hoi_info = load_hoi(f'{args.data_path}/{img_name}/', args.white_background, dataset_type=args.dataset_type)
        hoi_info['uid'] = 0
        # hoi_info['image_name'] = hoi_dir
        # hoi_info['image_name'] = 'HICO_train2015_00006488'

        hoi_info['image_name'] = img_name
        hoi_info['data_device'] = args.data_device

        if not os.path.exists(f"./output/test_output/{img_name}"):
            os.makedirs(f"./output/test_output/{img_name}")
        if not os.path.exists(f"./output/test_output/{img_name}/pcd"):
            os.makedirs(f"./output/test_output/{img_name}/pcd")
        if not os.path.exists(f"./output/test_output/{img_name}/render"):
            os.makedirs(f"./output/test_output/{img_name}/render")
            if not os.path.exists(f"./output/test_output/{img_name}/contact"):
                os.makedirs(f"./output/test_output/{img_name}/contact")

        self.hoi_infos.append(hoi_info)

        num_pts=10475
        num_pts_obj=self.hoi_infos[0]['num_obj']
        shs = np.random.random((num_pts, 3)) / 255.0
        shs_obj = np.random.random((num_pts_obj, 3)) / 255.0
        print(self.hoi_infos[0]['obj_color'].shape)

        obj_for_save=self.hoi_infos[0]['save_obj']
        self.pcd = BasicPointCloud(points=self.hoi_infos[0]['big_pose_world_vertex'], colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)),contacts=self.hoi_infos[0]['normals_h'])
        ## load object
        #self.obj_pcd = BasicPointCloud(points=self.hoi_infos[0]['obj_vertex'], colors=self.hoi_infos[0]['obj_color'], normals=np.zeros((num_pts_obj, 3)),contacts=self.hoi_infos[0]['normals_h'])
        self.obj_pcd = BasicPointCloud(points=self.hoi_infos[0]['obj_vertex'], colors=SH2RGB(shs_obj), normals=np.zeros((num_pts_obj, 3)),contacts=self.hoi_infos[0]['normals_h'])
        self.gaussians.create_from_pcd(self.pcd,self.obj_pcd,nerf_radius,obj_for_save)
        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale]=cameraList_from_camInfos(self.hoi_infos, resolution_scale, args)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(iteration), "ckpt.pth")
            mkdir_p(os.path.dirname(model_path))
            torch.save({
                'iter': iteration,
                'pose_decoder': self.gaussians.pose_decoder.state_dict(),
                'lweight_offset_decoder': self.gaussians.lweight_offset_decoder.state_dict(),
            }, model_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]