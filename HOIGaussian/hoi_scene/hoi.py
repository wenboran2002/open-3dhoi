import os
import json
import smplx
import torch
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import imageio
import cv2
from smpl.smpl_numpy import SMPL
from PIL import Image
import open3d as o3d
from copy import deepcopy
model_type='smplx'
model_folder="./data/SMPLX_NEUTRAL.npz"
layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False,
             'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False,
             'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
smplx_model = smplx.create(model_folder, model_type=model_type,
                     gender='neutral',
                     num_betas=10,
                     num_expression_coeffs=10, use_pca=False, use_face_contour=True, **layer_arg)
zero_pose = torch.zeros((1, 3)).float().repeat(1, 1)
def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d
def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy
def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask
def load_hoi(content_dir, white_background,dataset_type='BEHAVE',image_scaling=0.5):
    record={}

    ## load_image:
    image_path=os.path.join(content_dir, 'image.jpg')
    mask_path=os.path.join(content_dir, 'person_mask.png')
    obj_mask_path = os.path.join(content_dir, 'object_mask_refine.png')
    if not os.path.exists(obj_mask_path):
        obj_mask_path = os.path.join(content_dir, 'object_mask.png')
    h_msk = imageio.imread(mask_path)
    obj_msk = imageio.imread(obj_mask_path)
    h_msk = (h_msk != 0)
    obj_msk = (obj_msk != 0)
    obj_msk=obj_msk.astype(np.uint8)
    h_msk=h_msk.astype(np.uint8)
    msk=np.logical_or(h_msk,obj_msk).astype(np.uint8)
    image= np.array(imageio.imread(image_path).astype(np.float32)/255.)
    image_o=np.array(imageio.imread(image_path).astype(np.float32)/255.)
    image_h=np.array(imageio.imread(image_path).astype(np.float32)/255.)
    image[msk == 0] = 1 if white_background else 0

    image_vis=np.copy(image)
    image_o[obj_msk == 0] = 1 if white_background else 0
    image_h[h_msk == 0] = 1 if white_background else 0

    ratio = image_scaling/2.0


    ## get cameras
    R=None
    T=None
    K=None
    ## behave:
    if dataset_type=='BEHAVE':
        date=content_dir.split('/')[-1][:6]
        # load intrinsic
        calib_path=os.path.join(content_dir,"calibration.json")
        calib=json.load(open(calib_path))
        K = np.asarray(calib['K'])

        ## load w2c
        cam_config = json.load(open(os.path.join(content_dir,"extrinsic.json")))
        r = cam_config['rotation']
        t = cam_config['translation']
        R = np.asarray(r).reshape(3, 3)
        T = np.asarray(t).reshape(3, 1)
    IMAGE_SIZE=256
    if ratio != 1.:
        w, h = image.shape[1], image.shape[0]
        ratio = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
        # rescale
        W = int(ratio * w)
        H = int(ratio * h)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        image_o = cv2.resize(image_o, (W, H), interpolation=cv2.INTER_AREA)
        image_h = cv2.resize(image_h, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        obj_msk = cv2.resize(obj_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        h_msk = cv2.resize(h_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        K[:2] = K[:2] * ratio
    if len(image.shape) == 2:  # Check if the image is grayscale
        # Stack to create an RGB image by duplicating the grayscale channel
        image_rgb = np.stack((image,) * 3, axis=-1)
        image_rgb = np.array(image_rgb * 255.0, dtype=np.uint8)  # Scale and convert to uint8
        image = Image.fromarray(image_rgb, "RGB")
    else:
        # For RGB images with correct shape
        image = Image.fromarray(np.array(image * 255.0, dtype=np.uint8), "RGB")

    if len(image_o.shape) == 2:  # Check if the image is grayscale
        # Stack to create an RGB image by duplicating the grayscale channel
        image_rgb_o = np.stack((image_o,) * 3, axis=-1)
        image_rgb_o = np.array(image_rgb_o * 255.0, dtype=np.uint8)  # Scale and convert to uint8
        image_o = Image.fromarray(image_rgb_o, "RGB")
    else:
        # For RGB images with correct shape
        image_o = Image.fromarray(np.array(image_o * 255.0, dtype=np.uint8), "RGB")
    if len(image_h.shape) == 2:  # Check if the image is grayscale
        # Stack to create an RGB image by duplicating the grayscale channel
        image_rgb_h = np.stack((image_h,) * 3, axis=-1)
        image_rgb_h = np.array(image_rgb_h * 255.0, dtype=np.uint8)  # Scale and convert to uint8
        image_h = Image.fromarray(image_rgb_h, "RGB")
    else:
        # For RGB images with correct shape
        image_h = Image.fromarray(np.array(image_h * 255.0, dtype=np.uint8), "RGB")
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3:4] = T
    focalX = K[0, 0]
    focalY = K[1, 1]
    FovX = focal2fov(focalX, image.size[0])
    FovY = focal2fov(focalY, image.size[1])

    ## load canonical smpl
    num_pts = 10475  # 100_000
    print(f"Generating random point cloud ({num_pts})...")

    # smpl_model = SMPL(sex='neutral', model_dir="/Disk1/robot/boran/SMPL_NEUTRAL.pkl")
    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['shapes'] = np.zeros((1, 10)).astype(np.float32)
    big_pose_smpl_param['poses'] = np.zeros((1, 165)).astype(np.float32)

    output = smplx_model(betas=torch.tensor(big_pose_smpl_param['shapes']),
                         body_pose=torch.tensor(big_pose_smpl_param['poses'][:, :63]),
                         global_orient=zero_pose,
                         right_hand_pose=torch.tensor(big_pose_smpl_param['poses'][:, 63:108]),
                         left_hand_pose=torch.tensor(big_pose_smpl_param['poses'][:, 108:153]),
                         jaw_pose=zero_pose,
                         leye_pose=zero_pose,
                         reye_pose=zero_pose, expression=torch.zeros((1, 10)).float().repeat(1, 1))
    big_pose_xyz = output.vertices[0].detach().cpu().numpy()
    big_pose_xyz = (
            np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(
        np.float32)
    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= 0.05
    big_pose_max_xyz += 0.05
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

    ## create smpl param
    smplx_path = os.path.join(content_dir, 'smplx_parameters.json')
    smplx_param = json.load(open(smplx_path))
    if type(smplx_param) == list:
        smplx_param=smplx_param[0]

    smpl_param = {}
    smpl_param['shapes'] = np.asarray(smplx_param['shape']).reshape(1, 10)
    smpl_param['poses'] = []
    smpl_param['poses'].append(np.asarray(smplx_param['root_pose']).reshape(1, 3))
    smpl_param['poses'].append(np.asarray(smplx_param['body_pose']).reshape(1, -1))

    smpl_param['poses'].append(np.asarray(smplx_param['jaw_pose']).reshape(1, -1))

    smpl_param['poses'].append(np.asarray(smplx_param['lhand_pose']).reshape(1, -1))
    smpl_param['poses'].append(np.asarray(smplx_param['rhand_pose']).reshape(1, -1))
    smpl_param['poses'].append(np.zeros((1, 6)).reshape(1, -1))

    smpl_param['poses'] = np.concatenate(smpl_param['poses'], axis=1)
    smpl_param['Th'] = np.asarray(smplx_param['cam_trans'])
    print('pose', smpl_param['poses'].shape)
    # print(smpl_param['poses'].shape)
    Rh=np.zeros((1,3))
    smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)

    focal_param= np.asarray(smplx_param['focal'])
    princpt= np.asarray(smplx_param['princpt'])
    cam_param={'focal': focal_param, 'princpt': princpt}



    ## create xyz
    output = smplx_model(betas=torch.tensor(smplx_param['shape']), body_pose=torch.tensor(smplx_param['body_pose']),
                         global_orient=torch.tensor(smplx_param['root_pose']),
                         right_hand_pose=torch.tensor(smplx_param['rhand_pose']),
                         left_hand_pose=torch.tensor(smplx_param['lhand_pose']),
                         jaw_pose=torch.tensor(smplx_param['jaw_pose']),
                         leye_pose=zero_pose,
                         reye_pose=zero_pose, expression=torch.tensor(smplx_param['expr']))

    cam_trans = np.asarray(smplx_param['cam_trans']).reshape(3, 1)
    xyz = output.vertices[0].detach().cpu().numpy()
    xyz = xyz + cam_trans.reshape(1, 3)
    xyz_h_center= np.mean(xyz, axis=0)

    human_n=np.load(os.path.join(content_dir, 'normals_smplx.npy'))
    print('contact', np.sum(human_n))

    ## obj niormal

    ## create object xyz
    object_path = os.path.join(content_dir, 'obj_pcd_h_align.obj')
    obj=o3d.io.read_triangle_mesh(object_path)

    obj_sim=obj.simplify_quadric_decimation(target_number_of_triangles=8000)
    # obj=obj.simplify_quadric_decimation(target_number=4096)
    xyz_obj = np.asarray(obj.vertices)
    xyz_obj_save=deepcopy(xyz_obj)
    xyz_obj_sim=np.asarray(obj_sim.vertices)

    obj_faces= np.asarray(obj.triangles)
    obj_faces_sim= np.asarray(obj_sim.triangles)

    xyz_obj=xyz_obj_sim + cam_trans.reshape(1, 3)

    xyz_obj_save=xyz_obj_save+cam_trans.reshape(1, 3)

    obj_color=np.asarray(obj_sim.vertex_colors)

    human_normals = np.zeros((xyz.shape[0], 1))
    human_normals[human_n] = 1
    human_normals=np.concatenate([human_normals, np.zeros((xyz_obj.shape[0],1))], axis=0)

    xyz_obj[:, 2] = xyz_obj[:, 2]
    num_obj=xyz_obj.shape[0]

    ## merge
    xyz_bound = np.concatenate([xyz, xyz_obj], axis=0)

    # obtain the original bounds for point sampling
    min_xyz = np.min(xyz_bound, axis=0)
    max_xyz = np.max(xyz_bound, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    world_bound = np.stack([min_xyz, max_xyz], axis=0)

    # get bounding mask and background mask
    bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
    bound_mask = Image.fromarray(np.array(bound_mask * 255.0, dtype=np.byte))
    bkgd_mask = Image.fromarray(np.array(msk * 255.0, dtype=np.byte))
    bkgd_mask_o= Image.fromarray(np.array(obj_msk * 255.0, dtype=np.byte))
    bkgd_mask_h= Image.fromarray(np.array(h_msk * 255.0, dtype=np.byte))

    record['smpl_param']=smpl_param
    record['world_bound']=world_bound
    record['bound_mask']=bound_mask
    record['bkgd_mask']=bkgd_mask
    record['bkgd_mask_o']=bkgd_mask_o
    record['bkgd_mask_h']=bkgd_mask_h
    record['world_vertex']=xyz
    record['image']=image
    record['image_o']=image_o
    record['image_h']=image_h
    record['R']=R
    record['T']=T
    record['K']=K
    record['FovY']=FovY
    record['FovX']=FovX
    record['big_pose_smpl_param'] = big_pose_smpl_param
    record['big_pose_world_vertex'] = big_pose_xyz
    record['big_pose_world_bound'] = big_pose_world_bound
    record['obj_vertex']=xyz_obj
    record['save_obj']=xyz_obj_save
    record['obj_faces']=obj_faces
    record['obj_color']=obj_color
    record['num_obj']=num_obj
    record['normals_h']=human_normals
    record['cam_trans']=cam_trans
    record['cam_param']=cam_param
    record['img_vis']=image_vis
    record['sim_faces']=obj_faces_sim

    return record



