import numpy as np
import open3d as o3d
import json
import os
import smplx
import torch
from tqdm import tqdm
import argparse
model_type='smplx'
model_folder="./data/SMPLX_NEUTRAL.npz"
layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
model = smplx.create(model_folder, model_type=model_type,
                         gender='neutral',
                         num_betas=10,
                         num_expression_coeffs=10,use_pca=False,use_face_contour=True,**layer_arg)
zero_pose = torch.zeros((1, 3)).float().repeat(1, 1)

def compute_normals(data_dir):

    smpl_param= json.load(open(os.path.join(data_dir,"smplx_parameters.json")))
    output = model(betas=torch.tensor(smpl_param['shape']), body_pose=torch.tensor(smpl_param['body_pose']),
                    global_orient=torch.tensor(smpl_param['root_pose']),
                    right_hand_pose=torch.tensor(smpl_param['rhand_pose']),
                    left_hand_pose=torch.tensor(smpl_param['lhand_pose']),
                    jaw_pose=torch.tensor(smpl_param['jaw_pose']),
                    leye_pose=zero_pose,
                    reye_pose=zero_pose, expression=torch.tensor(smpl_param['expr']))

    human_points = np.asarray(output.vertices).squeeze(0)
    human = o3d.geometry.TriangleMesh()
    human.vertices = o3d.utility.Vector3dVector(human_points)

    human.compute_vertex_normals()
    trans=smpl_param['cam_trans']

    vertices = np.asarray(human.vertices)
    normals = np.asarray(human.vertex_normals)
    vertices+=trans
    # point_cloud.points = o3d.utility.Vector3dVector(np.asarray(h_mesh.vertices)+np.asarray(trans))

    # Camera extrinsics (R, t) and calculate camera center
    I = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])  # replace with the actual extrinsic matrix

    # The camera position in world coordinates
    camera_center = -np.linalg.inv(I[:3, :3]) @ I[:3, 3]

    # Compute the vector from camera center to each point
    vectors_to_camera = vertices - camera_center
    vectors_to_camera /= np.linalg.norm(vectors_to_camera, axis=1, keepdims=True)  # Normalize

    # Compute cosine similarity between normals and vectors to the camera
    cos_similarities = np.einsum('ij,ij->i', normals, vectors_to_camera)

    # Keep vertices with positive cosine similarity
    positive_cosine_indices = cos_similarities > 0

    np.save(os.path.join(data_dir, "normals_smplx.npy"), positive_cosine_indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory containing smplx_parameters.json')
    args = parser.parse_args()
    compute_normals(args.data_dir)

