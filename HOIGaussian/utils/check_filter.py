import open3d as o3d
import numpy as np

# Step 1: Create or load a point cloud (Nx3 for N points with 3D coordinates)
# points = np.array([[0.0, 0.0, 0.0],
#                    [1.0, 1.0, 1.0],
#                    [5.0, 5.0, 5.0],
#                    [0.5, 0.5, 0.5],
#                    [2.0, 2.0, 2.0],
#                    [20.0, 20.0, 20.0],
#                    [21.0, 21.0, 21.0]])

obj_mesh=o3d.io.read_triangle_mesh('/Disk1/robot/boran/3dhoi_Dataset/horse/HICO_train2015_00000982/obj_pcd_h_align.obj')
points=np.asarray(obj_mesh.vertices)
# Step 2: Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Step 3: Perform DBSCAN clustering (adjust eps and min_points for your use case)
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))

# Step 4: Identify the largest cluster (excluding noise labeled as -1)
max_label = labels.max()  # Maximum cluster label (not including noise)
largest_cluster_label = max(range(max_label + 1), key=lambda l: np.sum(labels == l))

# Step 5: Filter the points corresponding to the largest cluster
largest_cluster_indices = (labels == largest_cluster_label)
largest_cluster_points = points[largest_cluster_indices]

# Step 6: Create a new point cloud with the largest cluster
largest_cluster_pcd = o3d.geometry.PointCloud()
largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)

# Step 7: Visualize the original point cloud and the largest cluster
print(f"Original points:\n{points}")
print(f"Cluster labels:\n{labels}")
print(f"Largest cluster points:\n{largest_cluster_points}")

# Visualize the filtered point cloud
o3d.io.write_point_cloud('./filter.ply', largest_cluster_pcd)
