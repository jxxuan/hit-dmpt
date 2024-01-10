import numpy as np
import open3d as o3d
import pyvista as pv

# NumPy array with shape (n_points, 3)
# Load saved point cloud
pcd_load = o3d.io.read_point_cloud("bunny.ply")

# convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_load = np.asarray(pcd_load.points)

point_cloud = pv.PolyData(xyz_load)

mesh = point_cloud.reconstruct_surface()

mesh.save('mesh.stl')
mesh.plot(color='orange')
