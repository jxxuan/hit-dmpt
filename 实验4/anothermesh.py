import open3d as o3d
import numpy as np
# # 加载点云
# point_cloud = o3d.io.read_point_cloud("bunny.ply")
#
# # 估计法线
# point_cloud.estimate_normals()
#
# # 将点云转换为网格
# mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)
#
# # 计算顶点法线
# mesh.compute_vertex_normals()
#
# # 可视化网格
# o3d.visualization.draw_geometries([mesh])

# 球旋转算法
pc = o3d.io.read_point_cloud("bunny.ply")
pc.estimate_normals()
# estimate radius for rolling ball
distances = pc.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector([radius, radius * 2]))
print(mesh.get_surface_area())   # 表面积
o3d.visualization.draw_geometries([mesh], window_name='Open3D downSample', width=800, height=600, left=50, top=50,
                                  point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True,)