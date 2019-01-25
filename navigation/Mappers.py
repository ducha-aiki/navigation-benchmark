import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil,floor
import math
from .Reprojection import getMapSizeInCells, project2dPClIntoWorldMap, ReprojectLocal2Global


def DepthToLocal3D(depth, fx, fy, cx, cy):
    r"""Projects depth map to 3d point cloud
    with origin in the camera focus
    """
    device = depth.device
    h,w = depth.squeeze().size()
    npts = h*w
    x = torch.linspace(0, w-1, w).to(device)
    y = torch.linspace(0, h-1, h).to(device)
    xv, yv = torch.meshgrid([x, y])
    dfl = depth.t().flatten()
    return torch.cat([(dfl *(xv.flatten() - cx) / fx).unsqueeze(-1), #x
                      (dfl *(yv.flatten() - cy) / fy).unsqueeze(-1), #y
                      dfl.unsqueeze(-1)], dim = 1) #z


def pointCloud2ObstaclesNonDifferentiable(pts3D,
                  map_size = 40, 
                  cell_size = 0.2):
    r"""Counts number of 3d points in 2d map cell
    height is sum-pooled.
    """
    device = pts3D.device
    map_size_in_cells = getMapSizeInCells(map_size,cell_size) - 1
    init_map = torch.zeros((map_size_in_cells,map_size_in_cells), device = device)
    if len(pts3D) <= 1:
        return init_map
    num_pts,dim = pts3D.size()
    pts2D = torch.cat([pts3D[:,2:3],pts3D[:,0:1]], dim = 1)
    data_idxs = torch.round(project2dPClIntoWorldMap(pts2D, map_size, cell_size))
    if len(data_idxs) > 10:
        u, counts = np.unique(data_idxs.detach().cpu().numpy(), axis=0, return_counts = True)
        init_map[u[:,0],u[:,1] ] = torch.from_numpy(counts).to(dtype=torch.float32, device=device)
    return init_map

class DirectDepthMapper(nn.Module):
    r"""Estimates obstacle map given the depth image
    ToDo: replace numpy histogram counting with differentiable
    pytorch soft count like in 
    https://papers.nips.cc/paper/7545-unsupervised-learning-of-shape-and-pose-with-differentiable-point-clouds.pdf
    """
    def __init__(self, 
                 #fx = 0,
                 #fy = 0,
                 #cx = 0,
                 #cy = 0,
                 camera_height = 0,
                 near_th = 0.1, far_th = 4.0, h_min = 0.0, h_max = 1.0,
                 map_size = 40, map_cell_size = 0.1,
                 device = torch.device('cpu'),
                 **kwargs):
        super(DirectDepthMapper, self).__init__()
        self.device = device
        #self.fx = fx
        #self.fy = fy
        #self.cx = cx
        #self.cy = cy
        self.near_th = near_th
        self.far_th = far_th
        self.h_min_th = h_min
        self.h_max_th = h_max
        self.camera_height = camera_height 
        self.map_size_meters = map_size
        self.map_cell_size = map_cell_size
        return
    def forward(self, depth, pose = torch.eye(4).float()):
        self.device = depth.device
        #Works for FOV = 45 degrees in minos/sensors.yml. Should be adjusted, if FOV changed
        self.fx = float(depth.size(1))# / 2.0
        self.fy = float(depth.size(0))# / 2.0
        self.cx = int(self.fx)//2 - 1
        self.cy = int(self.fy)//2 - 1
        pose = pose.to(self.device)
        local_3d_pcl = DepthToLocal3D(depth, self.fx, self.fy, self.cx, self.cy)
        idxs = (torch.abs(local_3d_pcl[:,2]) < self.far_th) * (torch.abs(local_3d_pcl[:,2]) >= self.near_th)
        survived_points = local_3d_pcl[idxs]
        if len(survived_points) < 20:
            map_size_in_cells = getMapSizeInCells(self.map_size_meters,self.map_cell_size) - 1
            init_map = torch.zeros((map_size_in_cells,map_size_in_cells), device = self.device)
            return init_map
        global_3d_pcl = ReprojectLocal2Global(survived_points, pose)[:,:3]
        #Because originally y looks down and from agent camera height 
        global_3d_pcl[:,1] = -global_3d_pcl[:,1] + self.camera_height 
        idxs = (global_3d_pcl[:,1] > self.h_min_th) * (global_3d_pcl[:,1] < self.h_max_th)
        global_3d_pcl = global_3d_pcl[idxs]
        obstacle_map = pointCloud2ObstaclesNonDifferentiable(
            global_3d_pcl,
            self.map_size_meters, 
            self.map_cell_size)
        return obstacle_map

class SparseDepthMapper(nn.Module):
    r"""Estimates obstacle map given the 3d points from ORBSLAM
    Does not work well.
    """
    def __init__(self, 
                 fx = 0,
                 fy = 0,
                 cx = 0,
                 cy = 0,
                 camera_height = 0,
                 near_th = 0.1, far_th = 4.0, h_min = 0.0, h_max = 1.0,
                 map_size = 40, map_cell_size = 0.1,
                 device = torch.device('cpu'),
                 **kwargs):
        super(SparseDepthMapper, self).__init__()
        self.device = device
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.near_th = near_th
        self.far_th = far_th
        self.h_min_th = h_min
        self.h_max_th = h_max
        self.camera_height = camera_height 
        self.map_size_meters = map_size
        self.map_cell_size = map_cell_size

        return
    def forward(self, sparse_depth, pose = torch.eye(4).float()):
        global_3d_pcl = sparse_depth
        #Because originally y looks down and from agent camera height 
        global_3d_pcl[:,1] = -global_3d_pcl[:,1]# + self.camera_height 
        idxs = (global_3d_pcl[:,1] > self.h_min_th) * (global_3d_pcl[:,1] < self.h_max_th)
        global_3d_pcl = global_3d_pcl[idxs]
        obstacle_map = pointCloud2ObstaclesNonDifferentiable(
            global_3d_pcl,
            self.map_size_meters, 
            self.map_cell_size)
        return obstacle_map