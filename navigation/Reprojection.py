import numpy as np
import torch
import torch.nn.functional as F
import math
from math import ceil,floor


def Pzx(P):
    return P[(0,2),3]
def getMapSizeInCells(map_size_in_meters, cell_size_in_meters):
    return int(ceil(map_size_in_meters / cell_size_in_meters))+1

def getPosDiff(Pinit,Pfin):
    return Pzx(Pfin) - Pzx(Pinit)
def getPosDiffLength(Pinit,Pfin):
    return torch.norm(getPosDiff(Pinit,Pfin)) 

def getPosDiffs(Ps):
    return Ps[1:,(0,2),3] - Ps[:(Ps.size(0)-1),(0,2),3]


def angleToPi_2_MinusPi_2(angle):
    if angle < -np.pi:
        angle = 2.0*np.pi + angle
    if angle > np.pi:
        angle = -2.0*np.pi + angle
    return angle


def getDirection(Pinit,Pfin, ang_th = 0.2, pos_th = 0.1):
    pos_diff = getPosDiff(Pinit,Pfin)
    if torch.norm(pos_diff,2).item() < pos_th:
        return 0
    else:
        needed_angle =  torch.atan2(pos_diff[1], pos_diff[0])
        current_angle = torch.atan2(Pinit[2,0], Pinit[0,0])
    to_rotate = angleToPi_2_MinusPi_2(-np.pi/2.0 + needed_angle - current_angle)      
    if torch.abs(to_rotate).item() < ang_th:
        return 0
    return to_rotate

def ReprojectLocal2Global(xyz_local, P):
    device = xyz_local.device
    num, dim = xyz_local.size()
    if dim == 3:
        xyz = torch.cat([xyz_local, 
                         torch.ones((num,1), dtype = torch.float32, device=device)], dim = 1)
    elif dim == 4:
        xyz = xyz_local
    else:
        raise ValueError('3d point cloud dim is neighter 3, or 4 (homogenious)')
    #print(xyz.shape, P.shape)
    xyz_global = torch.mm(P.squeeze(),xyz.t())
    return xyz_global.t()


def project2dPClIntoWorldMap(zx, map_size, cell_size):
    device = zx.device
    shift = int(floor(getMapSizeInCells(map_size, cell_size)/2.0))
    topdown2index = torch.tensor([[1.0/cell_size, 0, shift],
                                  [0, 1.0/cell_size, shift],
                                  [0, 0, 1]], device = device)
    world_coords_h = torch.cat([zx.view(-1,2),
                                torch.ones((len(zx),1), device = device)], dim = 1)
    world_coords = torch.mm(topdown2index, world_coords_h.t())
    return world_coords.t()[:,:2]

def getPose2D(poses6d):
    poses6d = poses6d.view(-1,4,4);
    poses2d = poses6d[:,(0,2)]
    poses2d = poses2d[:,:,(0,2,3)]
    return poses2d

def get_rotation_matrix(angle_in_radians):
    angle_in_radians = angle_in_radians.view(-1, 1, 1);
    sin_a = torch.sin(angle_in_radians)
    cos_a = torch.cos(angle_in_radians)
    A1_x = torch.cat([cos_a, sin_a], dim = 2)
    A2_x = torch.cat([-sin_a, cos_a], dim = 2)
    transform = torch.cat([A1_x,A2_x], dim = 1)
    return transform

def normalizeZXOri(P):
    p2d = getPose2D(P)
    norms = torch.norm(p2d[:,0,:2],dim=1).view(-1,1,1)
    out = torch.cat([
                    torch.cat([P[:,:3,:3] / norms.expand(P.size(0),3,3), P[:,3:,:3]], dim = 1),
                    P[:,:,3:]], dim = 2)
    return out

def addRotWPs(P):
    plannedTPsNorm = normalizeZXOri(P)
    pos_diffs = getPosDiffs(plannedTPsNorm)
    
    angles = torch.atan2(pos_diffs[:,1], pos_diffs[:,0])
    rotmats = get_rotation_matrix(angles)
    plannedTPsNorm[:P.size(0)-1,0,0] = rotmats[:,0,0]
    plannedTPsNorm[:P.size(0)-1,0,2] = rotmats[:,0,1]
    plannedTPsNorm[:P.size(0)-1,2,0] = rotmats[:,1,0]
    plannedTPsNorm[:P.size(0)-1,2,2] = rotmats[:,1,1]
    
    plannedPoints2 = plannedTPsNorm.clone()
    
    plannedPoints2[1:,0,0] = plannedTPsNorm[:P.size(0)-1,0,0]
    plannedPoints2[1:,0,2] = plannedTPsNorm[:P.size(0)-1,0,2]
    plannedPoints2[1:,2,0] = plannedTPsNorm[:P.size(0)-1,2,0]
    plannedPoints2[1:,2,2] = plannedTPsNorm[:P.size(0)-1,2,2]
    
    out = torch.stack((plannedPoints2.unsqueeze(0),plannedTPsNorm.unsqueeze(0)), dim=0).squeeze()
    out = out.permute(1,0,2,3).contiguous().view(-1,4,4)
    return out

def plannedPath2TPs(path, cell_size, map_size, agent_h, addRot = False):
    path = torch.cat(path).view(-1,2)
    #print(path.size())
    num_pts = len(path);
    plannedTPs = torch.eye(4).unsqueeze(0).repeat((num_pts, 1,1))
    plannedTPs[:,0,3] = path[:,1]#switch back x and z
    plannedTPs[:,1,3] = agent_h
    plannedTPs[:,2,3] = path[:,0]#switch back x and z
    shift = int(floor(getMapSizeInCells(map_size, cell_size)/2.0))
    plannedTPs[:,0,3] = plannedTPs[:,0,3] - shift
    plannedTPs[:,2,3] = plannedTPs[:,2,3] - shift
    P = torch.tensor([
                   [1.0/cell_size, 0, 0, 0],
                   [0, 1.0/cell_size, 0, 0],
                   [0, 0, 1.0/cell_size, 0],
                   [0, 0, 0,  1]])
    plannedTPs = torch.bmm(P.inverse().unsqueeze(0).expand(num_pts,4,4), plannedTPs)
    if addRot:
        return addRotWPs(plannedTPs)
    return plannedTPs

def minosOffsetToGoal2MapGoalPosition(offset,
                                      P_curr,
                                      cell_size,
                                      map_size):
    device = offset.device
    goal_tp = minosOffsetToGoal2TP(offset,
                                   P_curr)
    goal_tp1 = torch.eye(4).to(device)
    goal_tp1[:,3:] = goal_tp
    projected_p = projectTPsIntoWorldMap(goal_tp1.view(1,4,4), 
                                         cell_size,
                                         map_size)
    return projected_p
def minosOffsetToGoal2TP(offset,
                         P_curr):
    device = offset.device
    if P_curr.size(1) == 3:
        P_curr = homogenizeP(P_curr)
    GoalTP = torch.mm(P_curr.to(device),
                      torch.cat([offset * torch.tensor([1.,1.,-1.],dtype=torch.float32,device=device),
                                 torch.tensor([1.],device=device)]).reshape(4,1))
    return GoalTP
def homogenizeP(tps):
    device = tps.device
    tps = tps.view(-1,3,4)
    return  torch.cat([tps.float(),torch.tensor([0,0,0,1.0]).view(1,1,4).expand(tps.size(0),1,4).to(device)],dim = 1)
def projectTPsIntoWorldMap(tps, cell_size, map_size, do_floor = True):
    if len(tps)  == 0:
        return []
    if type(tps) is list:
        return []
    device = tps.device
    topdownP = torch.tensor([[1.0,0,0,0],
                         [0,0,1.0,0]]).to(device)
    world_coords = torch.bmm(topdownP.view(1,2,4).expand(tps.size(0),2,4) , tps[:,:,3:].view(-1,4,1))
    shift = int(floor(getMapSizeInCells(map_size, cell_size)/2.0))
    topdown2index = torch.tensor([[1.0/cell_size, 0, shift],
                              [0, 1.0/cell_size, shift],
                              [0, 0, 1]]).to(device)
    world_coords_h = torch.cat([world_coords, torch.ones((len(world_coords),1,1)).to(device)], dim = 1)
    #print(topdown2index.size(),world_coords_h.size())
    world_coords = torch.bmm(topdown2index.unsqueeze(0).expand(world_coords_h.size(0),3,3), world_coords_h)[:,:2,0]
    if do_floor:
        return torch.floor(world_coords.flip(1)) + 1 #for having revesrve (z,x) ordering
    return world_coords.flip(1)
def projectTPsIntoWorldMap_numpy(tps, slam_to_world, cell_size, map_size):
    if len(tps)  == 0:
        return []
    if type(tps) is list:
        return []
    #tps is expected in [n,4,4] format
    topdownP = np.array([[slam_to_world,0,0,0],
                         [0,0,slam_to_world,0]])
    try:
        world_coords = np.matmul(topdownP.reshape(1,2,4) , tps[:,:,3:].reshape(-1,4,1))
    except:
        return []
    shift = int(floor(getMapSizeInCells(map_size, cell_size)/2.0))
    topdown2index = np.array([[1.0/cell_size, 0, shift],
                              [0, 1.0/cell_size, shift],
                              [0, 0, 1]])
    world_coords_h = np.concatenate([world_coords, np.ones((len(world_coords),1,1))], axis = 1)
    world_coords = np.matmul(topdown2index, world_coords_h)[:,:2,0]
    return world_coords[:,::-1].astype(np.int32) + 1 #for having revesrve (z,x) ordering