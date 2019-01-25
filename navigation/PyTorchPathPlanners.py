import numpy as np
import os
import random
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle as pickle
from .Utils import generate_2dgrid
from time import time


class SoftArgMin(nn.Module):
    def __init__(self, beta = 5):
        super(SoftArgMin, self).__init__()
        self.beta = beta
        return
    def forward(self, x, coords2d = None):
        bx_sm = F.softmax(self.beta * (-x).view(1,-1), dim=1)
        if coords2d is None:
            coords2d = generate_2dgrid(x.size(2), x.size(3),False)
        coords2d_flat = coords2d.view(2,-1) 
        return  (bx_sm.expand_as(coords2d_flat) * coords2d_flat).sum(dim=1) / bx_sm.sum(dim=1)

class HardArgMin(nn.Module):
    def __init__(self):
        super(HardArgMin, self).__init__()
        return
    def forward(self, x, coords2d = None):
        val,idx = x.view(-1).min(dim=0)
        if coords2d is None:
            coords2d = generate_2dgrid(x.size(2), x.size(3),False)
        coords2d_flat = coords2d.view(2,-1) 
        return  coords2d_flat[:,idx].view(2)

def safeROI2d(array2d, ymin,ymax,xmin,xmax):
        (H,W) = array2d.shape
        return max(0,ymin),min(ymax,H),max(0,xmin),min(xmax,W)

def f2ind(ten,i):
    #Float to index
    return torch.round(ten[i]).long()

def initNeights2channels(ks = 3):
    r"""Convolutional kernel,
    which maps nighborhood into channels
    """   
    weights = np.zeros((ks*ks,1,ks,ks), dtype= np.float32)
    for y in range(ks):
        for x in range(ks):
            weights[x*ks + y, 0, y, x] = 1.0
    return weights

class DifferentiableStarPlanner(nn.Module):
    def __init__(self, max_steps = 500, visualize = False, preprocess = False,
                 beta = 100,
                 connectivity = "eight",
                 device = torch.device('cpu'), **kwargs):
        super(DifferentiableStarPlanner, self).__init__()
        self.eps = 1e-12
        self.max_steps = max_steps;
        self.visualize = visualize
        self.inf = 1e7
        self.ob_cost = 10000.0
        self.device = device
        self.beta = beta
        self.preprocess = preprocess
        #self.argmin = SoftArgMin(beta)
        self.argmin = HardArgMin()
        self.neights2channels = nn.Conv2d(1, 9, kernel_size=(3,3), bias = False)
        self.neights2channels.weight.data = torch.from_numpy(initNeights2channels(3))
        self.neights2channels.to(device)
        self.preprocessNet = nn.Conv2d(1, 1, kernel_size=(3,3), padding = 1, bias = False)
        self.preprocessNet.weight.data = torch.from_numpy(np.array([[[[0.00001, 0.0001, 0.00001],
                                                                       [0.0001, 1, 0.0001],
                                                                       [0.00001, 0.0001, 0.00001]]]], dtype=np.float32))
        self.preprocessNet.to(device)
        if connectivity == "eight":
            self.gx_to_right =  nn.Conv2d(1, 1, kernel_size=(1,3), bias = False)
            self.gx_to_right.weight.data = torch.from_numpy(np.array([[[[0, 1, -1]]]], dtype=np.float32))
            self.gx_to_right.to(device)
            
            self.gx_to_left =  nn.Conv2d(1, 1, kernel_size=(1,3), bias = False)
            self.gx_to_left.weight.data = torch.from_numpy(np.array([[[[-1, 1, 0]]]], dtype=np.float32))
            self.gx_to_left.to(device)
            
            self.gy_to_up =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
            self.gy_to_up.weight.data = torch.from_numpy(np.array([[[[0], [1], [-1]]]], dtype=np.float32))
            self.gy_to_up.to(device)
            
            self.gy_to_down =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
            self.gy_to_down.weight.data = torch.from_numpy(np.array([[[[-1], [1], [0]]]], dtype=np.float32))
            self.gy_to_down.to(device)
        else:
            raise ValueError('Only "eight" connectivity now supported')
        return
    def preprocess_obstacle_map(self, obstacle_map):
        if self.preprocess:
            return self.preprocessNet(obstacle_map)#torch.clamp(self.preprocessNet(obstacle_map), min = 0.0, max = 1.0)
        return obstacle_map#torch.clamp(obstacle_map, min = 0.0, max = 1.0)
    def coords2grid(self, node_coords, h,w):
        grid = node_coords.squeeze() - torch.FloatTensor((h/2.0,w/2.0)).to(self.device)
        grid = grid / torch.FloatTensor((h/2.0,w/2.0)).to(self.device)
        return  grid.view(1,1,1,2).flip(3)
    def initCloseListMap(self):
        return torch.zeros_like(self.start_map).float()#(self.obstacles >= 0.7).float() 
    def initOpenListMap(self):
        return self.start_map.clone()
    def initGMap(self):
        return  torch.clamp(self.inf* (torch.ones_like(self.start_map) - \
                                                           self.start_map.clone()), min = 0,max = self.inf)
    def safeROI2d(self, ymin,ymax,xmin,xmax):
        return int(max(0,torch.round(ymin).item())),int(min(torch.round(ymax).item(),self.height)),\
               int(max(0,torch.round(xmin).item())),int(min(torch.round(xmax).item(),self.width))
    def forward(self, obstacles, coords, start_map, goal_map, 
                non_obstacle_cost_map = None,
                additional_steps = 50,
                return_path = True):######################################################
        self.trav_init_time = 0
        self.trav_mask_time = 0
        self.trav_soft_time = 0
        self.conv_time = 0
        self.close_time = 0
        
        t=time()
        self.obstacles = self.preprocess_obstacle_map(obstacles.to(self.device))
        self.start_map = start_map.to(self.device)
        self.been_there = torch.zeros_like(self.start_map).to(torch.device('cpu'))
        self.coords = coords.to(self.device)
        self.goal_map = goal_map.to(self.device)
        self.been_there = torch.zeros_like(self.goal_map).to(self.device)
        self.height = obstacles.size(2)
        self.width = obstacles.size(3)
        m,goal_idx = torch.max(self.goal_map.view(-1),0)
        #######################
        c_map = self.calculateLocalPathCosts(non_obstacle_cost_map) #this might be non persistent in map update
        ######################
        self.g_map = self.initGMap()
        padded_gmap = F.pad(self.g_map,(1,1, 1,1), 'replicate')
        ######
        self.close_list_map = self.initCloseListMap()
        self.open_list_map = self.initOpenListMap()
        #######
        not_done = False
        step = 0
        stopped_by_max_iter = False
        #print (time() -t, 'init time')
        t=time()
        if self.visualize:
            self.fig,self.ax = plt.subplots(1,1)
            self.image = self.ax.imshow(self.g_map.squeeze().cpu().detach().numpy().astype(np.float32),
                                    animated=True)
            self.fig.canvas.draw()
        not_done = (self.close_list_map.view(-1)[goal_idx].item() <  1.0) or (self.g_map.view(-1)[goal_idx].item() >=  0.9*self.ob_cost)
        R = 1
        self.start_coords= (self.coords * self.start_map.expand_as(self.coords)).sum(dim=2).sum(dim=2).squeeze()
        node_coords = self.start_coords
        self.goal_coords= (self.coords * self.goal_map.expand_as(self.coords)).sum(dim=2).sum(dim=2).squeeze()
        self.max_steps = 4* int(torch.sqrt(((self.start_coords -self.goal_coords )**2).sum() + 1e-6).item())
        while not_done:
            ymin,ymax,xmin,xmax = self.safeROI2d(node_coords[0]-R,
                                     node_coords[0]+R+1,
                                     node_coords[1]-R,
                                     node_coords[1]+R+1) 
            if (ymin-1 > 0) and (xmin-1 > 0) and (ymax+1 < self.height) and (xmax+1 < self.width):
                #t=time()
                n2c = self.neights2channels(self.g_map[:,:,ymin-1:ymax+1, xmin-1:xmax+1])
                self.g_map[:,:,ymin:ymax, xmin:xmax] = torch.min(self.g_map[:,:,ymin:ymax, xmin:xmax].clone(),
                             ( n2c +\
                              c_map[:,:,ymin:ymax, xmin:xmax]).min(dim = 1,
                                                    keepdim = True)[0])
                #self.conv_time +=time() - t
                t=time()
                self.close_list_map[:,:,ymin:ymax, xmin:xmax] = torch.max(self.close_list_map[:,:,ymin:ymax, xmin:xmax], 
                                                                          self.open_list_map[:,:,ymin:ymax, xmin:xmax])
                self.open_list_map[:,:,ymin:ymax, xmin:xmax] = F.relu(F.max_pool2d(self.open_list_map[:,:,ymin-1:ymax+1, xmin-1:xmax+1], 3, 
                                                         stride=1, padding=0) - self.close_list_map[:,:,ymin:ymax, xmin:xmax] -\
                                            self.obstacles[:,:,ymin:ymax, xmin:xmax])
                #self.close_time +=time() - t
            else:
                t=time()
                self.g_map = torch.min(self.g_map,
                             (self.neights2channels(F.pad(self.g_map,(1,1, 1,1), 'replicate'))  + c_map).min(dim = 1,
                                                    keepdim = True)[0])
                #self.conv_time +=time() - t
                #t=time()
                self.close_list_map = torch.max(self.close_list_map, self.open_list_map)
                self.open_list_map = F.relu(F.max_pool2d(self.open_list_map, 3, 
                                                         stride=1, padding=1) - self.close_list_map - self.obstacles)
                #self.close_time +=time() - t
            step+=1
            if step % 100 == 0:
                #print (step)
                if self.visualize:
                    plt.imshow(self.g_map.cpu().detach().numpy().squeeze().astype(np.float32))
                    self.image.set_data(gg)
                    self.fig.canvas.draw()
            if step >= self.max_steps:
                stopped_by_max_iter = True
                break
            not_done = (self.close_list_map.view(-1)[goal_idx].item() <  1.0) or\
            (self.g_map.view(-1)[goal_idx].item() >=  0.1*self.inf)
            R+=1
        if not stopped_by_max_iter:
            for i in range(additional_steps): #now propagating beyong start point 
                self.g_map = torch.min(self.g_map,
                            (self.neights2channels(F.pad(self.g_map,(1,1, 1,1), 'replicate'))  + c_map).min(dim = 1,
                                                    keepdim = True)[0])
                self.close_list_map = torch.max(self.close_list_map, self.open_list_map)
                self.open_list_map = F.relu(F.max_pool2d(self.open_list_map, 3, 
                                                     stride=1, padding=1) - self.close_list_map - self.obstacles) 
        #print (time() -t, 'prop time', step, ' steps')
        #print (self.conv_time, self.close_time, "conv, close time" )
        if return_path:
            t=time()
            out_path, cost = self.reconstructPath()
            #print (time() -t, 'recont time')
            return out_path, cost
        return
    def calculateLocalPathCosts(self, non_obstacle_cost_map = None):
        coords = self.coords
        h = coords.size(2)
        w = coords.size(3)
        obstacles_pd = F.pad(self.obstacles, (1,1, 1,1), 'replicate')         
        if non_obstacle_cost_map is None:
            learned_bias = torch.ones_like(self.obstacles).to(obstacles_pd.device)
        else:
            learned_bias = non_obstacle_cost_map.to(obstacles_pd.device)
        left_diff_sq = self.gx_to_left(F.pad(coords[:,1:2,:,:], (1,1, 0,0), 'replicate'))**2
        right_diff_sq = self.gx_to_right(F.pad(coords[:,1:2,:,:], (1,1, 0,0), 'replicate'))**2
        up_diff_sq = self.gy_to_up(F.pad(coords[:,0:1,:,:], (0,0, 1,1), 'replicate'))**2
        down_diff_sq = self.gy_to_down(F.pad(coords[:,0:1,:,:], (0,0, 1,1), 'replicate'))**2
        out = torch.cat([#Order in from up to down, from left to right, hopefully same as in PyTorch
            torch.sqrt(left_diff_sq + up_diff_sq + self.eps) + self.ob_cost*torch.max(obstacles_pd[:,:,0:h,0:w], obstacles_pd[:,:,1:h+1,1:w+1]),
            torch.sqrt(left_diff_sq + self.eps) + self.ob_cost*torch.max(obstacles_pd[:,:,0:h,1:w+1], obstacles_pd[:,:,1:h+1,1:w+1]),
            torch.sqrt(left_diff_sq + down_diff_sq + self.eps) + self.ob_cost*torch.max(obstacles_pd[:,:,2:h+2,0:w], obstacles_pd[:,:,1:h+1,1:w+1]),
            torch.sqrt(up_diff_sq + self.eps)  + self.ob_cost*torch.max(obstacles_pd[:,:,0:h,1:w+1], obstacles_pd[:,:,1:h+1,1:w+1]),
            0*right_diff_sq + self.ob_cost*obstacles_pd[:,:,1:h+1,1:w+1], #current center
            torch.sqrt(down_diff_sq + self.eps)+ self.ob_cost*torch.max(obstacles_pd[:,:,2:h+2,1:w+1], obstacles_pd[:,:,1:h+1,1:w+1]),
            torch.sqrt(right_diff_sq + up_diff_sq + self.eps) + self.ob_cost*torch.max(obstacles_pd[:,:,0:h,2:w+2], obstacles_pd[:,:,1:h+1,1:w+1]),
            torch.sqrt(right_diff_sq + self.eps) + self.ob_cost*torch.max(obstacles_pd[:,:,1:h+1,2:w+2], obstacles_pd[:,:,1:h+1,1:w+1]),
            torch.sqrt(right_diff_sq + down_diff_sq + self.eps)+ self.ob_cost*torch.max(obstacles_pd[:,:,2:h+2,2:w+2], obstacles_pd[:,:,1:h+1,1:w+1])
             ], dim = 1)
        return out + torch.clamp (learned_bias.expand_as(out), min = 0, max = self.ob_cost)
    def propagate_traversal(self,node_coords, close, g, coords):
        t=time()
        ymin,ymax,xmin,xmax = self.safeROI2d(node_coords[0]-1,
                                 node_coords[0]+2,
                                 node_coords[1]-1,
                                 node_coords[1]+2) 
        #mask = torch.zeros(1,1,ymax-ymin,xmax-xmin)
        self.trav_init_time +=time()-t
        t=time()
        mask = close[:,:,ymin:ymax, xmin:xmax] > 0
        mask[:,:, f2ind(node_coords, 0)-ymin,
                  f2ind(node_coords, 1)-xmin] = 0
        mask = mask > 0
        current_g_cost = g[:,:,ymin:ymax, xmin:xmax][mask].clone()
        if len(current_g_cost.view(-1)) == 0:
            #we are kind surrounded by obstacles, but still need to output something
            mask = torch.relu(1.0 - self.been_there[:,:,ymin:ymax, xmin:xmax])
            mask[:,:, f2ind(node_coords, 0)-ymin,
                   f2ind(node_coords, 1)-xmin] = 0
            mask = mask > 0
            current_g_cost = g[:,:,ymin:ymax, xmin:xmax][mask].clone()
        if len(current_g_cost.view(-1)) > 1:
            current_g_cost = (current_g_cost - torch.min(current_g_cost).item())
            current_g_cost = current_g_cost + 0.41*torch.randperm(len(current_g_cost),
                                       dtype=torch.float32, device = torch.device('cpu'))/(len(current_g_cost))
        self.trav_mask_time+=time() -t
        #
        t=time()
        coords_roi = coords[:,:,ymin:ymax, xmin:xmax]
        out =  self.argmin(current_g_cost,coords_roi[mask.expand_as(coords_roi)])
        self.trav_soft_time+=time() -t
        return out
    def getCleanCostMapAndGoodMask(self):
        good_mask = 1 - F.max_pool2d(self.obstacles, 3, stride=1, padding=1)
        costmap = self.g_map #* self.close_list_map #* good_mask
        obstacle_cost_corrected = 10000.0
        sampling_map = torch.clamp(costmap, min = 0, max = obstacle_cost_corrected)
        return sampling_map, good_mask
    def reconstructPath(self):
        #self.other_loop_time = 0
        #t=time()
        #print ("GOAL IS REACHED!")
        out_path = []
        goal_coords = self.goal_coords.cpu()#(self.coords * self.goal_map.expand_as(self.coords)).sum(dim=2).sum(dim=2).squeeze()
        start_coords = self.start_coords.cpu()#(self.coords * self.start_map.expand_as(self.coords)).sum(dim=2).sum(dim=2).squeeze()
        
        cost = self.g_map[:,:, f2ind(goal_coords, 0),
                               f2ind(goal_coords, 1)]
        #### Traversing
        done = False
        node_coords = goal_coords.cpu()
        out_path.append(node_coords)
        self.been_there = 0*self.been_there.cpu()
        self.been_there[:,:, f2ind(node_coords, 0),
                             f2ind(node_coords, 1)] = 1.0
        self.close_list_map = self.close_list_map.cpu()
        self.g_map = self.g_map.cpu()
        self.coords = self.coords.cpu()
        #print ('non loop time', time() -t )
        count1 = 0
        while not done:
            node_coords = self.propagate_traversal(node_coords, self.close_list_map,
                                                   self.g_map, self.coords)
            #t=time()
            self.been_there[:,:, f2ind(node_coords, 0),
                                 f2ind(node_coords, 1)] = 1.0
            #if len(node_coords) == 0:
            #    print("Warning! Empty next coords")
            #    return out_path, cost
            if (torch.norm(node_coords - out_path[-1], 2).item() < 0.3):
                y = node_coords.flatten()[0].long()
                x = node_coords.flatten()[1].long()
                print(self.g_map[0,0,y-2:y+3,x-2:x+3 ])
                print("loop in out_path",node_coords )
                #torch.save(self.g_map, 'gmap.pt')
                raise ValueError('loop in out_path')
                return out_path, cost
            out_path.append(node_coords)
            done = torch.norm(node_coords - start_coords.cpu(), 2).item() < 0.3
            #self.other_loop_time+=time() -t
            count1+=1
            if count1 > 250:
                break
        return out_path, cost

