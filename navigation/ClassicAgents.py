import numpy as np
import cv2
import torch
import random
import time
from math import ceil,floor, pi, sqrt
from copy import deepcopy
from PIL import Image, ImageOps

from .Utils import *
from .Reprojection import *
from .Reprojection import angleToPi_2_MinusPi_2 as norm_ang
from .Control import command2NumericAction, action2Command
import orbslam2
from .Mappers import DirectDepthMapper, SparseDepthMapper
from .PyTorchPathPlanners import DifferentiableStarPlanner



class Container(object):
    pass

def defaultAgentOptions(default_args = {'agent_type': 'ClassicRGBD'}):
    opts = {}
    #Input resolution
    opts['width']= 256
    opts['height']= 256
    opts['resolution']= [256, 256]
    #SLAM
    opts['slam_vocab_path'] = 'data/ORBvoc.txt'
    opts['slam_settings_path'] = 'data/mp3d3_small1k.yaml'
    if default_args['agent_type'] == 'ClassicRGB':
        opts['slam_settings_path'] = 'data/mp3d3_small5k.yaml'
        opts['height']= 512
        opts['width']= 512
        opts['resolution']= [512, 512]
    if default_args['agent_type'] == 'ClassicStereo':
        opts['slam_settings_path'] = 'data/mp3d3_small1k_stereo.yaml'
        opts['height']= 512
        opts['width']= 512
        opts['resolution']= [512, 512]
    #Camera intristics
    opts['fx']= opts['width']
    opts['fy']= opts['height']
    opts['cx']= opts['fx'] - 1
    opts['cy']= opts['fy'] - 1
    # Depth camera parameters
    opts['minos_depth_near'] = 0.001
    opts['minos_depth_far'] = 4.0
    #Mapping and planning
    opts['map_cell_size'] = 0.1# minos parameters, m
    opts['map_size'] = 40.# meters
    opts['map_resolution'] =   opts['map_cell_size']
    opts['world_camera_height'] = 1.35
    if 'agent' in default_args: 
        if 'eyeHeight' in default_args['agent']:
            opts['world_camera_height'] = 0.6 + float(default_args['agent']['eyeHeight'])/2.
    opts['camera_height'] = opts['world_camera_height']
    opts['beta'] = 100
    opts['preprocess'] = False
    #Points from h_min to h_max contribute to obstacle map
    opts['h_min'] = 0.3#meters
    opts['h_max'] = 0.9*opts['world_camera_height']#meters
    #Points from near_th to far_th contribute to obstacle map
    opts['near_th'] = 0.1#meters
    opts['far_th'] = 2.5#meters
    #Number of 3d points in cell to create obstacle:
    opts['obstacle_threshold'] = float(opts['width']) / 2.0
    opts['obstacle_th'] = opts['obstacle_threshold'] 
    opts['preprocess'] = True
    #parameters for control
    #rotate, if angle to goal is more than,
    opts['angle_threshold'] = float(np.deg2rad(15))
    #Go to waypoint, until reach distance
    opts['pos_threshold'] = 0.15
    #Select waypoint, which is at least this far
    opts['far_pos_threshold'] = 0.5
    #
    opts['device'] = torch.device('cuda:0')
    for k, v in default_args.items():
        opts[k] = v
    return opts

class RandomAgent(object):
    r"""Simplest agent, which returns random actions,
    until reach the goal
    """
    def __init__(self, 
                 num_actions = 3,
                 time_step = 0.1,
                 dist_threshold_to_stop = 0.01,
                 **kwargs):
        super(RandomAgent, self).__init__()
        self.num_actions = num_actions
        self.time_step = time_step
        self.dist_threshold_to_stop = dist_threshold_to_stop
        self.steps = 0
        return
    def reset(self):
        self.steps = 0
        return
    def updateInternalState(self,minos_observation):
        self.obs = minos_observation
        self.steps+=1
        return
    def isGoalReached(self):
        dist = self.obs['observation']['measurements']['distance_to_goal'][0]
        return dist <= self.dist_threshold_to_stop
    def act(self, minos_observation = None, random_prob = 1.0):
        self.updateInternalState(minos_observation)
        # Act
        action = np.zeros(self.num_actions)
        #Check if we are done
        success = self.isGoalReached()
        if not success:
            random_action = np.random.randint(0, self.num_actions, size = (1))
            action[random_action] = 1
        return action, success

class LoggingAgent(RandomAgent):
    r"""Auxilary agent, for purpose of logging and
    calculating SPL, since MINOS does not provide such functionality.
    Agent uses ground truth info, which is not available to the evaluated agents.       
    """   
    def __init__(self, 
                 map_size = 100,
                 map_resolution = 0.1,
                 device = torch.device('cpu'),
                 **kwargs):
        super(LoggingAgent, self).__init__()
        self.map_size_meters = map_size
        self.map_cell_size = map_resolution
        self.device = device
        self.reset()
        return 
    def getCurrentGTOrientationOnGTMap(self):
        return torch.tensor([[self.pose6D[0,0,0], self.pose6D[0,0,2]], 
                             [self.pose6D[0,2,0], self.pose6D[0,2,2]]])
    def getCurrentGTPositionOnGTMap(self, floor = True):
        return projectTPsIntoWorldMap(self.pose6D.view(1,4,4),
                                      self.map_cell_size, 
                                      self.map_size_meters, floor)
    def getMinosGeoDist(self):
        return self.obs['observation']['measurements']['shortest_path_to_goal']['distance']
    def mapSizeInCells(self):
        return int(self.map_size_meters / self.map_cell_size)   
    def initMap2D(self):
        return torch.zeros(1,1,self.mapSizeInCells(),self.mapSizeInCells()).float().to(self.device)
    def updateTravelledDist(self, obs):
        if self.InitGTPoseForBenchmark is None:
            self.InitGTPoseForBenchmark = deepcopy(obs['info']['agent_state'])
            self.InitGTPositionForBenchmark = deepcopy(np.array(self.InitGTPoseForBenchmark['position'])[::-1])
            self.InitGTOrientationForBenchmark = deepcopy(np.array(self.InitGTPoseForBenchmark['orientation'])[::-1])
            self.InitGTAngleOrig = norm_ang(np.arctan2(self.InitGTOrientationForBenchmark[0], self.InitGTOrientationForBenchmark[2]))
            self.CurrentGTPoseForBenchmark = torch.eye(4).float().to(self.device)
        pos = deepcopy(np.array(obs['info']['agent_state']['position'])[::-1])
        ori = deepcopy(np.array(obs['info']['agent_state']['orientation'])[::-1])
        angle =  norm_ang(np.arctan2(ori[0], ori[2]))
        P = np.zeros((3,4)).astype(np.float32)
        P[:,3] = pos - self.InitGTPositionForBenchmark
        P03 =  P[0,3] * np.cos(self.InitGTAngleOrig) - P[2,3] * np.sin(self.InitGTAngleOrig)
        P23 =  P[2,3] * np.cos(self.InitGTAngleOrig) + P[0,3] * np.sin(self.InitGTAngleOrig)
        P[0,3] = P03
        P[2,3] = P23
        P[1,1] = 1.0
        da =  angle - self.InitGTAngleOrig
        P[0,0] = np.cos(da)
        P[2,2] = np.cos(da)
        P[0,2] = np.sin(da)
        P[2,0] = -np.sin(da)
        CurrentGTPoseForBenchmark =  homogenizeP(torch.from_numpy(deepcopy(P)).view(3,4).to(self.device).float())
        step = getPosDiffLength(self.CurrentGTPoseForBenchmark.view(4,4),
                    CurrentGTPoseForBenchmark.view(4,4).to(self.device))
        self.CurrentGTPoseForBenchmark =  CurrentGTPoseForBenchmark
        self.travelledSoFarForEval = self.travelledSoFarForEval + step.cpu().detach().item()
        return
    def setOffsetToGoal(self, observation):
        self.offset_to_goal = torch.tensor(observation['observation']['measurements']['offset_to_goal'])
        self.estimatedGoalPos2D = minosOffsetToGoal2MapGoalPosition(self.offset_to_goal, 
                                                                    self.pose6D.squeeze(),
                                                                    self.map_cell_size, 
                                                                    self.map_size_meters)
        self.estimatedGoalPos6D =  plannedPath2TPs([self.estimatedGoalPos2D],  
                                   self.map_cell_size, 
                                   self.map_size_meters, 
                                   1.0).to(self.device)[0]
        return
    def updateInternalState(self, minos_observation):
        RandomAgent.updateInternalState(self,minos_observation)
        t= time.time()
        #
        if self.GT_modalities is None:
            self.GT_modalities = deepcopy(minos_observation['info']['agent_state'])
            self.pos_init = deepcopy(np.array(self.GT_modalities['position'])[::-1])
            self.ori_init = deepcopy(np.array(self.GT_modalities['orientation'])[::-1])
            self.angle_orig = norm_ang(np.arctan2(self.ori_init[0], self.ori_init[2]))
        pos = deepcopy(np.array(minos_observation['info']['agent_state']['position'])[::-1])
        ori = deepcopy(np.array(minos_observation['info']['agent_state']['orientation'])[::-1])
        angle =  norm_ang(np.arctan2(ori[0], ori[2]))
        P = np.zeros((3,4)).astype(np.float32)
        P[:,3] = pos - self.pos_init
        P03 =  P[0,3] * np.cos(self.angle_orig) - P[2,3] * np.sin(self.angle_orig) 
        P23 =  P[2,3] * np.cos(self.angle_orig) + P[0,3] * np.sin(self.angle_orig) 
        P[0,3] = P03
        P[2,3] = P23
        P[1,1] = 1.0
        da =  angle - self.angle_orig
        P[0,0] = np.cos(da)
        P[2,2] = np.cos(da) 
        P[0,2] = np.sin(da)
        P[2,0] = -np.sin(da)
        self.pose6D = homogenizeP(torch.from_numpy(deepcopy(P)).view(3,4).to(self.device).float())       
        t = time.time()
        self.setOffsetToGoal(minos_observation)
        self.position_history.append(self.pose6D)
        #Mapping
        if self.MapIsHere is None:
            global_map_GT = minos_observation['observation']['map']['data'][:,:,:3].astype(np.uint8)
            w = minos_observation['observation']['map']['width']
            h = minos_observation['observation']['map']['height']
            global_map_GT = global_map_GT.reshape(h,w,3)
            global_map_GT = (global_map_GT[:,:,2] ==127).astype(np.uint8) * 255
            goal_cell_pose = minos_observation['info']['goal'][0]['cell']
            goal_pose = minos_observation['info']['goal'][0]['position']
            b1 = -10 * goal_pose[0] + goal_cell_pose['i']
            b2 = -10 * goal_pose[2] + goal_cell_pose['j']
            ag_i = int(minos_observation['info']['agent_state']['position'][0] * 10 + b1)
            ag_j = int(minos_observation['info']['agent_state']['position'][2] * 10 + b2)
            rotmap = Image.fromarray(global_map_GT.astype(np.uint8))
            delta_w = max(ag_i, w - ag_i)
            delta_h = max(ag_j, h - ag_j)
            new_size = max(delta_w, delta_h) * 2
            padding = (new_size//2  - ag_i,
                       new_size//2  - ag_j, 
                       new_size//2  - (w - ag_i),
                       new_size//2  - (h - ag_j))
            padded_map = ImageOps.expand(rotmap, padding)
            agent_ori = np.array(minos_observation['info']['agent_state']['orientation'])[::-1]
            angle = np.degrees(norm_ang(-np.pi/2.0 + np.arctan2(agent_ori[0], agent_ori[2])))
            rotmap = padded_map.rotate(angle, expand=True)
            rotmap = np.array(rotmap).astype(np.float32)
            rotmap = rotmap / rotmap.max()
            cur_obs = torch.from_numpy(deepcopy(rotmap[:,::-1])).to(self.device).squeeze()
            ctr = cur_obs.size(0)//2
            cur_obs[ctr-1:ctr+2, ctr-1:ctr+2] = 0
            pd2 = (self.map2DObstacles.size(2) - cur_obs.size(0) )// 2 
            GP = self.estimatedGoalPos2D
            gy = GP[0,0].long()
            gx = GP[0,1].long()
            self.map2DObstacles[0,0,pd2:pd2+cur_obs.size(0),pd2:pd2+cur_obs.size(1)] = self.map2DObstacles[0,0,pd2:pd2+cur_obs.size(0),pd2:pd2+cur_obs.size(1)].clone() + 120*cur_obs
            self.map2DObstacles[0,0,gy, gx] = self.map2DObstacles[0,0,gy, gx]*0
            self.GTMAP = self.map2DObstacles
            self.MapIsHere = True
        return True
    def act(self, minos_observation = None, random_prob = 1.0):
        self.updateInternalState(minos_observation)
        self.updateTravelledDist(self.obs)
        return True
    def rawMap2PlanerReady(self, rawmap, start_map, goal_map):
        map1 = (rawmap / float(128))**2
        map1 = torch.clamp(map1, min=0, max=1.0) - start_map - F.max_pool2d(goal_map, 3, stride=1, padding=1)
        return torch.relu(map1)
    def reset(self):
        self.InitGTPoseForBenchmark = None
        self.travelledSoFarForEval = 0.0
        self.MapIsHere = None
        self.map2DObstacles = self.initMap2D()
        self.pose6D_GT_history = []
        self.GT_modalities = None
        self.position_history = []
        self.GTMAP = self.map2DObstacles
        return
        
class BlindAgent(RandomAgent):
    def __init__(self, 
                 pos_threshold = 0.01,
                 angle_threshold = float(np.deg2rad(15)),
                 **kwargs):
        super(BlindAgent, self).__init__()
        self.reset()
        self.pos_th = pos_threshold
        self.angle_th = angle_threshold
        return 
    def decideWhatToDo(self):
        distance_to_goal = self.obs['observation']['measurements']['distance_to_goal'][0]
        rot_vector_to_goal = np.array(self.obs['observation']['measurements']['direction_to_goal'])
        angle_to_goal = -norm_ang(np.arctan2(rot_vector_to_goal[0], rot_vector_to_goal[2]))
        #print (angle_to_goal)
        command = "Idle"
        if distance_to_goal <= self.pos_th:
            return command
        if (abs(angle_to_goal) < self.angle_th):
            command = "forwards"
        else:
            if (angle_to_goal > 0) and (angle_to_goal < pi):
                command = 'turnLeft'
            elif (angle_to_goal > pi):
                command = 'turnRight'
            elif (angle_to_goal < 0) and (angle_to_goal > -pi):
                command = 'turnRight'
            else:
                command = 'turnLeft'
        return command
    def act(self, minos_observation = None, random_prob = 0.1):
        self.updateInternalState(minos_observation)
        # Act
        action = np.zeros(self.num_actions)
        success = self.isGoalReached()
        if success:
            return action, success
        command = self.decideWhatToDo()
        random_action = np.random.randint(0, self.num_actions, size = (1))
        act_randomly =  np.random.uniform(0,1,1)  < random_prob
        if act_randomly:
            action[random_action] = 1
        else:
            action = command2NumericAction(command)
        return action, success
    
class ClassicAgentWithDepth(RandomAgent):
    def __init__(self, 
                 slam_vocab_path = '',
                 slam_settings_path = '',
                 pos_threshold = 0.15,
                 angle_threshold = float(np.deg2rad(15)),
                 far_pos_threshold = 0.5,
                 obstacle_th = 40,
                 map_size = 80,
                 map_resolution = 0.1,
                 device = torch.device('cpu'),
                 **kwargs):
        super(ClassicAgentWithDepth, self).__init__(**kwargs)
        self.slam_vocab_path = slam_vocab_path
        self.slam_settings_path = slam_settings_path
        self.slam = orbslam2.System(slam_vocab_path,slam_settings_path, orbslam2.Sensor.RGBD)
        self.slam.set_use_viewer(False)
        self.slam.initialize()
        self.tracking_is_OK = False
        self.device = device
        self.map_size_meters = map_size
        self.map_cell_size = map_resolution
        self.waypoint_id = 0
        self.slam_to_world = 1.0
        self.pos_th = pos_threshold
        self.far_pos_threshold = far_pos_threshold
        self.angle_th = angle_threshold
        self.obstacle_th = obstacle_th
        self.plannedWaypoints = []
        self.mapper = DirectDepthMapper(**kwargs)
        self.planner = DifferentiableStarPlanner(**kwargs)
        self.timing = False
        self.reset()
        return
    def reset(self):
        super(ClassicAgentWithDepth, self).reset()
        self.offset_to_goal = None
        self.waypointPose6D = None
        self.unseen_obstacle = False
        self.action_history = []
        self.plannedWaypoints = []
        self.map2DObstacles = self.initMap2D()
        n,ch, height, width = self.map2DObstacles.size()
        self.coordinatesGrid = generate_2dgrid(height, 
                                       width,False).to(self.device)
        self.pose6D = self.initPose6D()
        self.action_history = []
        self.pose6D_history = []
        self.position_history = []
        self.planned2Dpath = torch.zeros((0))
        self.slam.reset()
        self.toDoList = []
        if self.waypoint_id > 92233720368547758: #not used here, reserved for hybrid agent
            self.waypoint_id = 0
        return
    def updateInternalState(self, minos_observation):
        super(ClassicAgentWithDepth, self).updateInternalState(minos_observation)
        rgb, depth, cur_time = self.rgbAndDAndTimeFromObservation(minos_observation)
        t= time.time()
        try:
            self.slam.process_image_rgbd(rgb, depth, cur_time)
            if self.timing:
                print(time.time() -t , 'ORB_SLAM2')
            self.tracking_is_OK = str(self.slam.get_tracking_state()) == "OK"
        except:
            print ("Warning!!!! ORBSLAM processing frame error")
            self.tracking_is_OK = False
        if not self.tracking_is_OK:
            self.reset()
        t = time.time()
        self.setOffsetToGoal(minos_observation)
        if self.tracking_is_OK:
            trajectory_history = np.array(self.slam.get_trajectory_points())
            self.pose6D = homogenizeP(torch.from_numpy(trajectory_history[-1])[1:].view(3,4).to(self.device)).view(1,4,4)
            self.trajectory_history = trajectory_history
            if (len(self.position_history) > 1):
                previous_step = getPosDiffLength(self.pose6D.view(4,4),
                                torch.from_numpy(self.position_history[-1]).view(4,4).to(self.device))
                if action2Command(self.action_history[-1]) == "forwards":
                    self.unseen_obstacle = previous_step.item() <=  0.001 #hardcoded threshold for not moving
        current_obstacles = self.mapper(torch.from_numpy(depth).to(self.device).squeeze(),self.pose6D).to(self.device)
        self.current_obstacles = current_obstacles
        self.map2DObstacles =  torch.max(self.map2DObstacles, 
                                               current_obstacles.unsqueeze(0).unsqueeze(0))
        if self.timing:
            print(time.time() -t , 'Mapping')
        return True
    def initPose6D(self):
        return torch.eye(4).float().to(self.device)
    def mapSizeInCells(self):
        return int(self.map_size_meters / self.map_cell_size)
    def initMap2D(self):
        return torch.zeros(1,1,self.mapSizeInCells(),self.mapSizeInCells()).float().to(self.device)
    def getCurrentOrientationOnMap(self):
        self.pose6D = self.pose6D.view(1,4,4)
        return torch.tensor([[self.pose6D[0,0,0], self.pose6D[0,0,2]], 
                             [self.pose6D[0,2,0], self.pose6D[0,2,2]]])
    def getCurrentPositionOnMap(self, do_floor = True):
        return projectTPsIntoWorldMap(self.pose6D.view(1,4,4), self.map_cell_size, self.map_size_meters, do_floor)
    def act(self, minos_observation, random_prob = 0.1):
        # Update internal state
        t = time.time()
        cc= 0
        updateIsOK = self.updateInternalState(minos_observation)
        while not updateIsOK:
            updateIsOK = self.updateInternalState(minos_observation)
            cc+=1
            if cc>2:
                break
        if self.timing:
            print (time.time() - t, " s, update internal state")
        self.position_history.append(self.pose6D.detach().cpu().numpy().reshape(1,4,4))
        success = self.isGoalReached()
        if success:
            self.action_history.append(action)
            return action, success
        # Plan action
        t = time.time()
        self.planned2Dpath, self.plannedWaypoints = self.planPath()
        if self.timing:
            print (time.time() - t, " s, Planning")
        t = time.time()
        # Act
        if self.waypointPose6D is None:
            self.waypointPose6D = self.getValidWaypointPose6D()
        if self.isWaypointReached(self.waypointPose6D) or not self.tracking_is_OK:
            self.waypointPose6D = self.getValidWaypointPose6D()
            self.waypoint_id+=1
        action = self.decideWhatToDo()
        #May be random?
        random_idx = torch.randint(3, size = (1,)).long()
        random_action = torch.zeros(self.num_actions).cpu().numpy()
        random_action[random_idx] = 1
        what_to_do =  np.random.uniform(0,1,1)
        if what_to_do < random_prob:
            action = random_action
        if self.timing:
            print (time.time() - t, " s, plan 2 action")
        self.action_history.append(action)
        return action, success
    def isWaypointGood(self,pose6d):
        Pinit = self.pose6D.squeeze()
        dist_diff = getPosDiffLength(Pinit,pose6d)
        valid = dist_diff > self.far_pos_threshold
        return valid.item()
    def isWaypointReached(self,pose6d):
        Pinit = self.pose6D.squeeze()
        dist_diff = getPosDiffLength(Pinit,pose6d)
        reached = dist_diff <= self.pos_th  
        return reached.item()
    def getWaypointDistDir(self):
        angle = getDirection(self.pose6D.squeeze(), self.waypointPose6D.squeeze(),0,0)
        dist = getPosDiffLength(self.pose6D.squeeze(), self.waypointPose6D.squeeze())
        return torch.cat([dist.view(1,1), torch.sin(angle).view(1,1),torch.cos(angle).view(1,1)], dim = 1)
    def getValidWaypointPose6D(self):
        good_waypoint_found = False
        Pinit = self.pose6D.squeeze()
        Pnext = self.plannedWaypoints[0]
        while not self.isWaypointGood(Pnext):
            if (len(self.plannedWaypoints) > 1):
                self.plannedWaypoints = self.plannedWaypoints[1:]
                Pnext = self.plannedWaypoints[0]
            else:
                Pnext = self.estimatedGoalPos6D.squeeze()
                break
        return Pnext
    def setOffsetToGoal(self, observation):
        self.offset_to_goal = torch.tensor(observation['observation']['measurements']['offset_to_goal'])
        self.estimatedGoalPos2D = minosOffsetToGoal2MapGoalPosition(self.offset_to_goal, 
                                                                    self.pose6D.squeeze(),
                                                                    self.map_cell_size, 
                                                                    self.map_size_meters)
        self.estimatedGoalPos6D =  plannedPath2TPs([self.estimatedGoalPos2D],  
                                   self.map_cell_size, 
                                   self.map_size_meters, 
                                   1.0).to(self.device)[0]
        return
    def rgbAndDAndTimeFromObservation(self,minos_observation):
        rgb = minos_observation['observation']['sensors']['color']['data'][:,:,:3]
        depth = None
        if 'depth' in minos_observation['observation']['sensors']:
            depth = minos_observation['observation']['sensors']['depth']['data']
        cur_time = minos_observation['observation']['time']
        return rgb, depth, cur_time
    def prevPlanIsNotValid(self):
        if len(self.planned2Dpath) == 0:
            return True
        pp = torch.cat(self.planned2Dpath).detach().cpu().view(-1,2)
        binary_map = self.map2DObstacles.squeeze().detach() >= self.obstacle_th
        obstacles_on_path =  (binary_map[pp[:,0].long(),pp[:,1].long()]).long().sum().item() > 0
        return obstacles_on_path# obstacles_nearby or  obstacles_on_path
    def rawMap2PlanerReady(self, rawmap, start_map, goal_map):
        map1 = (rawmap / float(self.obstacle_th))**2
        map1 = torch.clamp(map1, min=0, max=1.0) - start_map - F.max_pool2d(goal_map, 3, stride=1, padding=1)
        return torch.relu(map1)
    def planPath(self, overwrite = False):
        t=time.time()
        if (not self.prevPlanIsNotValid()) and (not overwrite) and (len(self.plannedWaypoints) > 0):
            return self.planned2Dpath, self.plannedWaypoints         
        self.waypointPose6D = None
        current_pos = self.getCurrentPositionOnMap()
        start_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        start_map[0,0,current_pos[0,0].long(),current_pos[0,1].long()] = 1.0
        goal_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        goal_map[0,0,self.estimatedGoalPos2D[0,0].long(),self.estimatedGoalPos2D[0,1].long()] = 1.0
        path, cost = self.planner(self.rawMap2PlanerReady(self.map2DObstacles,  start_map, goal_map).to(self.device),
                                  self.coordinatesGrid.to(self.device), goal_map.to(self.device),  start_map.to(self.device))
        if len(path) == 0:
            return path, []
        if self.timing:
            print(time.time() - t, ' s, Planning')
        t=time.time()
        plannedWaypoints = plannedPath2TPs(path,  
                                   self.map_cell_size, 
                                   self.map_size_meters, 
                                   1.0, False).to(self.device)
        return path, plannedWaypoints
    def plannerPrediction2Command(self, Pnext):
        command = "Idle"
        Pinit = self.pose6D.squeeze()
        d_angle_rot_th = self.angle_th
        pos_th = self.pos_th
        if getPosDiffLength(Pinit,Pnext) <= pos_th:
            return command
        d_angle = angleToPi_2_MinusPi_2(getDirection(Pinit, Pnext, ang_th = d_angle_rot_th, pos_th = pos_th))
        if (abs(d_angle) < d_angle_rot_th):
            command = "forwards"
        else:
            if (d_angle > 0) and (d_angle < pi):
                command = 'turnLeft'
            elif (d_angle > pi):
                command = 'turnRight'
            elif (d_angle < 0) and (d_angle > -pi):
                command = 'turnRight'
            else:
                command = 'turnLeft'
        return command
    def decideWhatToDo(self):
        action = None
        if self.isGoalReached():
            action = command2NumericAction("Idle")
            print ("!!!!!!!!!!!! Goal reached!")
            return action
        if self.unseen_obstacle:
            command = 'turnRight'
            return command2NumericAction(command)
        command = "Idle"
        command = self.plannerPrediction2Command(self.waypointPose6D)
        return command2NumericAction(command)

class ClassicAgentWithStereo(RandomAgent):
    def __init__(self, 
                 slam_vocab_path = '',
                 slam_settings_path = '',
                 pos_threshold = 0.15,
                 angle_threshold = float(np.deg2rad(15)),
                 far_pos_threshold = 0.5,
                 obstacle_th = 40,
                 map_size = 80,
                 map_resolution = 0.1,
                 device = torch.device('cpu'),
                 **kwargs):
        super(ClassicAgentWithStereo, self).__init__(**kwargs)
        self.slam_vocab_path = slam_vocab_path
        self.slam_settings_path = slam_settings_path
        self.slam = orbslam2.System(slam_vocab_path,slam_settings_path, orbslam2.Sensor.STEREO)
        self.slam.set_use_viewer(False)
        self.slam.initialize()
        self.tracking_is_OK = False
        self.device = device
        self.map_size_meters = map_size
        self.map_cell_size = map_resolution
        self.waypoint_id = 0
        self.slam_to_world = 1.0
        self.pos_th = pos_threshold
        self.far_pos_threshold = far_pos_threshold
        self.angle_th = angle_threshold
        self.obstacle_th = obstacle_th
        self.plannedWaypoints = []
        self.mapper = DirectDepthMapper(**kwargs)
        self.planner = DifferentiableStarPlanner(**kwargs)
        self.timing = False
        window_size = 5                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        self.left_matcher = cv2.StereoSGBM_create(
             minDisparity=0,
             numDisparities=128,             # max_disp has to be dividable by 16 f. E. HH 192, 256
             blockSize=5,
             P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
             P2=32 * 3 * window_size ** 2,
             disp12MaxDiff=1,
             uniquenessRatio=15,
             speckleWindowSize=0,
             speckleRange=2,
             preFilterCap=63,
             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.2
        visual_multiplier = 1.0
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
        self.wls_filter.setLambda(lmbda)
        self.wls_filter.setSigmaColor(sigma)
        self.reset()
        return
    def reset(self):
        super(ClassicAgentWithStereo, self).reset()
        self.offset_to_goal = None
        self.waypointPose6D = None
        self.unseen_obstacle = False
        self.action_history = []
        self.plannedWaypoints = []
        self.map2DObstacles = self.initMap2D()
        n,ch, height, width = self.map2DObstacles.size()
        self.coordinatesGrid = generate_2dgrid(height, 
                                       width,False).to(self.device)
        self.pose6D = self.initPose6D()
        self.action_history = []
        self.pose6D_history = []
        self.position_history = []
        self.planned2Dpath = torch.zeros((0))
        self.slam.reset()
        self.toDoList = []
        if self.waypoint_id > 92233720368547758: #not used here, reserved for hybrid agent
            self.waypoint_id = 0
        return
    def updateInternalState(self, minos_observation):
        super(ClassicAgentWithStereo, self).updateInternalState(minos_observation)
        rgb, depth, cur_time, right = self.rgbAndDAndTimeFromObservation(minos_observation)
        t= time.time()
        try:
            self.slam.process_image_stereo(rgb, right, cur_time)
            if self.timing:
                print(time.time() -t , 'ORB_SLAM2')
            self.tracking_is_OK = str(self.slam.get_tracking_state()) == "OK"
        except:
            print ("Warning!!!! ORBSLAM processing frame error")
            self.tracking_is_OK = False
        if not self.tracking_is_OK:
            self.reset()
        t = time.time()
        self.setOffsetToGoal(minos_observation)
        if self.tracking_is_OK:
            trajectory_history = np.array(self.slam.get_trajectory_points())
            self.pose6D = homogenizeP(torch.from_numpy(trajectory_history[-1])[1:].view(3,4).to(self.device)).view(1,4,4)
            self.trajectory_history = trajectory_history
            if (len(self.position_history) > 1):
                previous_step = getPosDiffLength(self.pose6D.view(4,4),
                                torch.from_numpy(self.position_history[-1]).view(4,4).to(self.device))
                if action2Command(self.action_history[-1]) == "forwards":
                    self.unseen_obstacle = previous_step.item() <=  0.001 #hardcoded threshold for not moving
        current_obstacles = self.mapper(torch.from_numpy(depth).to(self.device).squeeze(),self.pose6D).to(self.device)
        self.current_obstacles = current_obstacles
        self.map2DObstacles =  torch.max(self.map2DObstacles, 
                                               current_obstacles.unsqueeze(0).unsqueeze(0))
        if self.timing:
            print(time.time() -t , 'Mapping')
        return True
    def initPose6D(self):
        return torch.eye(4).float().to(self.device)
    def mapSizeInCells(self):
        return int(self.map_size_meters / self.map_cell_size)
    def initMap2D(self):
        return torch.zeros(1,1,self.mapSizeInCells(),self.mapSizeInCells()).float().to(self.device)
    def getCurrentOrientationOnMap(self):
        self.pose6D = self.pose6D.view(1,4,4)
        return torch.tensor([[self.pose6D[0,0,0], self.pose6D[0,0,2]], 
                             [self.pose6D[0,2,0], self.pose6D[0,2,2]]])
    def getCurrentPositionOnMap(self, do_floor = True):
        return projectTPsIntoWorldMap(self.pose6D.view(1,4,4), self.map_cell_size, self.map_size_meters, do_floor)
    def act(self, minos_observation, random_prob = 0.1):
        # Update internal state
        t = time.time()
        cc= 0
        updateIsOK = self.updateInternalState(minos_observation)
        while not updateIsOK:
            updateIsOK = self.updateInternalState(minos_observation)
            cc+=1
            if cc>2:
                break
        if self.timing:
            print (time.time() - t, " s, update internal state")
        self.position_history.append(self.pose6D.detach().cpu().numpy().reshape(1,4,4))
        success = self.isGoalReached()
        if success:
            self.action_history.append(action)
            return action, success
        # Plan action
        t = time.time()
        self.planned2Dpath, self.plannedWaypoints = self.planPath()
        if self.timing:
            print (time.time() - t, " s, Planning")
        t = time.time()
        # Act
        if self.waypointPose6D is None:
            self.waypointPose6D = self.getValidWaypointPose6D()
        if self.isWaypointReached(self.waypointPose6D) or not self.tracking_is_OK:
            self.waypointPose6D = self.getValidWaypointPose6D()
            self.waypoint_id+=1
        action = self.decideWhatToDo()
        #May be random?
        random_idx = torch.randint(3, size = (1,)).long()
        random_action = torch.zeros(self.num_actions).cpu().numpy()
        random_action[random_idx] = 1
        what_to_do =  np.random.uniform(0,1,1)
        if what_to_do < random_prob:
            action = random_action
        if self.timing:
            print (time.time() - t, " s, plan 2 action")
        self.action_history.append(action)
        return action, success
    def isWaypointGood(self,pose6d):
        Pinit = self.pose6D.squeeze()
        dist_diff = getPosDiffLength(Pinit,pose6d)
        valid = dist_diff > self.far_pos_threshold
        return valid.item()
    def isWaypointReached(self,pose6d):
        Pinit = self.pose6D.squeeze()
        dist_diff = getPosDiffLength(Pinit,pose6d)
        reached = dist_diff <= self.pos_th  
        return reached.item()
    def getWaypointDistDir(self):
        angle = getDirection(self.pose6D.squeeze(), self.waypointPose6D.squeeze(),0,0)
        dist = getPosDiffLength(self.pose6D.squeeze(), self.waypointPose6D.squeeze())
        return torch.cat([dist.view(1,1), torch.sin(angle).view(1,1),torch.cos(angle).view(1,1)], dim = 1)
    def getValidWaypointPose6D(self):
        good_waypoint_found = False
        Pinit = self.pose6D.squeeze()
        Pnext = self.plannedWaypoints[0]
        while not self.isWaypointGood(Pnext):
            if (len(self.plannedWaypoints) > 1):
                self.plannedWaypoints = self.plannedWaypoints[1:]
                Pnext = self.plannedWaypoints[0]
            else:
                Pnext = self.estimatedGoalPos6D.squeeze()
                break
        return Pnext
    def setOffsetToGoal(self, observation):
        self.offset_to_goal = torch.tensor(observation['observation']['measurements']['offset_to_goal'])
        self.estimatedGoalPos2D = minosOffsetToGoal2MapGoalPosition(self.offset_to_goal, 
                                                                    self.pose6D.squeeze(),
                                                                    self.map_cell_size, 
                                                                    self.map_size_meters)
        self.estimatedGoalPos6D =  plannedPath2TPs([self.estimatedGoalPos2D],  
                                   self.map_cell_size, 
                                   self.map_size_meters, 
                                   1.0).to(self.device)[0]
        return
    def rgbAndDAndTimeFromObservation(self,minos_observation):
        rgb = minos_observation['observation']['sensors']['color']['data'][:,:,:3]
        right = minos_observation['observation']['sensors']['right']['data'][:,:,:3]
        imgL = rgb.astype(np.uint8)
        imgR = right.astype(np.uint8)
        displ = self.left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = self.right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        #From http://timosam.com/python_opencv_depthimage
        filteredImg = self.wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
        dd = np.clip(filteredImg/16., 0, 100)
        dd[dd<0] = 0
        #plt.imshow(dd)
        baseline = 0.2
        fx = 512.
        depth = baseline * fx / ((dd+1e-15))
        depth = depth / 0.8
        depth[depth > 3.0] = 0
        depth[depth < 0.1] = 0
        cur_time = minos_observation['observation']['time']
        return rgb, depth.astype(np.float32), cur_time, right
    def prevPlanIsNotValid(self):
        if len(self.planned2Dpath) == 0:
            return True
        pp = torch.cat(self.planned2Dpath).detach().cpu().view(-1,2)
        binary_map = self.map2DObstacles.squeeze().detach() >= self.obstacle_th
        obstacles_on_path =  (binary_map[pp[:,0].long(),pp[:,1].long()]).long().sum().item() > 0
        return obstacles_on_path# obstacles_nearby or  obstacles_on_path
    def rawMap2PlanerReady(self, rawmap, start_map, goal_map):
        map1 = (rawmap / float(self.obstacle_th))**2
        map1 = torch.clamp(map1, min=0, max=1.0) - start_map - F.max_pool2d(goal_map, 3, stride=1, padding=1)
        return torch.relu(map1)
    def planPath(self, overwrite = False):
        t=time.time()
        if (not self.prevPlanIsNotValid()) and (not overwrite) and (len(self.plannedWaypoints) > 0):
            return self.planned2Dpath, self.plannedWaypoints         
        self.waypointPose6D = None
        current_pos = self.getCurrentPositionOnMap()
        start_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        start_map[0,0,current_pos[0,0].long(),current_pos[0,1].long()] = 1.0
        goal_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        goal_map[0,0,self.estimatedGoalPos2D[0,0].long(),self.estimatedGoalPos2D[0,1].long()] = 1.0
        path, cost = self.planner(self.rawMap2PlanerReady(self.map2DObstacles,  start_map, goal_map).to(self.device),
                                  self.coordinatesGrid.to(self.device), goal_map.to(self.device),  start_map.to(self.device))
        if len(path) == 0:
            return path, []
        if self.timing:
            print(time.time() - t, ' s, Planning')
        t=time.time()
        plannedWaypoints = plannedPath2TPs(path,  
                                   self.map_cell_size, 
                                   self.map_size_meters, 
                                   1.0, False).to(self.device)
        return path, plannedWaypoints
    def plannerPrediction2Command(self, Pnext):
        command = "Idle"
        Pinit = self.pose6D.squeeze()
        d_angle_rot_th = self.angle_th
        pos_th = self.pos_th
        if getPosDiffLength(Pinit,Pnext) <= pos_th:
            return command
        d_angle = angleToPi_2_MinusPi_2(getDirection(Pinit, Pnext, ang_th = d_angle_rot_th, pos_th = pos_th))
        if (abs(d_angle) < d_angle_rot_th):
            command = "forwards"
        else:
            if (d_angle > 0) and (d_angle < pi):
                command = 'turnLeft'
            elif (d_angle > pi):
                command = 'turnRight'
            elif (d_angle < 0) and (d_angle > -pi):
                command = 'turnRight'
            else:
                command = 'turnLeft'
        return command
    def decideWhatToDo(self):
        action = None
        if self.isGoalReached():
            action = command2NumericAction("Idle")
            print ("!!!!!!!!!!!! Goal reached!")
            return action
        if self.unseen_obstacle:
            command = 'turnRight'
            return command2NumericAction(command)
        command = "Idle"
        command = self.plannerPrediction2Command(self.waypointPose6D)
        return command2NumericAction(command)
class ClassicAgentRGB(RandomAgent):
    def __init__(self, 
                 slam_vocab_path = '',
                 slam_settings_path = '',
                 pos_threshold = 0.15,
                 angle_threshold = float(np.deg2rad(15)),
                 far_pos_threshold = 0.5,
                 obstacle_th = 40,
                 map_size = 80,
                 map_resolution = 0.1,
                 device = torch.device('cpu'),
                 **kwargs):
        super(ClassicAgentRGB, self).__init__(**kwargs)
        self.slam_vocab_path = slam_vocab_path
        self.slam_settings_path = slam_settings_path
        self.slam = orbslam2.System(slam_vocab_path,slam_settings_path, orbslam2.Sensor.MONOCULAR)
        self.slam.set_use_viewer(False)
        self.slam.initialize()
        self.tracking_is_OK = False
        self.device = device
        self.map_size_meters = map_size
        self.map_cell_size = map_resolution
        self.waypoint_id = 0
        self.slam_to_world = 1.0
        self.pos_th = pos_threshold
        self.far_pos_threshold = far_pos_threshold
        self.angle_th = angle_threshold
        self.obstacle_th = obstacle_th
        self.plannedWaypoints = []
        self.mapper = SparseDepthMapper(**kwargs)
        self.planner = DifferentiableStarPlanner(**kwargs)
        self.timing = False
        self.reset()
        return
    def reset(self):
        super(ClassicAgentRGB, self).reset()
        self.offset_to_goal = None
        self.waypointPose6D = None
        self.unseen_obstacle = False
        self.action_history = []
        self.plannedWaypoints = []
        self.map2DObstacles = self.initMap2D()
        n,ch, height, width = self.map2DObstacles.size()
        self.coordinatesGrid = generate_2dgrid(height, 
                                       width,False).to(self.device)
        self.pose6D = self.initPose6D()
        self.action_history = []
        self.pose6D_history = []
        self.position_history = []
        self.planned2Dpath = torch.zeros((0))
        self.slam.reset()
        self.slam.slam_to_world = None
        self.slam.slam_camera_height = None
        self.slam.floor_level = None
        self.toDoList = []
        if self.waypoint_id > 92233720368547758: #not used here, reserved for hybrid agent
            self.waypoint_id = 0
        return
    def set_slam_scale(self):
        if len(self.toDoList) == 0:
            return
        if ((str(self.slam.get_tracking_state()) == "OK") 
            and (self.slam.slam_to_world is None) 
            and (self.toDoList[0] == 'lookUp')): #We are looking down
            new_map_pts = np.array(self.slam.get_tracked_mappoints()).reshape(-1,3)
            self.slam.floor_level = np.median(new_map_pts[:,1])
            self.slam.slam_camera_height = np.abs(self.slam.floor_level)
            self.slam.slam_to_world = self.world_camera_height /  self.slam.slam_camera_height
        return
    def updateInternalState(self, minos_observation):
        super(ClassicAgentRGB, self).updateInternalState(minos_observation)
        rgb, _ , cur_time = self.rgbAndDAndTimeFromObservation(minos_observation)
        t= time.time()
        self.setOffsetToGoal(self.obs)
        try:
            self.slam.process_image_mono(rgb, cur_time)
            if self.timing:
                print(time.time() -t , 'ORB_SLAM2')
        except:
            print ("Warning!!!! ORBSLAM processing frame error")
            self.tracking_is_OK = False
            self.reset()
        self.tracking_is_OK = str(self.slam.get_tracking_state()) == "OK"
        t =time.time()
        self.set_slam_scale()
        if self.tracking_is_OK:
            trajectory_history = np.array(self.slam.get_trajectory_points())
            self.pose6D = homogenizeP(torch.from_numpy(trajectory_history[-1]).float()[1:].view(3,4).to(self.device)).view(1,4,4)
            self.trajectory_history = trajectory_history
            if (len(self.position_history) > 1):
                previous_step = getPosDiffLength(self.pose6D.view(4,4),
                                torch.from_numpy(self.position_history[-1]).view(4,4).to(self.device))
                if action2Command(self.action_history[-1]) == "forwards":
                    self.unseen_obstacle = previous_step.item() <=  0.001 #hardcoded threshold for not moving
        if (str(self.slam.get_tracking_state()) == "OK") and (self.slam.slam_to_world is not None):
            current_map_pts = np.array(self.slam.get_tracked_mappoints()).reshape(-1,3)
            self.current_obstacles = self.mapper(torch.from_numpy(current_map_pts).float().to(self.device).squeeze(),
                                            self.pose6D).to(self.device)
            self.map2DObstacles =  torch.max(self.map2DObstacles, 
                                                   current_obstacles.unsqueeze(0).unsqueeze(0))#[0]
        if self.timing:
            print(time.time() -t , 'Mapping')
        return True
    def decideWhatToDo(self):
        action = None
        if self.isGoalReached():
            action = command2NumericAction("Idle")
            print ("!!!!!!!!!!!! Goal reached!")
            return action
        if (str(self.slam.get_tracking_state())  == 'NOT_INITIALIZED'):
            if random.random() > 0.3:
                return command2NumericAction('forwards')
            else:
                return command2NumericAction('backwards')
        if len(self.toDoList) > 0:
            command = self.toDoList.pop(0)
            return command2NumericAction(command)
        if (str(self.slam.get_tracking_state()) == "OK") and (self.slam.slam_to_world is None) :
            print ("ORBSLAM INITIALIZATION")
            for i in range(100//5):
                self.toDoList.append('lookDown')
                if i%3 ==0:
                    self.toDoList.append('forwards')
                if i%3 ==1:
                    self.toDoList.append('backwards')
            for i in range(90//5):
                self.toDoList.append('lookUp')
            command = self.toDoList.pop(0)
            return command2NumericAction(command)
        command = "Idle"
        command = self.plannerPrediction2Command(self.waypointPose6D)
        return command2NumericAction(command)
    def initPose6D(self):
        return torch.eye(4).float().to(self.device)
    def mapSizeInCells(self):
        return int(self.map_size_meters / self.map_cell_size)
    def initMap2D(self):
        return torch.zeros(1,1,self.mapSizeInCells(),self.mapSizeInCells()).float().to(self.device)
    def getCurrentOrientationOnMap(self):
        self.pose6D = self.pose6D.view(1,4,4)
        return torch.tensor([[self.pose6D[0,0,0], self.pose6D[0,0,2]], 
                             [self.pose6D[0,2,0], self.pose6D[0,2,2]]])
    def getCurrentPositionOnMap(self, do_floor = True):
        return projectTPsIntoWorldMap(self.pose6D.view(1,4,4), self.map_cell_size, self.map_size_meters, do_floor)
    def act(self, minos_observation, random_prob = 0.1):
        # Update internal state
        t = time.time()
        cc= 0
        updateIsOK = self.updateInternalState(minos_observation)
        while not updateIsOK:
            updateIsOK = self.updateInternalState(minos_observation)
            cc+=1
            if cc>2:
                break
        if self.timing:
            print (time.time() - t, " s, update internal state")
        self.position_history.append(self.pose6D.detach().cpu().numpy().reshape(1,4,4))
        success = self.isGoalReached()
        if success:
            self.action_history.append(action)
            return action, success
        # Plan action
        t = time.time()
        self.planned2Dpath, self.plannedWaypoints = self.planPath()
        if self.timing:
            print (time.time() - t, " s, Planning")
        t = time.time()
        # Act
        if self.waypointPose6D is None:
            self.waypointPose6D = self.getValidWaypointPose6D()
        if self.isWaypointReached(self.waypointPose6D) or not self.tracking_is_OK:
            self.waypointPose6D = self.getValidWaypointPose6D()
            self.waypoint_id+=1
        action = self.decideWhatToDo()
        #May be random?
        random_idx = torch.randint(3, size = (1,)).long()
        random_action = torch.zeros(self.num_actions).cpu().numpy()
        random_action[random_idx] = 1
        what_to_do =  np.random.uniform(0,1,1)
        if what_to_do < random_prob:
            action = random_action
        if self.timing:
            print (time.time() - t, " s, plan 2 action")
        self.action_history.append(action)
        return action, success
    def isWaypointGood(self,pose6d):
        Pinit = self.pose6D.squeeze()
        dist_diff = getPosDiffLength(Pinit,pose6d)
        valid = dist_diff > self.far_pos_threshold
        return valid.item()
    def isWaypointReached(self,pose6d):
        Pinit = self.pose6D.squeeze()
        dist_diff = getPosDiffLength(Pinit,pose6d)
        reached = dist_diff <= self.pos_th  
        return reached.item()
    def getWaypointDistDir(self):
        angle = getDirection(self.pose6D.squeeze(), self.waypointPose6D.squeeze(),0,0)
        dist = getPosDiffLength(self.pose6D.squeeze(), self.waypointPose6D.squeeze())
        return torch.cat([dist.view(1,1), torch.sin(angle).view(1,1),torch.cos(angle).view(1,1)], dim = 1)
    def getValidWaypointPose6D(self):
        good_waypoint_found = False
        Pinit = self.pose6D.squeeze()
        Pnext = self.plannedWaypoints[0]
        while not self.isWaypointGood(Pnext):
            if (len(self.plannedWaypoints) > 1):
                self.plannedWaypoints = self.plannedWaypoints[1:]
                Pnext = self.plannedWaypoints[0]
            else:
                Pnext = self.estimatedGoalPos6D.squeeze()
                break
        return Pnext
    def setOffsetToGoal(self, observation):
        self.offset_to_goal = torch.tensor(observation['observation']['measurements']['offset_to_goal'])
        self.estimatedGoalPos2D = minosOffsetToGoal2MapGoalPosition(self.offset_to_goal, 
                                                                    self.pose6D.squeeze(),
                                                                    self.map_cell_size, 
                                                                    self.map_size_meters)
        self.estimatedGoalPos6D =  plannedPath2TPs([self.estimatedGoalPos2D],  
                                   self.map_cell_size, 
                                   self.map_size_meters, 
                                   1.0).to(self.device)[0]
        return
    def rgbAndDAndTimeFromObservation(self,minos_observation):
        rgb = minos_observation['observation']['sensors']['color']['data'][:,:,:3]
        depth = None
        if 'depth' in minos_observation['observation']['sensors']:
            depth = minos_observation['observation']['sensors']['depth']['data']
        cur_time = minos_observation['observation']['time']
        return rgb, depth, cur_time
    def prevPlanIsNotValid(self):
        if len(self.planned2Dpath) == 0:
            return True
        pp = torch.cat(self.planned2Dpath).detach().cpu().view(-1,2)
        binary_map = self.map2DObstacles.squeeze().detach() >= self.obstacle_th
        obstacles_on_path =  (binary_map[pp[:,0].long(),pp[:,1].long()]).long().sum().item() > 0
        return obstacles_on_path# obstacles_nearby or  obstacles_on_path
    def rawMap2PlanerReady(self, rawmap, start_map, goal_map):
        map1 = (rawmap / float(self.obstacle_th))**2
        map1 = torch.clamp(map1, min=0, max=1.0) - start_map - F.max_pool2d(goal_map, 3, stride=1, padding=1)
        return torch.relu(map1)
    def planPath(self, overwrite = False):
        t=time.time()
        if (not self.prevPlanIsNotValid()) and (not overwrite) and (len(self.plannedWaypoints) > 0):
            return self.planned2Dpath, self.plannedWaypoints         
        self.waypointPose6D = None
        current_pos = self.getCurrentPositionOnMap()
        start_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        start_map[0,0,current_pos[0,0].long(),current_pos[0,1].long()] = 1.0
        goal_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        goal_map[0,0,self.estimatedGoalPos2D[0,0].long(),self.estimatedGoalPos2D[0,1].long()] = 1.0
        path, cost = self.planner(self.rawMap2PlanerReady(self.map2DObstacles,  start_map, goal_map).to(self.device),
                                  self.coordinatesGrid.to(self.device), goal_map.to(self.device),  start_map.to(self.device))
        if len(path) == 0:
            return path, []
        if self.timing:
            print(time.time() - t, ' s, Planning')
        t=time.time()
        plannedWaypoints = plannedPath2TPs(path,  
                                   self.map_cell_size, 
                                   self.map_size_meters, 
                                   1.0, False).to(self.device)
        return path, plannedWaypoints
    def plannerPrediction2Command(self, Pnext):
        command = "Idle"
        Pinit = self.pose6D.squeeze()
        d_angle_rot_th = self.angle_th
        pos_th = self.pos_th
        if getPosDiffLength(Pinit,Pnext) <= pos_th:
            return command
        d_angle = angleToPi_2_MinusPi_2(getDirection(Pinit, Pnext, ang_th = d_angle_rot_th, pos_th = pos_th))
        if (abs(d_angle) < d_angle_rot_th):
            command = "forwards"
        else:
            if (d_angle > 0) and (d_angle < pi):
                command = 'turnLeft'
            elif (d_angle > pi):
                command = 'turnRight'
            elif (d_angle < 0) and (d_angle > -pi):
                command = 'turnRight'
            else:
                command = 'turnLeft'
        return command
   
class CheatingAgentWithGTPose(ClassicAgentWithDepth):
    def __init__(self, 
                 pos_threshold = 0.15,
                 angle_threshold = float(np.deg2rad(15)),
                 far_pos_threshold = 0.5,
                 obstacle_th = 40,
                 map_size = 80,
                 map_resolution = 0.1,
                 device = torch.device('cpu'),
                 **kwargs):
        RandomAgent.__init__(self, **kwargs)
        self.tracking_is_OK = True
        self.device = device
        self.map_size_meters = map_size
        self.map_cell_size = map_resolution
        self.waypoint_id = 0
        self.pos_th = pos_threshold
        self.far_pos_threshold = far_pos_threshold
        self.angle_th = angle_threshold
        self.obstacle_th = obstacle_th
        self.plannedWaypoints = []
        self.mapper = DirectDepthMapper(**kwargs)
        self.planner = DifferentiableStarPlanner(**kwargs)
        self.timing = False
        self.reset()
        return
    def reset(self):
        self.steps = 0
        self.GT_modalities = None
        self.offset_to_goal = None
        self.waypointPose6D = None
        self.unseen_obstacle = False
        self.action_history = []
        self.plannedWaypoints = []
        self.map2DObstacles = self.initMap2D()
        n,ch, height, width = self.map2DObstacles.size()
        self.coordinatesGrid = generate_2dgrid(height, 
                                       width,False).to(self.device)
        self.pose6D = self.initPose6D()
        self.action_history = []
        self.pose6D_history = []
        self.position_history = []
        self.planned2Dpath = torch.zeros((0))
        self.MapIsHere = None
        self.toDoList = []
        if self.waypoint_id > 92233720368547758: #not used here, reserved for hybrid agent
            self.waypoint_id = 0
        return
    def updateInternalState(self, minos_observation):
        RandomAgent.updateInternalState(self,minos_observation)
        rgb, depth, cur_time = self.rgbAndDAndTimeFromObservation(minos_observation)
        t= time.time()
        #
        if self.GT_modalities is None:
            self.GT_modalities = deepcopy(minos_observation['info']['agent_state'])
            self.pos_init = deepcopy(np.array(self.GT_modalities['position'])[::-1])
            self.ori_init = deepcopy(np.array(self.GT_modalities['orientation'])[::-1])
            self.angle_orig = norm_ang(np.arctan2(self.ori_init[0], self.ori_init[2]))
        pos = deepcopy(np.array(minos_observation['info']['agent_state']['position'])[::-1])
        ori = deepcopy(np.array(minos_observation['info']['agent_state']['orientation'])[::-1])
        angle =  norm_ang(np.arctan2(ori[0], ori[2]))
        P = np.zeros((3,4)).astype(np.float32)
        P[:,3] = pos - self.pos_init
        P03 =  P[0,3] * np.cos(self.angle_orig) - P[2,3] * np.sin(self.angle_orig) 
        P23 =  P[2,3] * np.cos(self.angle_orig) + P[0,3] * np.sin(self.angle_orig) 
        P[0,3] = P03
        P[2,3] = P23
        P[1,1] = 1.0
        da =  angle - self.angle_orig
        P[0,0] = np.cos(da)
        P[2,2] = np.cos(da) 
        P[0,2] = np.sin(da)
        P[2,0] = -np.sin(da)
        self.pose6D = homogenizeP(torch.from_numpy(deepcopy(P)).view(3,4).to(self.device).float())       
        t = time.time()
        self.setOffsetToGoal(minos_observation)
        if (len(self.position_history) > 1):
            previous_step = getPosDiffLength(self.pose6D.view(4,4),
                            torch.from_numpy(self.position_history[-1]).view(4,4).to(self.device))
            if action2Command(self.action_history[-1]) == "forwards":
                self.unseen_obstacle = previous_step.item() <=  0.001 #hardcoded threshold for not moving
        self.position_history.append(self.pose6D)
        current_obstacles = self.mapper(torch.from_numpy(depth).to(self.device).squeeze(),self.pose6D).to(self.device)
        self.current_obstacles = current_obstacles
        self.map2DObstacles =  torch.max(self.map2DObstacles, 
                                               current_obstacles.unsqueeze(0).unsqueeze(0))
        if self.timing:
            print(time.time() -t , 'Mapping')
        return True

class CheatingAgentWithGTPoseAndMap(CheatingAgentWithGTPose):
    def updateInternalState(self, minos_observation):
        RandomAgent.updateInternalState(self,minos_observation)
        rgb, depth, cur_time = self.rgbAndDAndTimeFromObservation(minos_observation)
        t= time.time()
        #
        if self.GT_modalities is None:
            self.GT_modalities = deepcopy(minos_observation['info']['agent_state'])
            self.pos_init = deepcopy(np.array(self.GT_modalities['position'])[::-1])
            self.ori_init = deepcopy(np.array(self.GT_modalities['orientation'])[::-1])
            self.angle_orig = norm_ang(np.arctan2(self.ori_init[0], self.ori_init[2]))
        pos = deepcopy(np.array(minos_observation['info']['agent_state']['position'])[::-1])
        ori = deepcopy(np.array(minos_observation['info']['agent_state']['orientation'])[::-1])
        angle =  norm_ang(np.arctan2(ori[0], ori[2]))
        P = np.zeros((3,4)).astype(np.float32)
        P[:,3] = pos - self.pos_init
        P03 =  P[0,3] * np.cos(self.angle_orig) - P[2,3] * np.sin(self.angle_orig) 
        P23 =  P[2,3] * np.cos(self.angle_orig) + P[0,3] * np.sin(self.angle_orig) 
        P[0,3] = P03
        P[2,3] = P23
        P[1,1] = 1.0
        da =  angle - self.angle_orig
        P[0,0] = np.cos(da)
        P[2,2] = np.cos(da) 
        P[0,2] = np.sin(da)
        P[2,0] = -np.sin(da)
        self.pose6D = homogenizeP(torch.from_numpy(deepcopy(P)).view(3,4).to(self.device).float())       
        t = time.time()
        self.setOffsetToGoal(minos_observation)
        if (len(self.position_history) > 1):
            previous_step = getPosDiffLength(self.pose6D.view(4,4),
                            torch.from_numpy(self.position_history[-1]).view(4,4).to(self.device))
            if action2Command(self.action_history[-1]) == "forwards":
                self.unseen_obstacle = previous_step.item() <=  0.001 #hardcoded threshold for not moving
        self.position_history.append(self.pose6D)
        #Mapping
        if self.MapIsHere is None:
            global_map_GT = minos_observation['observation']['map']['data'][:,:,:3].astype(np.uint8)
            w = minos_observation['observation']['map']['width']
            h = minos_observation['observation']['map']['height']
            global_map_GT = global_map_GT.reshape(h,w,3)
            global_map_GT = (global_map_GT[:,:,2] ==127).astype(np.uint8) * 255
            goal_cell_pose = minos_observation['info']['goal'][0]['cell']
            goal_pose = minos_observation['info']['goal'][0]['position']
            b1 = -10 * goal_pose[0] + goal_cell_pose['i']
            b2 = -10 * goal_pose[2] + goal_cell_pose['j']
            ag_i = int(minos_observation['info']['agent_state']['position'][0] * 10 + b1)
            ag_j = int(minos_observation['info']['agent_state']['position'][2] * 10 + b2)
            rotmap = Image.fromarray(global_map_GT.astype(np.uint8))
            delta_w = max(ag_i, w - ag_i)
            delta_h = max(ag_j, h - ag_j)
            new_size = max(delta_w, delta_h) * 2
            padding = (new_size//2  - ag_i,
                       new_size//2  - ag_j, 
                       new_size//2  - (w - ag_i),
                       new_size//2  - (h - ag_j))
            padded_map = ImageOps.expand(rotmap, padding)
            agent_ori = np.array(minos_observation['info']['agent_state']['orientation'])[::-1]
            angle = np.degrees(norm_ang(-np.pi/2.0 + np.arctan2(agent_ori[0], agent_ori[2])))
            rotmap = padded_map.rotate(angle, expand=True)
            rotmap = np.array(rotmap).astype(np.float32)
            rotmap = rotmap / rotmap.max()
            cur_obs = torch.from_numpy(deepcopy(rotmap[:,::-1])).to(self.device).squeeze()
            ctr = cur_obs.size(0)//2
            cur_obs[ctr-1:ctr+2, ctr-1:ctr+2] = 0
            pd2 = (self.map2DObstacles.size(2) - cur_obs.size(0) )// 2 
            GP = self.estimatedGoalPos2D
            gy = GP[0,0].long()
            gx = GP[0,1].long()
            self.map2DObstacles[0,0,pd2:pd2+cur_obs.size(0),pd2:pd2+cur_obs.size(1)] = self.map2DObstacles[0,0,pd2:pd2+cur_obs.size(0),pd2:pd2+cur_obs.size(1)].clone() + 120*cur_obs
            self.map2DObstacles[0,0,gy, gx] = self.map2DObstacles[0,0,gy, gx]*0
            self.MapIsHere = True
        if self.timing:
            print(time.time() -t , 'Mapping')
        return True
