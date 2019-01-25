import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from copy import deepcopy
from math import floor
from matplotlib.patches import Wedge
import io
from PIL import Image
from time import time
from PIL import ImageFont,ImageDraw
import imageio
from .Reprojection import projectTPsIntoWorldMap_numpy as NP_project
from .Reprojection import minosOffsetToGoal2MapGoalPosition


def drawMapGoalAgentAda(fig,ax, image, omap, goalpos, agentpos, agentori, plan_path, done_path, waypoint = None):
    t=time()
    while len(ax.lines) > 0:
        ax.lines = []
    while len(ax.artists) > 0:
        ax.artists = []
    h,w = omap.shape
    offset_h = 0
    offset_w = 0
    show_h, show_w = h,w
    if show_h == h:
        image.set_data(omap)
    else:
        image.set_data(omap[offset_h:offset_h+show_h,offset_w:offset_w+show_w])
    AP = agentpos.squeeze()
    GP = goalpos.squeeze()
    if len(plan_path) > 1:
        ax.plot(np.array(plan_path[:,1]) - offset_w, 
                show_h - (np.array(plan_path[:,0]) - offset_h), 
                        'o', linewidth=5.0, markersize=10, color='red')
    if len(done_path) > 1:
        ax.plot(np.array(done_path[:,1]) - offset_w, 
                show_h - (np.array(done_path[:,0]) - offset_h), 
                 'o', linewidth=5.0, markersize=10,  color='green')
    draw_goal = plt.Circle((GP[1] - offset_w, show_h - (GP[0] - offset_h)), 3, color='yellow', fill=True)
    draw_agent = plt.Circle((AP[1] - offset_w,show_h - (AP[0] - offset_h)), 3, color='blue', fill=True)
    agent_angle = -(np.degrees(np.arctan2(agentori[1,0], agentori[0,0])) + 90)
    draw_agent_fov = Wedge((AP[1] - offset_w,show_h - (AP[0] - offset_h)), 10, 
                           agent_angle - 45, 
                           agent_angle + 45,
                           color="blue", alpha=0.5)
    t=time()
    ax.add_artist(draw_goal)
    ax.add_artist(draw_agent)
    ax.add_artist(draw_agent_fov)
    if waypoint is not None:
        CP = waypoint.squeeze()
        draw_waypoint = plt.Circle((CP[1] - offset_w,show_h - (CP[0] - offset_h)), 3, color='orange', fill=False)
        ax.add_artist(draw_waypoint)
    fig.canvas.draw()
    #print(time() - t, 's, draw')
    return 

def drawMapGoalAgent(fig,ax, image, omap, goalpos, agentpos, agentori, plan_path, done_path, waypoint = None):
    t=time()
    while len(ax.lines) > 0:
        ax.lines = []
    while len(ax.artists) > 0:
        ax.artists = []
    h,w = omap.shape
    offset_h = 0
    offset_w = 0
    show_h, show_w = h,w
    if show_h == h:
        image.set_data(omap)
    else:
        image.set_data(omap[offset_h:offset_h+show_h,offset_w:offset_w+show_w])
    AP = agentpos.squeeze()
    GP = goalpos.squeeze()
    if len(plan_path) > 1:
        ax.plot(np.array(plan_path[:,1]) - offset_w, 
                show_h - (np.array(plan_path[:,0]) - offset_h), 
                        '.', linewidth=0.2, color='red')
    if len(done_path) > 1:
        ax.plot(np.array(done_path[:,1]) - offset_w, 
                show_h - (np.array(done_path[:,0]) - offset_h), 
                 '.', linewidth=0.2, color='green')
    draw_goal = plt.Circle((GP[1] - offset_w, show_h - (GP[0] - offset_h)), 3, color='yellow', fill=True)
    draw_agent = plt.Circle((AP[1] - offset_w,show_h - (AP[0] - offset_h)), 3, color='blue', fill=True)
    agent_angle = -(np.degrees(np.arctan2(agentori[1,0], agentori[0,0])) + 90)
    draw_agent_fov = Wedge((AP[1] - offset_w,show_h - (AP[0] - offset_h)), 10, 
                           agent_angle - 45, 
                           agent_angle + 45,
                           color="blue", alpha=0.5)
    t=time()
    ax.add_artist(draw_goal)
    ax.add_artist(draw_agent)
    ax.add_artist(draw_agent_fov)
    if waypoint is not None:
        CP = waypoint.squeeze()
        draw_waypoint = plt.Circle((CP[1] - offset_w,show_h - (CP[0] - offset_h)), 3, color='orange', fill=False)
        ax.add_artist(draw_waypoint)
    fig.canvas.draw()
    #print(time() - t, 's, draw')
    return 

def fig2img ( fig ):
    w,h = fig.canvas.get_width_height()
    return  np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(w, h, 3)

def prepareImgForVideo(rgb, depth, global_map, status):
    dr = np.expand_dims(depth,2)
    depth3 = np.concatenate([dr, dr, dr], axis = 2)
    upper = np.concatenate([rgb, depth3], axis = 1)
    pil_rgb = Image.fromarray(upper)
    if len(status) > 0:
        draw = ImageDraw.Draw(pil_rgb)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 24, encoding="unic")
        except:
            font = ImageFont.load_default()
        draw.text((30, 30),status,(255,255,255),font=font)
    out_img =  np.concatenate([np.array(pil_rgb), global_map], axis = 0)
    return out_img

def initMapImgFig(init_map):
    fig,ax = plt.subplots(1,1,figsize=(50, 50))
    fig.tight_layout(pad=0)
    h,w = init_map.shape
    offset_h = 0
    offset_w = 0
    show_h, show_w = h,w
    image = ax.imshow(init_map,extent=[0, show_h, 0, show_w])#, animated=True)
    ax.invert_yaxis()
    fig.canvas.draw()
    return fig, ax, image
def getVideoLoggingImg(pos_history, 
                       slam_to_world,
                       map_cell_size,
                       map_size,
                       obstacle_map,
                       goalpos,
                       agent_pos,
                       agent_ori,
                       planned_path,
                       observation,
                       fig,ax, image,
                       status,
                       waypoint = None):
    depth_preprocess = observation['observation']['sensors']['depth']['data']
    depth_preprocess = (depth_preprocess/(1e-7 + depth_preprocess.max()) * 255).astype(np.uint8)
    rgbimg = observation['observation']['sensors']['color']['data'][:,:,:3]
    h,w = depth_preprocess.shape
    if True:
        if (type(pos_history) is list ) and (len(pos_history) > 2):
            try:
                ph = np.stack(pos_history)
            except:
                for i in  pos_history:
                    print (i)
                sys.exit(0)
            already_path = ph.reshape(-1,4,4)
        else:
            already_path = []
        already_path_map = NP_project(already_path, 
                                                slam_to_world,
                                                map_cell_size,
                                                map_size)
        drawMapGoalAgent(fig,ax, image,
                        obstacle_map,
                        goalpos, 
                        agent_pos,
                        agent_ori,
                        planned_path,
                        already_path_map,
                        waypoint)
        map2save = np.array(Image.fromarray(fig2img(fig)).resize((2*w, 2*h), Image.BILINEAR))
    else:#except Exception as e:
        print(e)
        map2save = np.array(Image.fromarray(fig2img(fig)).resize((2*w, 2*h), Image.BILINEAR))
    video_frame = prepareImgForVideo(rgbimg,
                                    depth_preprocess,
                                    map2save ,status)
    return video_frame

class VideoLogWriter():
    def __init__(self, dirname, 
                 map_size_in_cells =200,
                 draw_map = True):
        self.dirname = dirname
        self.draw_map = draw_map
        self.map_in_cells = map_size_in_cells
        self.init_map_for_video = np.eye(map_size_in_cells,dtype = np.float32) * 50
        self.RGB_VIDEO_WRITER = imageio.get_writer(dirname + "_RGB.mp4", fps=10)
        self.DEPTH_VIDEO_WRITER = imageio.get_writer(dirname + "_depth.mp4", fps=10)
        if self.draw_map:
            self.fig,self.ax,self.image = initMapImgFig(self.init_map_for_video)
            self.MAP_VIDEO_WRITER = imageio.get_writer(dirname + "_map.mp4", fps=10)
        self.figGT, self.axGT,self.imageGT = initMapImgFig(self.init_map_for_video)
        self.GT_MAP_VIDEO_WRITER = imageio.get_writer(dirname + "_GTmap.mp4", fps=10)
        return
    def add_frame(self, observation, agent, logging_agent):                ###RGB and depth
        self.RGB_VIDEO_WRITER.append_data(observation['observation']['sensors']['color']['data'][:,:,:3])
        depth1 = observation['observation']['sensors']['depth']['data']
        depth1 = np.clip(depth1, 0,4)
        depth1 = (255 * (depth1/4.0)).astype(np.uint8)
        self.DEPTH_VIDEO_WRITER.append_data(depth1)
        if self.draw_map:
            #Map
            try:
                pp = agent.planned2Dpath
                pp = torch.cat(pp).detach().cpu().numpy().reshape(-1,2)
            except:
                pp = []
            pos1 = agent.getCurrentPositionOnMap().detach().cpu().numpy()
            
            ori1 = agent.getCurrentOrientationOnMap().detach().cpu().numpy()
            map1 = torch.zeros_like(logging_agent.map2DObstacles)
            try:
                waypoint = agent.getWaypointPositionOnMap().detach().cpu().numpy()
                already_path = np.stack(agent.position_history).reshape(-1,4,4)
                stw=agent.slam_to_world
            except:
                waypoint = None
                stw=1.0
                already_path = []
            already_path_map = NP_project(already_path, 
                                                    1.0,
                                                    logging_agent.map_cell_size,
                                                    logging_agent.map_size_meters)
            drawMapGoalAgentAda(self.fig,self.ax, self.image,
                        30*agent.rawMap2PlanerReady(agent.map2DObstacles,map1,map1).detach().cpu().squeeze().numpy(),
                        agent.estimatedGoalPos2D.detach().cpu().numpy(), 
                        pos1,
                        ori1,
                        pp,
                        already_path_map,
                        waypoint)
            map2 = fig2img(self.fig)
            mh,mw,mch = map2.shape
            self.MAP_VIDEO_WRITER.append_data(map2)
        #######GT map
        pos1 = logging_agent.getCurrentGTPositionOnGTMap().detach().cpu().numpy()
        ori1 = logging_agent.getCurrentGTOrientationOnGTMap().detach().cpu().numpy()
        map1_gt = torch.zeros_like(logging_agent.map2DObstacles)
        waypoint = None
        try:
            already_path =  np.stack(logging_agent.pose6D_GT_history).reshape(-1,4,4)
        except:
            already_path = []
        already_path_map = NP_project(already_path, 
                                                1.0,
                                                logging_agent.map_cell_size,
                                                logging_agent.map_size_meters)
        drawMapGoalAgentAda(self.figGT,self.axGT, self.imageGT,
                    100*logging_agent.rawMap2PlanerReady(logging_agent.GTMAP,       map1_gt,map1_gt).detach().cpu().squeeze().numpy(),
                    minosOffsetToGoal2MapGoalPosition(logging_agent.offset_to_goal, 
                                                        logging_agent.pose6D.squeeze(),
                                                        logging_agent.map_cell_size, 
                                                        logging_agent.map_size_meters).detach().cpu().numpy(), 
                    pos1,
                    ori1,
                    [],
                    already_path_map,
                    None)
        map2 = fig2img(self.figGT)
        self.GT_MAP_VIDEO_WRITER.append_data(map2)
        ####  End of Drawing 
        return
    def finish(self):
        self.RGB_VIDEO_WRITER.close()
        self.DEPTH_VIDEO_WRITER.close()
        if self.draw_map:
            self.GT_MAP_VIDEO_WRITER.close()
            self.MAP_VIDEO_WRITER.close()
        return
        
        
        
       
        
        
        
        