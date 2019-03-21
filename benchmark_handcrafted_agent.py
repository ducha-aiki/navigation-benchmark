def PatchNumpy():
    # https://github.com/Marcello-Sega/pytim/blob/f16643653d8415038dbeb52f580e353f255cda6e/pytim/patches.py
    # this try/except block patches numpy and provides _validate_lengths 
    # to skimage<=1.14.1
    import numpy
    try:
        numpy.lib.arraypad._validate_lengths
    except AttributeError:
        def patch_validate_lengths(ar, crop_width):
            return numpy.lib.arraypad._as_pairs(crop_width, ar.ndim, as_index=True)
        numpy.lib.arraypad._validate_lengths = patch_validate_lengths
PatchNumpy()
import sys
import os.path
import time
import gc
from time import sleep
from copy import deepcopy
import numpy as np
import argparse
import imageio
from PIL import Image
import random

import gym
import gym_minos
from minos.config.sim_args import parse_sim_args

from navigation.Utils import gettimestr, gettimestr
from navigation.visualizingUtils import VideoLogWriter
from navigation.ClassicAgents import *
from navigation.Metrics import calcSPL
from navigation.LoggerBench import TestLogger


def experiment_name(agent, args):
    return  '_'.join([args['env_config'],
                      args['episode_schedule'],
                      args['exp_name'], 
                      args['agent_type']]) 

parser = argparse.ArgumentParser(description='MINOS gym benchmark')

parser.add_argument('--agent-type', type=str,
                    default="Random",
                    help='Random, Blind, ClassicRGB, ClassicStereoCNN, ClassicStereoOpenCV, ClassicMonoDepth, ClassicRGBD, ClassicGTPose, ClassicGTMapGTPose')#Logging 
parser.add_argument('--logdir-prefix', type=str,
                    default='LOGS/',
                    help='init driving net from pretrained')
parser.add_argument('--save-video', type=str2bool,
                    default=False,
                    help='')
parser.add_argument('--exp-name', type=str,
                    default='',
                    help='Experiment name')
parser.add_argument('--timing', type=str2bool,
                    default=False,
                    help='show time for steps')
# Benchmark opts
parser.add_argument('--start-from-episode', type=int,
                    default=0,
                    help='skip this number of episodes')
parser.add_argument('--num-episodes-per-scene', type=int,
                    default=10,
                    help='Number of diffent random start and end locations')
parser.add_argument('--episode-schedule', type=str,
                    default='test',
                    help='Split to use. Possible are: train, val, test. Paper reports results on test')
parser.add_argument('--seed', type=int,
                    default=42,
                    help='Random seed')


args = parse_sim_args(parser)

#For storing video and logs
#get map and depth anyway for video storing.
#Agent has its own setup
args['observations']['map']  = True
args['observations']['depth'] = True
args['log_action_trace'] = True
#args['width'] = 512
#args['height'] = 512
#args['resolution'] = [512,512]

opts = defaultAgentOptions(args)


RANDOM_SEED = args['seed']

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


if __name__ == "__main__": 
    draw_map = False
    if args['agent_type'] == 'Random':
        agent = RandomAgent(**opts)
    elif args['agent_type'] == 'Blind':
        agent = BlindAgent(**opts)
    elif args['agent_type'] == 'ClassicRGB':
        draw_map = True
        agent = ClassicAgentRGB(**opts)
    elif args['agent_type'] == 'ClassicStereoCNN':
        draw_map = True
        agent = ClassicAgentWithCNNStereoDepth(**opts)
    elif args['agent_type'] == 'ClassicStereoOpenCV':
        draw_map = True
        agent = ClassicAgentWithStereo(**opts)
    elif args['agent_type'] == 'ClassicRGBD':
        draw_map = True
        agent = ClassicAgentWithDepth(**opts)
    elif args['agent_type'] == 'ClassicGTPose':
        draw_map = True
        agent = CheatingAgentWithGTPose(**opts)
    elif args['agent_type'] == 'ClassicGTMapGTPose':
        draw_map = True
        agent = CheatingAgentWithGTPoseAndMap(**opts)
    else:
        raise ValueError(args['agent_type'] + ' inknown type of agent. Try Random, Blind, ClassicRGB, ClassicRGBD, ClassicStereoCNN, ClassicStereoOpenCV, ClassicMonoDepth, ClassicGTPose, ClassicGTMapGTPose')
    logging_agent = LoggingAgent(**opts)
    del opts['device']
    out_dir_name = os.path.join(args['logdir_prefix'], experiment_name(agent, args) + gettimestr())
    if not os.path.isdir(out_dir_name):
        os.makedirs(out_dir_name)
    logger1 = TestLogger(out_dir_name)
    logger1.add_init_args(opts)
    env = gym.make('indoor-v0')
    env.configure(opts)
    logger1.add_sim_params(env.simulator.params) 
    
    num_good = 0
    num_failed_map_inits = 0
    num_episodes = env._sim.get_episode_scheduler(env._sim.curr_schedule).num_states() * args['num_episodes_per_scene']
    episodes_done = 0
    i_episode=0
    shortest_paths_length = []
    real_paths_length = []
    successes = []
    coefs = []
    times = []
    SPL = 0.0
    for epps in range(num_episodes):
        print('Starting episode %d' % i_episode, 'out of', num_episodes)
        env.simulator.move_to(tilt=0)
        observation = env.reset()
        counter1 = 0
        #Skip episodes, which has no shortest_path_to_goal, i.e are path is untraversable.
        while not 'distance' in observation['observation']['measurements']['shortest_path_to_goal'].keys():
            print ('Warning, no shortest path exists, trying to reset.')
            #print (observation)
            observation = env.reset()
            counter1+=1
            if counter1 > 10:
                action = np.zeros(8)
                observation, reward, done, info = env.step(action)
                print ('Skipping episode, because there is no shortest path for it')
                counter1 = 0
        if (opts['start_from_episode'] > 0):
            while i_episode < opts['run_specific']:
                action = np.zeros(8)
                observation = env.reset()
                observation, reward, done, info = env.step(action)
                i_episode+=1
        #Reinit map size for agent. It does not affect results, but saves memory for small maps. 
        global_map_GT = observation['observation']['map']['data'][:,:,:3].astype(np.uint8)
        h = observation['observation']['map']['height']
        w = observation['observation']['map']['width']
        map_size_in_meters = findMapSize(h,w)
        agent.map_size_meters =  map_size_in_meters
        logging_agent.map_size_meters = map_size_in_meters
        try:
            agent.mapper.map_size_meters = agent.map_size_meters
        except:
            #If agent does not have mapper, e.g. RandomAgent
            pass
        map_in_cells = getMapSizeInCells(agent.map_size_meters, opts['map_cell_size'])
        agent.reset()
        logging_agent.reset()
        gt_shortest_path = float(observation['observation']['measurements']['shortest_path_to_goal']['distance'])
        init_dist = observation['observation']['measurements']['distance_to_goal'][0]
        print ("MINOS shortest path = ", gt_shortest_path)
        shortest_paths_length.append(gt_shortest_path)
        done = False
        num_steps = 0
        if opts['save_video']:
            ep_base_fname = os.path.join(out_dir_name, str(i_episode))
            VIDEO_WRITER = VideoLogWriter(ep_base_fname, map_size_in_cells = map_in_cells, draw_map = draw_map)
        action = None
        while not done:
            tt=time.time()
            with torch.no_grad():
                action, agent_thinks_its_done = agent.act(observation)
                dummy = logging_agent.act(observation)
            #### Drawing 
            t=time.time()
            if opts['save_video']:
                VIDEO_WRITER.add_frame(observation, agent, logging_agent)
                ####  End of Drawing 
                if opts['timing']:
                    print (time.time() - t, ' s video save')
            t=time.time()
            observation, reward, env_success, info = env.step(action)
            if opts['timing']:
                print (time.time() - t, ' s MINOS step')
            num_steps += 1
            if num_steps % 20 == 0:
                print (num_steps, time.time() - tt, ' s per step')
            done = (env_success or (num_steps >= 500) or agent_thinks_its_done)
            if done:
                real_path = logging_agent.travelledSoFarForEval
                real_paths_length.append(real_path)
                times.append(float(num_steps))
                successes.append(float(observation['success']))
                coef = gt_shortest_path / (max(gt_shortest_path, real_path))
                coefs.append(coef)
                print("Episode finished after {} steps; success={}, travelled path = {},  path to shortest path = {}".format(num_steps, observation['success'], real_path, coef))
                print ("SPL = ", calcSPL(coefs, successes))
                if  observation['success']:
                    num_good+=1
                episodes_done+=1
        if opts['save_video']:
            VIDEO_WRITER.finish()
            agent.reset()
            logging_agent.reset()
        print (episodes_done , "done", num_good, "success")
        print ('success rate = ', float(num_good)/float(episodes_done), 
               'average num_steps =', np.array(times).mean())
        i_episode+=1
        logger1.add_episode_stats(epps,
                                  env._sim.scene_id,
                                  real_path,  
                                  gt_shortest_path,
                                  num_steps,
                                  observation['success'],  h, w, init_dist )
        logger1.save()
        logger1.save_csv()
    print ('Final SPL=',calcSPL(coefs, successes),
           'success rate = ', float(num_good)/float(episodes_done), 
           'average num_steps =', np.array(times).mean())
