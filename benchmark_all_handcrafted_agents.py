import os
import shutil
import time
import subprocess
from os.path import expanduser
from shutil import copyfile

n_runs =10
LOGDIR = 'logs/benchmark_classic/'
VIDEO=False

def gettimestr():
    return time.strftime("%Y-%m-%d--%H_%M_%S", time.gmtime())
try:
    os.makedirs(LOGDIR)
except:
    pass
agents = [ 'Blind','ClassicRGBD', 'ClassicRGB', 'ClassicGTPose', 'ClassicGTMapGTPose']

envs = ["pointgoal_mp3d_m",
        'pointgoal_suncg_mf',
        'pointgoal_suncg_me']           

for env in envs:
    for ag in agents:
        print ("Testing", ag, "in", env)
        t=time.time()
        if "mp3d" in env:
            n_runs = 20
        else:
            n_runs = 10
        run_string = 'python3 -utt benchmark_handcrafted_agent.py --env_config ' + env + ' --logdir-prefix=' +  LOGDIR +  " --save-video=" + str(VIDEO) + ' --agent-type=' + ag +  ' --episode-schedule=test --num-episodes-per-scene=' + str(n_runs) + ' 2>&1 | tee ' + LOGDIR+ag+env+ "_" + gettimestr()+ '_verblog.log'
        subprocess.call(run_string,shell=True)
        print ('done in', time.time() - t, 'sec')