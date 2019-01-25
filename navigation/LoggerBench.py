import pickle
import csv
from copy import deepcopy
import os

class TestLogger():
    def __init__(self, logdir = "logs/"):
        self.episodes_detalied = []
        self.episodes_stats = []
        self.init_args = {}
        self.logdir = logdir
        self.detailed_fname = 'detailed_log.pickle'
        self.csvfname = 'short_stats.csv'
        self.num_eps = 0
        return
    def add_init_args(self,args):
        self.init_args = deepcopy(args)
        return
    def add_sim_params(self,params):
        self.sim_params = deepcopy(params)
        return 
    #def add_observation(self, ep_id, observation, agent):
    #    obs_to_save
    ##    return
    def add_episode_stats(self, ep_id, map_name, path_len, shortest_path_length, num_steps, success, h, w, init_dist):
        ep1 = {"episode_id": ep_id,
               "map_name": map_name,
               "path_len": path_len,
               "shortest_path_length": shortest_path_length,
               "num_steps": num_steps,
               "success": success, 
               "h": h, 
               "w": w,
               "init_dist_to_goal": init_dist}
        self.episodes_stats.append(ep1)
        return
    def save_csv(self):
        with open(os.path.join(self.logdir, self.csvfname), 'w') as csvfile:
            fieldnames = sorted(self.episodes_stats[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.episodes_stats)
        return
    def save(self):
        full_data = {}
        full_data["episodes_detalied"] = self.episodes_detalied
        full_data["short_stats"] = self.episodes_stats
        #full_data["sim_params"] = self.sim_params
        #full_data["init_args"] = self.init_args
        det_name = os.path.join(self.logdir, self.detailed_fname)
        try:
            with open(det_name, 'wb') as dp:
                pickle.dump(full_data, dp, protocol=2)
        except:
            print("Cannot write to ", det_name)
        return
    def load(self):
        det_name = os.path.join(self.logdir, self.detailed_fname)
        with open(det_name, 'rb') as dp:
            full_data = pickle.load(dp)
        #except:
        #    print("Cannot load from to ", det_name)
        self.episodes_detalied = full_data["episodes_detalied"]
        self.episodes_stats = full_data["short_stats"]

        return