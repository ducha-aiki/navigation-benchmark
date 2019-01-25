import numpy as np
import json
import os
import torch
from PIL import Image
import time

def readRLE(data, w, h):
    out_img = []
    numel = int(len(data) / 2)
    for i in range(numel):
        val = int(data[2*i])
        num = int(data[2*i+1])
        out_img.append(val*np.ones((num, 1), dtype=np.bool))
    out_img = np.concatenate(out_img, axis=0)
    out_img = out_img.reshape((h, w))
    return out_img


def readNavgrid_10File(fname):
    with open(fname, 'r') as f:
        nav_grid = json.load(f)
    if 'tileAttributes' not in nav_grid:
        nav_grid = nav_grid['grids'][0]
    occ_grid = readRLE(nav_grid['tileAttributes']['occupancy']['data'], nav_grid['width'], nav_grid['height'])
    return occ_grid


def navgridFname(suncgdir, hash1):
    return os.path.join('', *[suncgdir,hash1, hash1+'.furnished.grid.json'])

def generate_2dgrid(h,w, centered = False):
    if centered:
        x = torch.linspace(-w/2+1, w/2, w)
        y = torch.linspace(-h/2+1, h/2, h)
    else:
        x = torch.linspace(0, w-1, w)
        y = torch.linspace(0, h-1, h)
    grid2d = torch.stack([y.repeat(w,1).t().contiguous().view(-1), x.repeat(h)],1)
    return grid2d.view(1, h, w,2).permute(0,3,1, 2)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    

def ResizePIL(np_img, size = 128):
    im1 = Image.fromarray(np_img)
    im1.thumbnail((size,size))
    return np.array(im1)

def prepareGray(rgb, size = 128):
    return torch.from_numpy(((ResizePIL(rgb.mean(axis=2), size = size)-90.0)/50.).astype(np.float32)).view(1,1,size,size)
def prepareRGB(rgb, size = 128):
    return torch.from_numpy(((ResizePIL(rgb, size = size)-90.0)/50.).astype(np.float32)).unsqueeze(0).permute(0,3,1,2)
def prepareDepth(depth, size = 128):
    return torch.from_numpy(((ResizePIL(depth, size = size)-1.8)/ 1.2).astype(np.float32)).view(1,1,size,size)
def prepareDist(dist):
    return torch.from_numpy(dist ).float().view(1,1)
def prepareDir(direction):
    return torch.from_numpy(direction).float().view(1,2)

def findMapSize(h, w):
    map_size_in_meters = int(0.1 * 3 * max(h,w))
    if map_size_in_meters % 10 != 0:
        map_size_in_meters = map_size_in_meters + (10 - (map_size_in_meters % 10))
    return map_size_in_meters
    
def gettimestr():
    return time.strftime("%Y-%m-%d--%H_%M_%S", time.gmtime())