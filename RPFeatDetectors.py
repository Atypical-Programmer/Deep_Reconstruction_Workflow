import os
from PIL import Image
import numpy as np
import torch
import cv2
import argparse
from utils import *
from tqdm import tqdm

from tools import common
from tools.dataloader import norm_RGB
from RPFeatNets.patchnet import *
from RPFeat import *


class RPFeatDetector(object):
    def __init__(self, config={}):
        print("creating RPFeat detector...")

    def __call__(self, image_name):
        default_config = {
            "num": 8000,
            "top_k": 2000,
            "scale_f": 2**0.25,
            "min_size": 100,
            "max_size": 2000,
            "min_scale": 0,
            "max_scale": 1,
            "reliability_thr": 0,
            "repeatability_thr": 0.95,
            #"path": "./weights/rpfeat2_2_N16_epoch50.pt",
            "path": "./weights/60.pt",
            "cuda": True
        }

        net = load_network(default_config["path"])
        img = Image.fromarray(
            cv2.cvtColor(
                cv2.imread(image_name),
                cv2.COLOR_BGR2RGB
                )
            )
        W, H = img.size
        img = norm_RGB(img)[None]
        net = net.cuda()
        img = img.cuda()
        xys, desc, scores = get_rpfeat_feature(net, img,
                                               NonMaxSuppression(
                                                   rel_thr=default_config["reliability_thr"],
                                                   rep_thr=default_config["repeatability_thr"]
                                                   ),
                                               scale_f=default_config["scale_f"],
                                               min_scale=default_config["min_scale"],
                                               max_scale=default_config["max_scale"],
                                               min_size=default_config["min_size"],
                                               max_size=default_config["max_size"],
                                               top_k=default_config["top_k"],
                                               verbose=True)

        index = scores.argsort()[-default_config["num"]:].cpu().detach().numpy()
        ret_dict = {
            "image_size": [H, W],
            "keypoints": xys.cpu().detach().numpy()[index,0:2],
            "scores": scores.cpu().detach().numpy().transpose()[index],
            "descriptors": desc.cpu().detach().numpy()[index,:]
        }
        return ret_dict

def get_rpfeat_from_scenes_return(image_path):
    image_names = []
    for name in os.listdir(image_path):
        if 'jpg' in name or 'png' in name:
            image_names.append(name)
    detector = RPFeatDetector()
    points = {}
    for name in tqdm(sorted(image_names)):
        image_name = os.path.join(image_path, name)
        ret_dict = detector(image_name)
        points[name] = ret_dict
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RPFeat detector')
    args = parser.parse_args()
