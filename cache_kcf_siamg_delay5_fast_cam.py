from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from ast import arg

import os
import argparse
from tkinter.messagebox import RETRY
from tkinter.tix import Tree
from turtle import update

import cv2
import torch
import numpy as np
from glob import glob

import sys 
sys.path.append(os.getcwd())  
from time import time

from siamg.core.config import cfg
from siamg.models.model_builder import ModelBuilder
from siamg.tracker.tracker_builder import build_tracker
from siamg.utils.model_load import load_pretrain
from kcf import Tracker


torch.set_num_threads(1)  


def parse():
    parser = argparse.ArgumentParser(description='tracking demo') 
    parser.add_argument('--config', default='./models/config/config.yaml',type=str, help='config file')
    parser.add_argument('--snapshot', default='./models/snapshot/checkpoint_e26.pth', type=str, help='model name')
    parser.add_argument('--video_name', default='./tv_tuanliu.mkv', type=str, help='videos or image files')
    args = parser.parse_args()
    return args
    

def set_param(args):
    cap = cv2.VideoCapture()
    cap.open(0, cv2.CAP_DSHOW)

    
    ok, frame = cap.read()
    frame_size = (frame.shape[1], frame.shape[0]) # (w, h)
    init_rect = cv2.selectROI("video_test", frame, False, False)

    return ok, cap, init_rect, frame

def main(args):
    duration = 0.01

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker_siamg = build_tracker(model)
    tracker_kcf = Tracker()


    ok, cap, init_rect, frame = set_param(args)    

    if not ok:
        print("error reading video")
        exit(-1)
  
    tracker_siamg.init(frame, init_rect)
    # 初始kcf的标志
    init_flag = False 

    # siamg需要修正的标志
    need_fix = False

    # siamg是否已经保存了cache
    already_save = False
    cnt_save = 0

    cache_info = ()
    
    # 上一轮 siamg 的得分
    last = 1
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # update siamg
        t0 = time()
        
        if need_fix == True and last > 0.99:
            outputs_siamg = tracker_siamg.track_cache(frame, cache_info)
            need_fix = False
            already_save = False
            # cnt_save = 5
            
        else:
            outputs_siamg = tracker_siamg.track(frame)
            

        bbox_siamg = list(map(int, outputs_siamg['bbox']))
        # update kcf
        
        if outputs_siamg['best_score'] < 0.99:
            if init_flag == False:
                tracker_kcf.init(frame, bbox_siamg)
                init_flag = True
                outputs_kcf = tracker_kcf.update(frame)

            elif init_flag == True:
                outputs_kcf = tracker_kcf.update(frame)

            if already_save == False:
                cache_info = tracker_siamg.save_cache(frame, bbox_siamg)
                # cnt_save = -1
                already_save = True 
        else:
            init_flag = False
            # if already_save == False:
            #     cache_info = tracker_siamg.save_cache(frame, bbox_siamg)
            #     # cnt_save = -1
            #     already_save = True

        bbox_siamg = list(map(int, outputs_siamg['bbox']))
        if  outputs_siamg['best_score'] < 0.98:
            bbox_kcf = list(map(int, outputs_kcf))
            tracker_siamg.init(frame, bbox_kcf)
            need_fix = True
        last = outputs_siamg['best_score']
        t1 = time()

        duration = 0.8 * duration + 0.2 * (t1 - t0)
        cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(frame, (bbox_siamg[0], bbox_siamg[1]),
                    (bbox_siamg[0]+bbox_siamg[2], bbox_siamg[1]+bbox_siamg[3]),
                    (0, 255, 0), 3)
        font=cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame,'{:.4f}'.format(outputs_siamg['best_score']), (bbox_siamg[0]+bbox_siamg[2], bbox_siamg[1]), font, 0.5, (0, 255, 255), 1)

        cv2.imshow('tracking', frame)
       
        c = cv2.waitKey(1) & 0xFF
        if c==27 or c==ord('q'):
            break
    

if __name__ == '__main__':
    args = parse()
    main(args)
