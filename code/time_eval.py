import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils
import time
from skimage import io
import dlib

dlib.DLIB_USE_CUDA = True


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
                        default='../models/hopenet_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
                        default='../models/dlib_model/mmod_human_face_detector.dat', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video', default='../models/test.mp4')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file',
                        default='_lr')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int, default=300)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = parser.parse_args()
    return args


def eval_diff_bottlenecks():
    mdls_lst = {"resnet101": torchvision.models.resnet101, "resnet50": torchvision.models.resnet50,
                "mobilenet": torchvision.models.mobilenet_v2, "vgg19": torchvision.models.vgg19,
                "resnet34": torchvision.models.resnet34, "vgg11": torchvision.models.vgg11,
                "squeezenet1_1": torchvision.models.squeezenet1_1, "alexnet": torchvision.models.alexnet}
    x = torch.randn(1, 3, 224, 224, device='cuda:0', dtype=torch.float32)
    for mdl_name in mdls_lst:
        model = mdls_lst[mdl_name]()
        model.cuda(gpu)

        ts = []
        for i in range(1000):
            et = time.time()

            _ = model(x)
            ts.append(time.time() - et)

        a = str("average for " + mdl_name).ljust(30, ' ')
        b = str(round(np.average(ts), 7)).ljust(5, ' ')
        print a + b


def eval_diff_img_size():
    szs = {"720x1280":(720, 1280, 1), "360x640":(360, 640, 1), "180x320":(180, 320, 1), "90x160":(90, 160, 1)}
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

    for sc in [0, 1, 2]:
        for s_name in szs:
            sz = szs[s_name]
            x = np.random.randint(low=0, high=255, size=sz, dtype=np.uint8)

            ts = []
            for i in range(100):
                et = time.time()
                _ = cnn_face_detector(x, sc)
                ts.append(time.time() - et)

            a = str("average for " + s_name + " using scale " + str(sc)).ljust(40, ' ')
            b = str(round(np.average(ts), 7)).ljust(5, ' ')
            print a + b


et = time.time()
if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1

    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_path = args.video_path

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gpu = args.gpu_id
    eval_diff_bottlenecks()
    eval_diff_img_size()


print 'Took : ', time.time() - et
