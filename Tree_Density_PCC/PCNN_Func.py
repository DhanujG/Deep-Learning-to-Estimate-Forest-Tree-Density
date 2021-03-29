# A small set of helper functions found in Github of  Paper - J. Gao, Q. Wang and X. Li, "PCC Net: Perspective Crowd Counting via Spatial Convolutional Network," doi: 10.1109/TCSVT.2019.2919139.
# https://github.com/gjy3035/PCC-Net/tree/ori_pt1_py3/misc
import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import timer
import cv2
import numpy as np
import os
# ===============================img tranforms============================

class Compose_img(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, seg):
        for t in self.transforms:
            img, mask,seg = t(img, mask,seg)
        return img, mask, seg

class RandomHorizontallyFlip_img(object):
    def __call__(self, img, mask, seg):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), seg.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, seg


class RandomCrop_img(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, seg):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            seg = ImageOps.expand(seg, border=self.padding, fill=0)

        assert img.size == mask.size
        assert img.size == seg.size
        w, h = img.size
        th, tw  = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST), seg.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), seg.crop((x1, y1, x1 + tw, y1 + th))


# ===============================label tranforms============================

class DeNormalize_label(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor_label(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class Timer(object):

    def __init__(self):
        self.tot_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.tot_time  += self.diff
        self.calls += 1
        self.average_time = self.tot_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def save_results(input_img, gt_data,density_map,output_dir, fname='results.png'):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img.shape[1],input_img.shape[0]))
        gt_data = cv2.resize(gt_data, (input_img.shape[1],input_img.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    cv2.imwrite(os.path.join(output_dir,fname),result_img)
    

def save_density_map(density_map,output_dir, fname='results.png'):    
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    cv2.imwrite(os.path.join(output_dir,fname),density_map)
    
def display_results(input_img, gt_data,density_map):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
         input_img = cv2.resize(input_img, (density_map.shape[1],density_map.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    result_img  = result_img.astype(np.uint8, copy=False)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)