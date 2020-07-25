import os
from easydict import EasyDict as edict
import time
import torch
import os
from PIL import Image

# init
__C = edict()

cfg = __C
__C.DATA = edict()
__C.TRAIN = edict()
__C.VAL = edict()
__C.VIS = edict()

input_direc = os.getcwd()

#------------------------------DATA------------------------


__C.DATA.STD_SIZE = (512,680)
__C.DATA.DATA_PATH = input_direc + "/data/trees"                 

__C.DATA.MEAN_STD = ([0.444637000561], [0.226200059056]) # part B
__C.DATA.DEN_ENLARGE = 1.

#------------------------------TRAIN------------------------
__C.TRAIN.INPUT_SIZE = (512,680)
__C.TRAIN.SEED = 640
__C.TRAIN.RESUME = ''#model path
__C.TRAIN.BATCH_SIZE = 6 #imgs
__C.TRAIN.BCE_WIEGHT = 1e-4

__C.TRAIN.SEG_LR = 1e-2
__C.TRAIN.SEG_WIEGHT = 1e-4

__C.TRAIN.NUM_BOX = 20 #boxes

__C.TRAIN.GPU_ID = [1]

# base lr
__C.TRAIN.LR = 1e-4
__C.TRAIN.LR_DECAY = 1
__C.TRAIN.NUM_EPOCH_LR_DECAY = 1 # epoches

__C.TRAIN.MAX_EPOCH = 30

# output 
__C.TRAIN.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.TRAIN.EXP_NAME = 'PCC_Net' + now

__C.TRAIN.EXP_PATH = './exp'

#------------------------------VAL------------------------
__C.VAL.BATCH_SIZE = 1 # imgs
__C.VAL.FREQ = 1

#------------------------------VIS------------------------
__C.VIS.VISIBLE_NUM_IMGS = 20

#------------------------------MISC------------------------


#================================================================================
#================================================================================
#================================================================================  