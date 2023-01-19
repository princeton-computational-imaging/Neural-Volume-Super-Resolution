import cv2
import os
import imageio
import numpy as np

IM_PATHS = ['/tigress/yb6751/projects/NeuralMFSR/results/ours/lego_DS2_PlRes800_32/blind_SR/42_PSNR0_49.png','/tigress/yb6751/projects/NeuralMFSR/results/baselines/lego/42_EDSR_Pretrained_upscaleX4.png']
LR_IM_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/ours/lego_DS8_PlRes200_32/fine/42_PSNR31_86.png'

lr_im = imageio.imread(LR_IM_PATH)/255
for path in IM_PATHS:
    im = imageio.imread(path)/255
    ds_im = cv2.resize(im, dsize=(lr_im.shape[0],lr_im.shape[1]), interpolation=cv2.INTER_AREA)
    rmse = np.sqrt(np.mean((ds_im-lr_im)**2))
    print(path.split('/')[-1],255*rmse)
