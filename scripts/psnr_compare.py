import os
import numpy as np
from re import search
from glob import glob

PATHS = {'preSR':'/tigress/yb6751/projects/NeuralMFSR/results/planes_on_SR_Res800_32_0/%s_DS2_PlRes800_32/blind_fine',
        'ours':'/tigress/yb6751/projects/NeuralMFSR/results/ours/%s_DS2_PlRes800_32/blind_SR',
        'ref':'/tigress/yb6751/projects/NeuralMFSR/results/ours/%s_DS2_PlRes800_32/blind_fine'}

SCENE = 'mic'

im_names = dict([(k,sorted([f for f in glob(v%(SCENE)+'/*.png')],key=lambda x:int(search('(?<=/)(\d)+(?=_)',x).group(0)))) for k,v in PATHS.items()])
# images_preSR = sorted([f for f in glob(PATH_preSR%(SCENE)+'/*.png')],key=lambda x:int(search('(?<=/)(\d)+(?=_)',x).group(0)))
im_PSNRs = dict([(k,[float(search('(?<=_PSNR)(\d)+_(\d)+(?=\.png)',f).group(0).replace('_','.')) for f in v]) for k,v in im_names.items()])
adv = [(i,im_PSNRs['ours'][i]+im_PSNRs['ref'][i]-im_PSNRs['preSR'][i]) for i in range(len(im_PSNRs['ours']))]
sorted_ims = sorted(adv,key=lambda x:-x[1])
for i in range(10):
    print(sorted_ims[i])
# for 

