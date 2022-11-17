import imageio
import os
from re import search
import cv2
from collections import OrderedDict
from glob import glob
import numpy as np
from tqdm import trange

SCENE = 'ship'
LEFT_SOURCE = 'ours'
RIGHT_SOURCE = 'edsr'
FPS = 20
ZOOMIN_FACTOR = 1.5
ZOOM_TRANSITION = 1

HR_DS_FACTOR = 2
SR_FACTOR = 4
OUR_RESULTS_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/ours'
BSELINES_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/baselines'
OUTPUT_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/comparisons'
SOURCES = {
    'GT':{'p_im':'(?<=\/r_)(\d)+(?=\.png$)','p_scene':lambda scene:scene+'/test/*','path':'/scratch/gpfs/yb6751/datasets/Synthetic',},
    'LR':{'title':'Low-res.','p_im':'(?<=\/)(\d)+(?=\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*LR/*'%(HR_DS_FACTOR*SR_FACTOR),'path':OUR_RESULTS_PATH,},
    'ours':{'title':'Ours','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR),'path':OUR_RESULTS_PATH},
    'naive':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*fine/*'%(HR_DS_FACTOR),'path':OUR_RESULTS_PATH},
    'edsr':{'title':'EDSR','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'edsr_pre':{'title':'EDSR (Data)','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'srgan':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'srgan_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'rstt':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'rstt_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'swin':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'swin_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'preSR':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/blind_fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/planes_on_SR_Res800_32_0'},
    'view':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/Synt_Res16Sc800_32_400_NerfUseviewdirsTrue_1'},
    'no_view':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/Synt_Res16Sc800_32_400_NerfUseviewdirsFalse_0'},
    'PlRes100':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes100*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    'PlRes400':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes400*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    'PlRes1600':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes1600*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    'bicubic':{},
}

im_paths = {}
methods_2_load = [LEFT_SOURCE,RIGHT_SOURCE]
# for scene in SCENES:
for i_method,method in enumerate(methods_2_load):
    if 'path' not in SOURCES[method]:    continue
    im_paths[method] = [f for f in glob(os.path.join(SOURCES[method]['path'],SOURCES[method]['p_scene'](SCENE))) if search(SOURCES[method]['p_im'],f) is not None]
    im_paths[method] = OrderedDict(sorted([(int(search(SOURCES[method]['p_im'],f).group(0)),f) for f in im_paths[method]],key=lambda x:x[0]))
    if i_method>0: #Not the first method for this scene, assert the number of images is bigger than 0 and the same
        assert len(im_paths[method])>0
        assert len(im_paths[method])==len(im_paths[methods_2_load[i_method-1]])

frames = []
def read_image(path):
    return imageio.imread(path)


SEPARATOR_WIDTH = 2
FONT_SCALE = 2
THICKNESS = 1
TEXT_Y = 30

frame_shape = np.array(read_image(im_paths[LEFT_SOURCE][0]).shape[:2])
num_frames = len(im_paths[LEFT_SOURCE])
zoomin = np.ones(num_frames)
if ZOOMIN_FACTOR is not None:
    zoomin = [np.ones(num_frames//2)]
    zoomin.append(np.linspace(1,ZOOMIN_FACTOR,ZOOM_TRANSITION*FPS))
    zoomin.append(ZOOMIN_FACTOR*np.ones(num_frames-2*ZOOM_TRANSITION*FPS))
    zoomin.append(np.linspace(ZOOMIN_FACTOR,1,ZOOM_TRANSITION*FPS))
    zoomin.append(np.ones(num_frames//2))
    zoomin = np.concatenate(zoomin)
for f_num in trange(len(zoomin)):
    left = read_image(im_paths[LEFT_SOURCE][f_num%num_frames])
    if zoomin[f_num]>1:
        left = cv2.resize(left, dsize=(0,0),fx=zoomin[f_num],fy=zoomin[f_num], interpolation=cv2.INTER_AREA)
        leftovers = np.array(left.shape)[:2]-frame_shape
        left = left[leftovers[0]//2:-(leftovers[0]-leftovers[0]//2),leftovers[1]//2:-(leftovers[1]-leftovers[1]//2),:]
    left = cv2.putText(
        left,
        SOURCES[LEFT_SOURCE]['title'],
        (0,TEXT_Y),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=FONT_SCALE,
        color=[255,255,255],
        thickness=THICKNESS,
    )
    separator = 255*np.ones([frame_shape[0],SEPARATOR_WIDTH,3])
    right = read_image(im_paths[RIGHT_SOURCE][f_num%num_frames])
    if zoomin[f_num]>1:
        right = cv2.resize(right, dsize=(0,0),fx=zoomin[f_num],fy=zoomin[f_num], interpolation=cv2.INTER_AREA)
        leftovers = np.array(right.shape)[:2]-frame_shape
        right = right[leftovers[0]//2:-(leftovers[0]-leftovers[0]//2),leftovers[1]//2:-(leftovers[1]-leftovers[1]//2),:]
    right = cv2.putText(
        right,
        SOURCES[RIGHT_SOURCE]['title'],
        (frame_shape[1]-cv2.getTextSize(SOURCES[RIGHT_SOURCE]['title'], cv2.FONT_HERSHEY_PLAIN, FONT_SCALE,THICKNESS)[0][0],TEXT_Y),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=FONT_SCALE,
        color=[255,255,255],
        thickness=THICKNESS,
    )
    frames.append(np.concatenate([left[:,:frame_shape[1]//2,:],separator,right[:,-(frame_shape[1]//2+SEPARATOR_WIDTH):,:]],1).astype(np.uint8))




vid_path = os.path.join(OUTPUT_PATH,'%s_%s_%s_FPS%d%s.mp4'%(SCENE,LEFT_SOURCE,RIGHT_SOURCE,FPS,('_Zoom%.1f'%(ZOOMIN_FACTOR)).replace('.','_') if ZOOMIN_FACTOR else ''))
# imageio.mimwrite(vid_path, [np.array(255*torch.clamp(im,0,1).cpu()).astype(np.uint8) for im in images], fps = FPS, macro_block_size = 8)  # pip install imageio-ffmpeg
imageio.mimwrite(vid_path, frames, fps = FPS, macro_block_size = 8)  # pip install imageio-ffmpeg
