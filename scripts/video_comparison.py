import imageio
import os
from re import search
import cv2
from collections import OrderedDict,defaultdict
from glob import glob
import numpy as np
from tqdm import trange

SYNT_SCENES = ['chair','drums','ficus','hotdog','lego','materials','mic','ship','bugatti','materials','motorbike']
LLFF_SCENES = ['orchids','fern','flower','fortress','horns','leaves','room','trex']
# SCENE = 'ship'
SCRIPT = [ # CVPR submission supplementary (I think...)
    (0,('scene','lego')),
    (0,('left_view','LR')),
    (0.25,('zoom',1.8)),
    (0.5,('left_view','ours')),
    (0.25,('zoom',1.5)),
    (0.0,('right_view','ours')),
    (0.5,('effect','half50')),
    (0.0,('right_view','edsr')),
    (0.5,('right_view','srgan')),
    # (0.5,('zoom',1.2)),
    (0.5,('right_view','rstt')),
    (0.5,('scene','donut')),
    (1,('scene','dragon')),
    (0.0,('zoom',1.2)),
    (0.0,('right_view','swin')),
    (0.5,('zoom',1)),
    # (1,('scene','mic')),
    # (0.5,('effect','half0')),
#     ]
# SCRIPT = [
    (1.5,('scene','bugatti')),
    # (0.0,('effect','half50')),
    (0,('left_view','view')),
    (0.0,('right_view','no_view')),
    (1,('scene','materials')),
    (1,('scene','motorbike')),
    # (1,('scene','cola')),
    (0.75,('effect','half0')),
]
SCRIPT = [
    # (0,('left_view','ours')),
    # (0,('right_view','edsr')),
    # (0,('scene','dragon')),
    # (0.1,('effect','half50')),
    # (0.1,('zoom',1.5)),
    # (0.4,('right_view','swin')),
    # (0.5,('right_view','srgan')),
    # (0,('scene','lego')),
    # (0.4,('right_view','rstt')),
    # (0.1,('zoom',1.)),
    (0,('left_view','ours')),
    (0,('right_view','edsr')),
    (0,('scene','dragon')),
    (0.1,('effect','half50')),
    (0.1,('zoom',1.5)),
    (0.8,('right_view','swin')),
    (0.5,('scene','lego')),
    (0.1,('zoom',1.)),
    (0.5,('right_view','srgan')),
    (0.5,('scene','ship')),
    (0.5,('right_view','rstt')),
    (0.7,('effect','half0')),
]
SCRIPT = [
    (0,('left_view','LR')),
    (0,('right_view','ours')),
    (0,('scene','lego')),
    # (0,('scene','fern')),
    (0.1,('effect','half50')),
    (0.5,('zoom',1.5)),
    # (0.5,('right_view','preSR')),
    (1.4,('scene','ship')),
    (0.2,('effect','half100')),
    (0,('left_view','edsr')),
    (0.8,('scene','mic')),
    (0.2,('zoom',1)),
    (0.3,('effect','half50')),
    # (1,('scene','mic')),
    # (1,('scene','ship')),
    # (1,('scene','orchids')),
    # (0.4,('right_view','rstt')),
    # (0.1,('zoom',1.)),
    # (0.6,('effect','half100')),
]
FPS = 20
SCRIPT = [ #Real scenes for ICCV supplementary
    (0,('left_view','LR')),
    (0,('right_view','ours')),
    (0,('scene','orchids')),
    (0.1,('effect','half50')),
    (0.8,('effect','half100')),
    (0.9,('left_view','mip_nerf')),
    (0.1,('effect','half50')),
    (1.1,('scene','fern')),
    (0.,('left_view','LR')),
    (0.,('effect','half0')),
    (0.2,('effect','half50')),
    (0.2,('left_view','mip_nerf')),
    (0.5,('effect','half100')),
    (0.1,('scene','flower')),
    (0.,('left_view','LR')),
    (0.,('effect','half0')),
    (0.2,('effect','half50')),
    (0.4,('left_view','rstt')),
    (0.7,('effect','half100')),
    (0.7,('scene','fortress')),
    (0.,('left_view','LR')),
    (0.,('effect','half0')),
    (0.2,('effect','half50')),
    (0.5,('effect','half100')),
    (0.3,('scene','room')),
    (0.,('left_view','LR')),
    (0.,('effect','half0')),
    (0.2,('effect','half50')),
    (0.5,('effect','half100')),
    (0.3,('scene','horns')),
    (0.,('left_view','LR')),
    (0.,('effect','half0')),
    (0.2,('effect','half50')),
    (0.4,('left_view','mip_nerf')),
    (0.5,('effect','half100')),
]
FPS = 15
SCRIPT = [ #Synthetic scenes for ICCV supplementary
    (0,('left_view','LR')),
    (0,('right_view','ours')),
    (0,('scene','lego')),
    (0.1,('effect','half50')),
    (0.6,('effect','half100')),
    (0.4,('left_view','rstt')),
    (0.1,('effect','half50')),
    (0.7,('left_view','swin')),
    (0.8,('left_view','edsr')),
    (0.3,('scene','ship')),
    (0.,('left_view','LR')),
    (0.,('effect','half0')),
    (0.2,('effect','half50')),
    (0.3,('zoom',1.5)),
    (0.2,('left_view','preSRscratch')),
    (0.8,('effect','half100')),
    (0.2,('zoom',1.)),
    (0.3,('scene','mic')),
    (0.,('left_view','LR')),
    (0.,('effect','half0')),
    (0.2,('effect','half50')),
    (0.4,('left_view','mip_nerf')),
    (0.7,('effect','half100')),
    (0.7,('scene','chair')),
    (0.,('left_view','LR')),
    (0.,('effect','half0')),
    (0.2,('effect','half50')),
    (0.3,('left_view','srgan')),
    (0.4,('effect','half100')),
]
SCRIPT = [
    (0.1,('scene','bugatti')),
    (0.,('effect','half50')),
    (0,('left_view','view')),
    (0.0,('right_view','no_view')),
    (0.3,('scene','materials')),
    (0.3,('scene','motorbike')),
    # (0.75,('effect','half0')),
]

TRANSITION_TIMES = {'zoom':1,'fade':0.5,'half':1}

HR_DS_FACTOR = dict([(sc,2) for sc in SYNT_SCENES])
HR_DS_FACTOR.update(dict([(sc,8) for sc in LLFF_SCENES]))
SR_FACTOR = 4 # On top of HR_DS_FACTOR
HIGH_RES_OUTPUT = True
EXCLUDE_TITLE = False #False #
FRAME_NUM = False
# GIF_LIKE_SAVING = [36,67]
GIF_LIKE_SAVING = None
SEPARATOR_WIDTH = 2
FONT_SCALE = 2#3
THICKNESS = 2#3
TEXT_Y = 30
X_OFFSET = 30

# OUR_RESULTS_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/ours'
# OUR2_RESULTS_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/E2E_Synt_Res29Sc200_27Sc800_32_LR100_400_fromDetachedLR_imConsistLossFreq10nonSpatial_WOplanes_HrLr_micShip_0'
RESULTS_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/'
OUTPUT_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/comparisons'
EXPERIMENT_ID = 'ours_ICCV_perSceneRefine'
# EXPERIMENT_ID = 'ours_gaussian'
OUR_RESULTS_PATH = RESULTS_PATH+EXPERIMENT_ID
BASELINES_PATH = RESULTS_PATH+'baselines'
POST_SR_PATH = os.path.join(BASELINES_PATH,'postSR_MipNeRF')

postSR_pattern = lambda scene:scene+'_DS%d/*'%(HR_DS_FACTOR[scene]*SR_FACTOR)

SOURCES = {
    # 'GT_synt':{'p_im':'(?<=\/r_)(\d)+(?=\.png$)','p_scene':lambda scene:(scene[:scene.find('##')] if '##' in scene else scene)+'/test/*','path':GT_IMS_PATH,},
    # 'GT_real':{'p_im':'(?<=\/image)(\d)+(?=\.png$)','p_scene':lambda scene:scene+'/images_8/*','path':'/scratch/gpfs/yb6751/datasets/LLFF',},
    'LR':{'title':'Low-res.','p_im':'(?<=\/)(\d)+(?=\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*LR/*'%(HR_DS_FACTOR[scene]*SR_FACTOR),'path':OUR_RESULTS_PATH,},
    'ours':{'title':'Ours','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR[scene]),'path':OUR_RESULTS_PATH},
    'naive':{'title':'Naive: Planes','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*fine/*'%(HR_DS_FACTOR[scene]),'path':OUR_RESULTS_PATH},
    # 'nerf':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)\.png$)','p_scene':lambda scene:'lr_nerf_'+scene+'_0/'+scene+'_DS%d/blind_fine/*'%(HR_DS_FACTOR[scene]),'path':os.path.join(BASELINES_PATH,'nerf'),'own_lr':True},
    'mip_nerf':{'title':'Mip-NeRF','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d/blind_fine/*'%(HR_DS_FACTOR[scene]),'path':os.path.join(RESULTS_PATH,'baselines/MipNeRF')}, # New pre-SR baseline configuration, using Mip-NeRF representation model trained on images super-resolved using a pre-trained SwinIR model.
    # 'rstt_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    'rstt':{'title':'post. 2D (video) SR','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'swin_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'srgan_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'edsr_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    'swin':{'title':'post. 2D SR (SwinIR)','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    'srgan':{'title':'post. 2D SR (SRGAN)','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    'edsr':{'title':'post. 2D SR (EDSR)','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'preSR':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/blind_fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/planes_on_SR_Res800_32_0'}, # Old pre-SR baseline configuration, using planes representation model trained on images super-resolved using a trained-from-scratch EDSR model
    # 'preSR':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2/blind_fine/*','path':os.path.join(RESULTS_PATH,'baselines/MipNeRF_preSR')}, # New pre-SR baseline configuration, using Mip-NeRF representation model trained on images super-resolved using a pre-trained SwinIR model.
    'preSRscratch':{'title':'pre. 2D SR','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d/blind_fine/*'%(HR_DS_FACTOR[scene]),'path':os.path.join(RESULTS_PATH,'baselines/MipNeRF_preSRscratch')}, # New pre-SR baseline configuration, using Mip-NeRF representation model trained on images super-resolved using a SwinIR model trained from scratch on our scene training set.
    # Old:
    # 'GT':{'p_im':'(?<=\/r_)(\d)+(?=\.png$)','p_scene':lambda scene:scene+'/test/*','path':'/scratch/gpfs/yb6751/datasets/Synthetic',},
    # 'LR':{'title':'Low-res.','p_im':'(?<=\/)(\d)+(?=\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*LR/*'%(HR_DS_FACTOR[scene]*SR_FACTOR),'path':OUR_RESULTS_PATH,},
    # 'ours':{'title':'Ours','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR[scene]),'path':OUR_RESULTS_PATH},
    # 'ours2':{'title':'Ours2','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR[scene]),'path':OUR2_RESULTS_PATH},
    # 'naive':{'title':'Naive','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*fine/*'%(HR_DS_FACTOR[scene]),'path':OUR_RESULTS_PATH},
    # 'edsr':{'title':'EDSR','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    # 'edsr_pre':{'title':'EDSR (Data)','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    # 'srgan':{'title':'SRGAN','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    # 'srgan_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    # 'rstt':{'title':'Video SR','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    # 'rstt_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    # 'swin':{'title':'SwinIR','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    # 'swin_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    # 'preSR':{'title':'Pre-SR','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/blind_fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/planes_on_SR_Res800_32_0'},
    'view':{'title':'4 planes','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/Synt_Res16Sc800_32_400_NerfUseviewdirsTrue_1'},
    'no_view':{'title':'3 planes','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/Synt_Res16Sc800_32_400_NerfUseviewdirsFalse_0'},
    # 'PlRes100':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes100*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    # 'PlRes400':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes400*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    # 'PlRes1600':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes1600*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    # 'bicubic':{},
}

cur_scene,cur_left,cur_right = None,None,None
method_scene_pairs = []
for sc in SCRIPT:
    if cur_scene is not None and cur_left is not None and sc[0]>0:
        method_scene_pairs.append((cur_left,cur_scene))
    if cur_scene is not None and cur_right is not None and sc[0]>0:
        method_scene_pairs.append((cur_right,cur_scene))
    if sc[1][0]=='scene':
        cur_scene = sc[1][1]
    if 'left_view'==sc[1][0]:
        cur_left = sc[1][1]
    if 'right_view'==sc[1][0]:
        cur_right = sc[1][1]
if cur_scene is not None and cur_left is not None and sc[1][0] in ['scene','left_view']:
    method_scene_pairs.append((cur_left,cur_scene))
if cur_scene is not None and cur_right is not None and sc[1][0] in ['scene','right_view']:
    method_scene_pairs.append((cur_right,cur_scene))


included_baselines = list(set([v[1][1] for v in SCRIPT if v[1][0] in ['right_view','left_view']]))
methods_2_load = included_baselines
scenes = list(set([v[1][1] for v in SCRIPT if v[1][0]=='scene']))
assert all([sc in SYNT_SCENES for sc in scenes]) or all([sc in LLFF_SCENES for sc in scenes]),'Cannot mix synthetic and real world scenes.'
synt_scenes = all([sc in SYNT_SCENES for sc in scenes])
im_paths = defaultdict(dict)
# for scene in scenes:
#     for i_method,method in enumerate(methods_2_load):
num_frames = None
f_num_dict = {}
for i_method,(method,scene) in enumerate(set(method_scene_pairs)):
    if 'path' not in SOURCES[method]:    continue
    path_filter = os.path.join(SOURCES[method]['path'],"%s"+SOURCES[method]['p_scene'](scene))
    path_filter = path_filter%('4video_') if scene in LLFF_SCENES and len(glob(path_filter%('4video_')))>0 else path_filter%('')
    im_paths[method][scene] = [f for f in glob(path_filter) if search(SOURCES[method]['p_im'],f) is not None]
    im_paths[method][scene] = OrderedDict(sorted([(int(search(SOURCES[method]['p_im'],f).group(0)),f) for f in im_paths[method][scene]],key=lambda x:x[0]))
    if num_frames is not None and scene in SYNT_SCENES:
    # if i_method>0: #Not the first method for this scene, assert the number of images is bigger than 0 and the same
        assert len(im_paths[method][scene])==num_frames
    elif scene in LLFF_SCENES:
        assert len(im_paths[method][scene])>0
        if num_frames is None:
            num_frames = len(im_paths[method][scene])
        else:
            num_frames = max(num_frames,len(im_paths[method][scene]))
    else:
        num_frames = len(im_paths[method][scene])
        assert num_frames>0
    f_num_dict[scene] = np.arange(len(im_paths[method][scene]))
num_frames = min(200,num_frames)
# f_num_dict = defaultdict(list)
# for scene in scenes:
#     num_existing_frames = len(im_paths['ours'][scene])
#     i,going_up = 0,True
#     for _ in range(num_frames):
#         f_num_dict[scene].append(i)
#         if going_up:
#             if i==(num_existing_frames-1):  going_up = False
#         else:
#             if i==0:    going_up = True
#         i += 1 if going_up else -1
# f_num_dict = dict([(sc,np.arange(len(im_paths['ours'][sc]))) for sc in scenes])
script,n_frames = [],0
zoomin,fade,right_views,left_views,scene2show,halfing = [],[],[],[],[],[]
latest_zoom,latest_disp_portion = 1,0
for sc in SCRIPT:
    if sc[1][0]=='effect':
        if 'half' in sc[1][1]:
            disp_portion = int(search('(?<=half)(\d)+$',sc[1][1]).group(0))/100
            # if n_frames+sc[0]*num_frames==0:
            if (n_frames+sc[0]*num_frames,1) in fade or n_frames==0:
                halfing.append((n_frames+sc[0]*num_frames,disp_portion))
            else:
                halfing.extend(list(zip(
                    [int(np.round(n_frames+sc[0]*num_frames-FPS*TRANSITION_TIMES['half']/2+i)) for i in range(int(np.round(FPS*TRANSITION_TIMES['half'])))],
                    np.linspace(latest_disp_portion,disp_portion,FPS*TRANSITION_TIMES['half']))))
                # halfing.extend(list(zip(
                #     [int(n_frames+sc[0]*num_frames-FPS*TRANSITION_TIMES['half']/2+i) for i in range(int(FPS*TRANSITION_TIMES['half']))],
                #     np.linspace(0,2*disp_portion,FPS*TRANSITION_TIMES['half'])[::(1 if sc[1][1]=='half' else -1)])))
            latest_disp_portion = 1*disp_portion
        else:
            raise Exception
    elif sc[1][0]=='zoom':
        zoomin.extend(list(zip(
            [int(np.round(n_frames+sc[0]*num_frames-FPS*TRANSITION_TIMES['zoom']/2+i)) for i in range(int(np.round(FPS*TRANSITION_TIMES['zoom'])))],
            np.linspace(latest_zoom,sc[1][1],FPS*TRANSITION_TIMES['zoom']))))
        latest_zoom = sc[1][1]
    elif sc[1][0]=='scene':
        if n_frames>FPS*TRANSITION_TIMES['fade']/2:
            fade.extend(list(zip(
                [int(np.round(n_frames+sc[0]*num_frames-FPS*TRANSITION_TIMES['fade']/2+i)) for i in range(int(np.round(FPS*TRANSITION_TIMES['fade'])))],
                np.concatenate([np.linspace(0,1,int(np.round(FPS*TRANSITION_TIMES['fade']/2))),np.linspace(1,0,int(np.round(FPS*TRANSITION_TIMES['fade']/2)))]))))
        scene2show.append((n_frames+sc[0]*num_frames,sc[1][1]))
    elif sc[1][0]=='right_view':
        right_views.append((n_frames+sc[0]*num_frames,sc[1][1]))
    elif sc[1][0]=='left_view':
        left_views.append((n_frames+sc[0]*num_frames,sc[1][1]))
    else:
        raise Exception
    n_frames += sc[0]*num_frames
# total_frames = int(scene2show[-1][0]+num_frames)
if sc[1][0]=='scene':
    n_frames += sc[0]*num_frames
total_frames = int(max(1,np.ceil(n_frames/num_frames))*num_frames)
frames = []
def read_image(path):
    return imageio.imread(path)

for i,cur_scene in enumerate(scenes):
    for m in methods_2_load:
        if m in im_paths and cur_scene in im_paths[m]:  break
    if i==0:
        frame_shape = np.array(read_image(im_paths[m][cur_scene][0]).shape[:2])
    else:
        assert np.all(frame_shape==np.array(read_image(im_paths[m][cur_scene][0]).shape[:2]))

if HIGH_RES_OUTPUT:
    TEXT_Y = int(1080/frame_shape[1]*TEXT_Y)
    FONT_SCALE = int(0.6*(1080 if synt_scenes else 1440)/frame_shape[1]*FONT_SCALE)
    THICKNESS = int(0.6*(1080 if synt_scenes else 1440)/frame_shape[1]*THICKNESS)

cur_fade,cur_zoom,cur_disp_portion = 0,1,0
output_width = (1080 if synt_scenes else 1440) if HIGH_RES_OUTPUT else frame_shape[1]

def im_path(method,scene,f_num):
    # if f_num%(2*num_frames)<num_frames:
    #     return im_paths[method][scene][f_num]
    # else:
    #     return im_paths[method][scene][num_frames-f_num]
    return im_paths[method][scene][f_num_dict[scene][f_num]]

for f_num in trange(total_frames):
    if len(left_views)>0 and f_num>=left_views[0][0]:
        cur_left = left_views.pop(0)[1]
    if len(scene2show)>0 and f_num>=scene2show[0][0]:
        cur_scene = scene2show.pop(0)[1]
    if len(right_views)>0 and f_num>=right_views[0][0]:
        cur_right = right_views.pop(0)[1]
    if len(fade)>0 and f_num>=fade[0][0]:
        cur_fade = fade.pop(0)[1]
    if len(zoomin)>0 and f_num>=zoomin[0][0]:
        cur_zoom = zoomin.pop(0)[1]
    if len(halfing)>0 and f_num>=halfing[0][0]:
        cur_disp_portion = halfing.pop(0)[1]
    new_frame = []
    if cur_disp_portion<1:
        left = read_image(im_path(cur_left,cur_scene,f_num%num_frames))
        if cur_left=='GT' and HR_DS_FACTOR[cur_scene]>1:
            left = cv2.resize(left, dsize=(0,0),fx=1/HR_DS_FACTOR[cur_scene],fy=1/HR_DS_FACTOR[cur_scene], interpolation=cv2.INTER_AREA)
        if cur_zoom>1:
            left = cv2.resize(left, dsize=(0,0),fx=cur_zoom,fy=cur_zoom, interpolation=cv2.INTER_AREA)
            leftovers = np.array(left.shape)[:2]-frame_shape
            left = left[leftovers[0]//2:-(leftovers[0]-leftovers[0]//2),leftovers[1]//2:-(leftovers[1]-leftovers[1]//2),:]
        if HIGH_RES_OUTPUT:
            left = cv2.resize(left, dsize=(1080,1080) if synt_scenes else (1440,1080),interpolation=cv2.INTER_AREA)
        if cur_disp_portion<0.75:
            title = '' if EXCLUDE_TITLE else SOURCES[cur_left]['title']+(', %d'%(f_num) if FRAME_NUM else '')
            left = cv2.putText(
                left,
                title,
                (X_OFFSET,TEXT_Y),
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=FONT_SCALE,
                color=[255,255,255],
                thickness=THICKNESS,
            )
        new_frame.append(left[:,:int(output_width*(1-cur_disp_portion)),:])
    if 0<cur_disp_portion<1:
        separator = 255*np.ones([1080 if HIGH_RES_OUTPUT else frame_shape[0],SEPARATOR_WIDTH,3])
        new_frame.append(separator)
    if cur_disp_portion>0:
        right = read_image(im_path(cur_right,cur_scene,f_num%num_frames))
        if cur_left=='GT' and HR_DS_FACTOR[cur_scene]>1:
            right = cv2.resize(right, dsize=(0,0),fx=1/HR_DS_FACTOR[cur_scene],fy=1/HR_DS_FACTOR[cur_scene], interpolation=cv2.INTER_AREA)
        if cur_zoom>1:
            right = cv2.resize(right, dsize=(0,0),fx=cur_zoom,fy=cur_zoom, interpolation=cv2.INTER_AREA)
            leftovers = np.array(right.shape)[:2]-frame_shape
            right = right[leftovers[0]//2:-(leftovers[0]-leftovers[0]//2),leftovers[1]//2:-(leftovers[1]-leftovers[1]//2),:]
        if HIGH_RES_OUTPUT:
            right = cv2.resize(right, dsize=(1080,1080) if synt_scenes else (1440,1080),interpolation=cv2.INTER_AREA)
        # if cur_disp_portion==1:
        if cur_disp_portion>0.25:
            title = '' if EXCLUDE_TITLE else SOURCES[cur_right]['title']+(', %d'%(f_num) if FRAME_NUM else '')
            right = cv2.putText(
                right,
                title,
                (output_width-cv2.getTextSize(title, cv2.FONT_HERSHEY_PLAIN, FONT_SCALE,THICKNESS)[0][0]-X_OFFSET,TEXT_Y),
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=FONT_SCALE,
                color=[255,255,255],
                thickness=THICKNESS,
            )
        # new_frame.extend([separator,right[:,-(1+max(SEPARATOR_WIDTH,(int(frame_shape[1]*cur_disp_portion/2-SEPARATOR_WIDTH)))):,:]])
        new_frame.append(right[:,-(1+max(SEPARATOR_WIDTH,(int(output_width*cur_disp_portion-SEPARATOR_WIDTH*(cur_disp_portion<1))))):,:])
    # if cur_fade>0:
    #     left = (1-cur_fade)*left
    #     right = (1-cur_fade)*right
    # new_frame = []
    new_frame = np.concatenate(new_frame,1)[:,:output_width,:]
    if HIGH_RES_OUTPUT and synt_scenes:
        new_frame = np.pad(new_frame,((0,0),(180,180),(0,0)))
    if cur_fade>0:
        new_frame = (1-cur_fade)*new_frame
    frames.append(
        new_frame
        # if cur_half*frame_shape[1]/2-SEPARATOR_WIDTH>=1:
        # [left[:,:int(frame_shape[1]*(1-cur_half/2)-SEPARATOR_WIDTH/2),:],
        #     separator,
        #     right[:,-(1+max(SEPARATOR_WIDTH,(int(frame_shape[1]*cur_half/2-SEPARATOR_WIDTH/2)))):,:]]
        # [left[:,:max(0,int(frame_shape[1]*cur_half/2-SEPARATOR_WIDTH/2)),:],
        #     separator,
        #     right[:,-(1+max(SEPARATOR_WIDTH,(int(frame_shape[1]*(1-cur_half/2)-SEPARATOR_WIDTH/2)))):,:]]
        .astype(np.uint8))




vid_path = os.path.join(OUTPUT_PATH,'%s_B%s_FPS%d%s.mp4'%('_'.join(sorted(scenes)),'_'.join(sorted(included_baselines)),FPS,'gif_like' if GIF_LIKE_SAVING is not None else ''))
if GIF_LIKE_SAVING is not None:
    frames = frames[GIF_LIKE_SAVING[0]:GIF_LIKE_SAVING[1]+1]+frames[GIF_LIKE_SAVING[1]-1:GIF_LIKE_SAVING[0]-1:-1]
imageio.mimwrite(vid_path, frames, fps = FPS, macro_block_size = 8)  # pip install imageio-ffmpeg
