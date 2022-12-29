import imageio
import os
from re import search
import cv2
from collections import OrderedDict,defaultdict
from glob import glob
import numpy as np
from tqdm import trange

# SCENE = 'ship'
SCRIPT = [
    (0,('scene','lego')),
    (0,('left_view','LR')),
    (0.25,('zoom',1.8)),
    (0.5,('left_view','ours')),
    (0.25,('zoom',1.5)),
    (0.0,('right_view','ours')),
    (0.5,('effect','half')),
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
    # (0.5,('effect','unhalf')),
#     ]
# SCRIPT = [
    (1.5,('scene','bugatti')),
    # (0.0,('effect','half')),
    (0,('left_view','view')),
    (0.0,('right_view','no_view')),
    (1,('scene','materials')),
    (1,('scene','motorbike')),
    # (1,('scene','cola')),
    (0.75,('effect','unhalf')),

]
SCRIPT = [
    (0,('left_view','ours')),
    (0,('right_view','edsr')),
    (0,('scene','dragon')),
    (0.1,('effect','half')),
    (0.1,('zoom',1.5)),
    (0.4,('right_view','swin')),
    (0.5,('right_view','srgan')),
    (0,('scene','lego')),
    (0.4,('right_view','rstt')),
    (0.1,('zoom',1.)),
    # (0.2,('effect','unhalf')),
]
SCRIPT = [
    (0,('left_view','edsr')),
    # (0,('right_view','preSR')),
    (0,('scene','lego')),
    # (0.1,('effect','half')),
    # (0.1,('zoom',1.5)),
    # (0.5,('right_view','preSR')),
    # (0,('scene','lego')),
    # (0.4,('right_view','rstt')),
    # (0.1,('zoom',1.)),
    # (0.2,('effect','unhalf')),
]
FPS = 5
# ZOOMIN_FACTOR = 1.5
TRANSITION_TIMES = {'zoom':1,'fade':0.5,'half':1}

HR_DS_FACTOR = 2
SR_FACTOR = 4
HIGH_RES_OUTPUT = False
EXCLUDE_TITLE = True #False #
FRAME_NUM = False
GIF_LIKE_SAVING = [36,67]

OUR_RESULTS_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/ours'
BSELINES_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/baselines'
OUTPUT_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/comparisons'
SOURCES = {
    'GT':{'p_im':'(?<=\/r_)(\d)+(?=\.png$)','p_scene':lambda scene:scene+'/test/*','path':'/scratch/gpfs/yb6751/datasets/Synthetic',},
    'LR':{'title':'Low-res.','p_im':'(?<=\/)(\d)+(?=\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*LR/*'%(HR_DS_FACTOR*SR_FACTOR),'path':OUR_RESULTS_PATH,},
    'ours':{'title':'Ours','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR),'path':OUR_RESULTS_PATH},
    'naive':{'title':'Naive','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*fine/*'%(HR_DS_FACTOR),'path':OUR_RESULTS_PATH},
    'edsr':{'title':'EDSR','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'edsr_pre':{'title':'EDSR (Data)','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'srgan':{'title':'SRGAN','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'srgan_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'rstt':{'title':'Video SR','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'rstt_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'swin':{'title':'SwinIR','p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'swin_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':lambda scene:scene+'/*','path':BSELINES_PATH},
    'preSR':{'title':'Pre-SR','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/blind_fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/planes_on_SR_Res800_32_0'},
    'view':{'title':'4 planes','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/Synt_Res16Sc800_32_400_NerfUseviewdirsTrue_1'},
    'no_view':{'title':'3 planes','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/Synt_Res16Sc800_32_400_NerfUseviewdirsFalse_0'},
    'PlRes100':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes100*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    'PlRes400':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes400*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    'PlRes1600':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes1600*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    'bicubic':{},
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
# if cur_scene is not None and cur_left is not None and sc[1][0]=='scene':
#     method_scene_pairs.append((cur_left,cur_scene))
# if cur_scene is not None and cur_right is not None and sc[1][0]=='scene':
#     method_scene_pairs.append((cur_right,cur_scene))
if cur_scene is not None and cur_left is not None and sc[1][0] in ['scene','left_view']:
    method_scene_pairs.append((cur_left,cur_scene))
if cur_scene is not None and cur_right is not None and sc[1][0] in ['scene','right_view']:
    method_scene_pairs.append((cur_right,cur_scene))


included_baselines = list(set([v[1][1] for v in SCRIPT if v[1][0] in ['right_view','left_view']]))
methods_2_load = included_baselines
scenes = list(set([v[1][1] for v in SCRIPT if v[1][0]=='scene']))
im_paths = defaultdict(dict)
# for scene in scenes:
#     for i_method,method in enumerate(methods_2_load):
for i_method,(method,scene) in enumerate(set(method_scene_pairs)):
    if 'path' not in SOURCES[method]:    continue
    im_paths[method][scene] = [f for f in glob(os.path.join(SOURCES[method]['path'],SOURCES[method]['p_scene'](scene))) if search(SOURCES[method]['p_im'],f) is not None]
    im_paths[method][scene] = OrderedDict(sorted([(int(search(SOURCES[method]['p_im'],f).group(0)),f) for f in im_paths[method][scene]],key=lambda x:x[0]))
    if i_method>0: #Not the first method for this scene, assert the number of images is bigger than 0 and the same
        # assert len(im_paths[method][scene])>0
        assert len(im_paths[method][scene])==num_frames
    else:
        num_frames = len(im_paths[method][scene])
        assert num_frames>0

script,n_frames = [],0
zoomin,fade,right_views,left_views,scene2show,halfing = [],[],[],[],[],[]
latest_zoom = 1
for sc in SCRIPT:
    if sc[1][0]=='effect':
        if 'half' in sc[1][1]:
            if n_frames+sc[0]*num_frames==0:
                halfing.append((0,1))
            else:
                halfing.extend(list(zip(
                    [int(n_frames+sc[0]*num_frames-FPS*TRANSITION_TIMES['half']/2+i) for i in range(int(FPS*TRANSITION_TIMES['half']))],
                    np.linspace(0,1,FPS*TRANSITION_TIMES['half'])[::(1 if sc[1][1]=='half' else -1)])))
        else:
            raise Exception
    elif sc[1][0]=='zoom':
        zoomin.extend(list(zip(
            [int(n_frames+sc[0]*num_frames-FPS*TRANSITION_TIMES['zoom']/2+i) for i in range(int(FPS*TRANSITION_TIMES['zoom']))],
            np.linspace(latest_zoom,sc[1][1],FPS*TRANSITION_TIMES['zoom']))))
        latest_zoom = sc[1][1]
    elif sc[1][0]=='scene':
        if n_frames>FPS*TRANSITION_TIMES['fade']/2:
            fade.extend(list(zip(
                [int(n_frames+sc[0]*num_frames-FPS*TRANSITION_TIMES['fade']/2+i) for i in range(int(FPS*TRANSITION_TIMES['fade']))],
                np.concatenate([np.linspace(0,1,int(FPS*TRANSITION_TIMES['fade']/2)),np.linspace(1,0,int(FPS*TRANSITION_TIMES['fade']/2))]))))
        scene2show.append((n_frames+sc[0]*num_frames,sc[1][1]))
    elif sc[1][0]=='right_view':
        right_views.append((n_frames+sc[0]*num_frames,sc[1][1]))
    elif sc[1][0]=='left_view':
        left_views.append((n_frames+sc[0]*num_frames,sc[1][1]))
    else:
        raise Exception
    n_frames += sc[0]*num_frames
# total_frames = int(scene2show[-1][0]+num_frames)
total_frames = int(max(1,np.ceil(n_frames/num_frames))*num_frames)
frames = []
def read_image(path):
    return imageio.imread(path)

frame_shape = np.array(read_image(im_paths[cur_left][cur_scene][0]).shape[:2])

SEPARATOR_WIDTH = 2
FONT_SCALE = 2
THICKNESS = 2
TEXT_Y = 30
if HIGH_RES_OUTPUT:
    TEXT_Y = int(1080/frame_shape[1]*TEXT_Y)
    FONT_SCALE = int(0.6*1080/frame_shape[1]*FONT_SCALE)
    THICKNESS = int(0.6*1080/frame_shape[1]*THICKNESS)
cur_fade,cur_zoom,cur_half = 0,1,0
output_width = 1080 if HIGH_RES_OUTPUT else frame_shape[1]
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
        cur_half = halfing.pop(0)[1]
    left = read_image(im_paths[cur_left][cur_scene][f_num%num_frames])
    if cur_zoom>1:
        left = cv2.resize(left, dsize=(0,0),fx=cur_zoom,fy=cur_zoom, interpolation=cv2.INTER_AREA)
        leftovers = np.array(left.shape)[:2]-frame_shape
        left = left[leftovers[0]//2:-(leftovers[0]-leftovers[0]//2),leftovers[1]//2:-(leftovers[1]-leftovers[1]//2),:]
    if HIGH_RES_OUTPUT:
        left = cv2.resize(left, dsize=(1080,1080),interpolation=cv2.INTER_AREA)
    title = '' if EXCLUDE_TITLE else SOURCES[cur_left]['title']+(', %d'%(f_num) if FRAME_NUM else '')
    left = cv2.putText(
        left,
        title,
        (0,TEXT_Y),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=FONT_SCALE,
        color=[255,255,255],
        thickness=THICKNESS,
    )
    new_frame = [left[:,:int(output_width*(1-cur_half/2)),:]]
    if cur_half>0:
        separator = 255*np.ones([1080 if HIGH_RES_OUTPUT else frame_shape[0],SEPARATOR_WIDTH,3])
        right = read_image(im_paths[cur_right][cur_scene][f_num%num_frames])
        if cur_zoom>1:
            right = cv2.resize(right, dsize=(0,0),fx=cur_zoom,fy=cur_zoom, interpolation=cv2.INTER_AREA)
            leftovers = np.array(right.shape)[:2]-frame_shape
            right = right[leftovers[0]//2:-(leftovers[0]-leftovers[0]//2),leftovers[1]//2:-(leftovers[1]-leftovers[1]//2),:]
        if HIGH_RES_OUTPUT:
            right = cv2.resize(right, dsize=(1080,1080),interpolation=cv2.INTER_AREA)
        if cur_half==1:
            title = '' if EXCLUDE_TITLE else SOURCES[cur_right]['title']+(', %d'%(f_num) if FRAME_NUM else '')
            right = cv2.putText(
                right,
                title,
                (output_width-cv2.getTextSize(title, cv2.FONT_HERSHEY_PLAIN, FONT_SCALE,THICKNESS)[0][0],TEXT_Y),
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=FONT_SCALE,
                color=[255,255,255],
                thickness=THICKNESS,
            )
        # new_frame.extend([separator,right[:,-(1+max(SEPARATOR_WIDTH,(int(frame_shape[1]*cur_half/2-SEPARATOR_WIDTH)))):,:]])
        new_frame.extend([separator,right[:,-(1+max(SEPARATOR_WIDTH,(int(output_width*cur_half/2-SEPARATOR_WIDTH)))):,:]])
    # if cur_fade>0:
    #     left = (1-cur_fade)*left
    #     right = (1-cur_fade)*right
    # new_frame = []
    new_frame = np.concatenate(new_frame,1)[:,:output_width,:]
    if HIGH_RES_OUTPUT:
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
    frames = frames[GIF_LIKE_SAVING[0]:GIF_LIKE_SAVING[1]+1]+frames[GIF_LIKE_SAVING[1]-1:GIF_LIKE_SAVING[0]:-1]
imageio.mimwrite(vid_path, frames, fps = FPS, macro_block_size = 8)  # pip install imageio-ffmpeg
