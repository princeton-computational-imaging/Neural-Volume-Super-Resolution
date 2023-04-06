import numpy as np
import os
from glob import glob
from re import search
from collections import OrderedDict,defaultdict
import imageio
import cv2
from tqdm import tqdm
import math
from skimage.metrics import structural_similarity
import lpips
import torch
import sys
sys.path.append('.')
import pickle
from nerf_helpers import imread,calc_resize_crop_margins
# from nerf_helpers import bicubic_interp

DEBUG_MODE = False #False #True
FIND_LEADING = True #True #False
REF_SCORES = 'save' #None #'load','save'
SORT_SCORES = True #False
DTU_SCENES = False #False #True
LLFF_SCENES = False
if DTU_SCENES:
    USE_ALL_SCENES = True
    HR_DS_FACTOR = 1
    GT_IMS_PATH = '/scratch/gpfs/yb6751/datasets/rs_dtu_4/DTU'
elif LLFF_SCENES:
    USE_ALL_SCENES = False
    HR_DS_FACTOR = 8
else:
    USE_ALL_SCENES = False
    HR_DS_FACTOR = 2
    GT_IMS_PATH = '/scratch/gpfs/yb6751/datasets/Synthetic'

SR_FACTOR = 4
GT_AS_LR_INPUTS = False #True
# SCENES = ['ship','mic','chair','lego'] # Table 1 in paper
# SCENES = ['ship','motorbike','dragon','bugatti'] #,'donut','cola','dragon']
# SCENES = ['lego','motorbike','chair','bugatti'] #,'donut','cola','dragon']
# SCENES = ['motorbike','bugatti','holiday','cola','dragon','materials','ship'] # 4th plane ablation study
# SCENES = ['motorbike','bugatti','chair','donut','lego','materials','teddy'] # Plane resolution ablation study
# SCENES = ['ship##Gauss2','mic##Gauss2'] #,'leaves','horns' Real scenes
# SCENES = ['ship##Gauss2','mic##Gauss2','lego##Gauss2','chair##Gauss2']
# SCENES = ['ship','mic','lego','chair']
SCENES = ['chair','mic','ship','lego'] #
# SCENES = ['orchids','fern'] #,'chair','ship'
DEGRADATION = 'Noise5' #None #'Gauss1'

RESULTS_PATH = '/tigress/yb6751/projects/NeuralMFSR/results/'
SAVE_BICUBIC = RESULTS_PATH+'bicubic_ours'
# OUR_RESULTS_PATH = RESULTS_PATH+'ours'
# EXPERIMENT_ID = 'ours'
# EXPERIMENT_ID = 'E2E_Synt_Res29Sc200_27Sc800_32_LR100_400_fromDetachedLR_imConsistLossFreq10nonSpatial_WOplanes_HrLr_micShip_0_intermediate'
# EXPERIMENT_ID = 'ours_ICCV_perSceneRefine'
EXPERIMENT_ID = 'ours_gaussian'
# OUR_RESULTS_PATH = RESULTS_PATH+EXPERIMENT_ID
if DTU_SCENES:
    OUR_RESULTS_PATH = RESULTS_PATH+'E2E_DTU_103ScRes200_88Sc800_32_LR100_400_posFeatCatDecCh256_0'
else:
    # OUR_RESULTS_PATH = RESULTS_PATH+'E2E_Synt_Res29Sc200_27Sc800_32_LR100_400_posFeatCatDecCh256_andGauss_0'
    OUR_RESULTS_PATH = RESULTS_PATH+EXPERIMENT_ID

# OUR2_RESULTS_PATH = RESULTS_PATH+'E2E_Synt_Res29Sc200_27Sc800_32_LR100_400_posFeatCat_andGauss_0'
# BASELINES_PATH = RESULTS_PATH+'baselines'+'/'+EXPERIMENT_ID
# BASELINES_PATH = OUR_RESULTS_PATH.replace(RESULTS_PATH,RESULTS_PATH+'baselines/')
BASELINES_PATH = RESULTS_PATH+'baselines'
POST_SR_PATH = os.path.join(BASELINES_PATH,'postSR_MipNeRF')
PER_SCENE = False
PER_MODEL_PSNR_GAIN = True # When using models other than our plane-based (e.g. NeRF), compute PSNR gain with respect to the LR version produced by that model.

postSR_pattern = lambda scene:scene+'_DS%d/*'%(HR_DS_FACTOR*SR_FACTOR)
METHODS = OrderedDict({
    'GT_synt':{'p_im':'(?<=\/r_)(\d)+(?=\.png$)','p_scene':lambda scene:(scene[:scene.find('##')] if '##' in scene else scene)+'/test/*','path':GT_IMS_PATH,},
    # 'GT_real':{'p_im':'(?<=\/image)(\d)+(?=\.png$)','p_scene':lambda scene:scene+'/images_8/*','path':'/scratch/gpfs/yb6751/datasets/LLFF',},
    # 'LR':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*fine/*'%(HR_DS_FACTOR*SR_FACTOR),'path':OUR_RESULTS_PATH,},
    'ours':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR),'path':OUR_RESULTS_PATH},
    # 'ours_noSR':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:'noSR_'+scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR),'path':OUR_RESULTS_PATH},
    # 'ours_preICCV':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:'old_'+scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR),'path':OUR_RESULTS_PATH},
    # 'ours_noImConsist':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:'noImConsist_'+scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR),'path':OUR_RESULTS_PATH},
    # 'ours2':{'title':'Ours','p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*SR/*'%(HR_DS_FACTOR),'path':OUR2_RESULTS_PATH},
    'naive':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d_PlRes*/*fine/*'%(HR_DS_FACTOR),'path':OUR_RESULTS_PATH},
    # 'nerf':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)\.png$)','p_scene':lambda scene:'lr_nerf_'+scene+'_0/'+scene+'_DS%d/blind_fine/*'%(HR_DS_FACTOR),'path':os.path.join(BASELINES_PATH,'nerf'),'own_lr':True},
    # 'mip_nerf':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)\.png$)','p_scene':lambda scene:'lr_nerf_'+scene+'_mip_0/'+scene+'_DS%d/blind_fine/*'%(HR_DS_FACTOR),'path':os.path.join(BASELINES_PATH,'nerf'),'own_lr':True},
    'mip_nerf':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d/blind_fine/*'%(HR_DS_FACTOR),'path':os.path.join(RESULTS_PATH,'baselines/MipNeRF')}, # New pre-SR baseline configuration, using Mip-NeRF representation model trained on images super-resolved using a pre-trained SwinIR model.
    # 'rstt_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'rstt':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_RSTT_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'swin_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'srgan_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'edsr_pre':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Pretrained_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'swin':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SWIN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'srgan':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_SRGAN_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'edsr':{'p_im':'(?<=\/)(\d)+(?=(_PSNR.*)?_EDSR_Scratch_upscaleX%d\.png$)'%(SR_FACTOR),'p_scene':postSR_pattern,'path':POST_SR_PATH},
    # 'preSR':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/blind_fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/planes_on_SR_Res800_32_0'}, # Old pre-SR baseline configuration, using planes representation model trained on images super-resolved using a trained-from-scratch EDSR model
    # 'preSR':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2/blind_fine/*','path':os.path.join(RESULTS_PATH,'baselines/MipNeRF_preSR')}, # New pre-SR baseline configuration, using Mip-NeRF representation model trained on images super-resolved using a pre-trained SwinIR model.
    # 'preSRscratch':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS%d/blind_fine/*'%(HR_DS_FACTOR),'path':os.path.join(RESULTS_PATH,'baselines/MipNeRF_preSRscratch')}, # New pre-SR baseline configuration, using Mip-NeRF representation model trained on images super-resolved using a SwinIR model trained from scratch on our scene training set.
    # 'view':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/Synt_Res16Sc800_32_400_NerfUseviewdirsTrue_1'},
    # 'no_view':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/Synt_Res16Sc800_32_400_NerfUseviewdirsFalse_0'},
    # 'PlRes100':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes100*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    # 'PlRes400':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes400*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    # 'PlRes1600':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes1600*/*fine/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/plane_resolution_RES100_400_1600_0'},
    # 'no_PlMeanZero':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*SR/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/E2E_SyntAndReal_20vs4_1'},
    # 'PlMeanZero':{'p_im':'(?<=\/)(\d)+(?=_PSNR.*\.png$)','p_scene':lambda scene:scene+'_DS2_PlRes*/*SR/*','path':'/tigress/yb6751/projects/NeuralMFSR/results/E2E_SyntAndReal_20vs4_PZerMean1_1'},
    # 'bicubic':{},
})
GT_method = 'GT_synt' if 'GT_synt' in METHODS else 'GT_real'
# METHODS_2_SHOW = ['edsr','bicubic','ours']
SCORES_2_SHOW = ['PSNR','SSIM','LPIPS'] #
# METHODS_2_SHOW = OrderedDict([(k,i) for i,k in enumerate(METHODS_2_SHOW)])
# assert [m in METHODS for m in METHODS_2_SHOW]
def calc_psnr(im1,im2):
    return -10.0 * math.log10(np.mean((im1-im2)**2))

def load_im(path):
    return  imread(path)
    # return (imageio.imread(path)/ 255.0).astype(np.float32)[...,:3]

assert REF_SCORES in ['save','load',None]
if 'LPIPS' in SCORES_2_SHOW:
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
im_paths = defaultdict(dict)
methods_2_load = list(METHODS.keys())
SCENES = [sc+'##%s'%(DEGRADATION) if DEGRADATION is not None else sc for sc in SCENES]
if USE_ALL_SCENES:
    assert DEGRADATION is None
    SCENES = [sc[:sc.find('_DS')] for sc in os.listdir(METHODS['ours']['path']) if '_DS%d'%(HR_DS_FACTOR) in sc]
for scene in SCENES:
    for i_method,method in enumerate(methods_2_load):
        if 'path' not in METHODS[method]:    continue
        if DTU_SCENES and method=='GT_synt':
            im_paths[method][scene] = [f for f in glob(os.path.join(METHODS[method]['path'],scene+'/image/*')) if any(['%03d.png'%(n)in f for n in [11, 23, 35, 47]])]
            # im_paths[method][scene] = OrderedDict(sorted([(int(search('(?<=\/)(\\d)+(?=\\.png$)',f).group(0)),f) for f in im_paths[method][scene]],key=lambda x:x[0]))
            im_paths[method][scene] = OrderedDict([(i,file) for i,file in enumerate(sorted([f for f in im_paths[method][scene]],key=lambda f:int(search('(?<=\/)(\\d)+(?=\\.png$)',f).group(0))))])
        else:
            im_paths[method][scene] = [f for f in glob(os.path.join(METHODS[method]['path'],METHODS[method]['p_scene'](scene))) if search(METHODS[method]['p_im'],f) is not None]
            im_paths[method][scene] = OrderedDict(sorted([(int(search(METHODS[method]['p_im'],f).group(0)),f) for f in im_paths[method][scene]],key=lambda x:x[0]))
        assert len(im_paths[method][scene])>0
        if i_method>0: #Not the first method for this scene, assert the number of images is bigger than 0 and the same
            assert len(im_paths[method][scene])==len(im_paths[methods_2_load[i_method-1]][scene])

scores = ['PSNR','SSIM','LPIPS']+(['psnr_gain'] if 'bicubic' in METHODS else [])
scores = dict(zip(scores,[defaultdict(lambda: defaultdict(list)) for s in scores]))
# psnrs,psnr_gain,ssim,lpips_scores = defaultdict(lambda: defaultdict(list)),defaultdict(lambda: defaultdict(list)),defaultdict(lambda: defaultdict(list)),defaultdict(lambda: defaultdict(list))
def for_lpips(im):
    return 2*torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0).cuda()-1
def bicubic_interp(im,sf):
    return cv2.resize(im, dsize=(im.shape[1]*sf,im.shape[0]*sf), interpolation=cv2.INTER_CUBIC)

for sc_num,scene in enumerate(SCENES):
    if SAVE_BICUBIC is not None and 'bicubic' in METHODS:
        if not os.path.isdir(os.path.join(SAVE_BICUBIC,scene)): os.mkdir(os.path.join(SAVE_BICUBIC,scene))
    # LR_ims,GT_ims,bicubic_ims = {},{},{}
    for im_num,im_path in tqdm(im_paths[GT_method][scene].items(),desc='Processing scene %s (%d/%d)'%(scene,sc_num+1,len(SCENES))):
        if DEBUG_MODE and im_num>3:
            print("!!!!!!!!!!! DEBUG MODE !!!!!!!!!!!!!")
            break
        GT_im = load_im(im_path)
        marg2crop = calc_resize_crop_margins(GT_im.shape,SR_FACTOR)
        if marg2crop is not None:
            GT_im = GT_im[marg2crop[0]:-marg2crop[0] if marg2crop[0]>0 else None,marg2crop[1]:-marg2crop[1] if marg2crop[1]>0 else None,:]
        if HR_DS_FACTOR>1 and 'images_%d'%(HR_DS_FACTOR) not in im_path:
            GT_im = cv2.resize(GT_im, dsize=(GT_im.shape[1]//HR_DS_FACTOR,GT_im.shape[0]//HR_DS_FACTOR), interpolation=cv2.INTER_AREA)
        if 'bicubic' in METHODS:
            if GT_AS_LR_INPUTS:
                LR_im = cv2.resize(GT_im, dsize=(GT_im.shape[1]//SR_FACTOR,GT_im.shape[0]//SR_FACTOR), interpolation=cv2.INTER_AREA)
            else:
                LR_im = load_im(im_paths['LR'][scene][im_num])
            bicubic_im = np.clip(bicubic_interp(LR_im,sf=SR_FACTOR),0,1)
        for i_method,method in enumerate(methods_2_load):
            if method in [GT_method,'LR']:    continue
            method_im =  bicubic_im if method=='bicubic' else load_im(im_paths[method][scene][im_num])
            scores['PSNR'][method][scene].append(calc_psnr(GT_im,method_im))
            if method=='bicubic' and SAVE_BICUBIC is not None:
                im_name = '%s%s.png'%(im_path.split('/')[-1].replace('.png',''),('_PSNR%.2f'%(scores['PSNR'][method][scene][-1])).replace('.','_'))
                imageio.imwrite(os.path.join(SAVE_BICUBIC,scene,im_name),(255*bicubic_im).astype(np.uint8))
            scores['SSIM'][method][scene].append(structural_similarity(GT_im,method_im,data_range=1,channel_axis=2))
            if 'LPIPS' in SCORES_2_SHOW:
                scores['LPIPS'][method][scene].append(loss_fn_alex(for_lpips(GT_im),for_lpips(method_im)).item())
        if 'bicubic' in scores['PSNR']:
            for method in scores['PSNR']:
                if method!='bicubic':
                    if PER_MODEL_PSNR_GAIN and METHODS[method].get('own_lr',False):
                        lr_file = im_paths[method][scene][im_num].replace('_DS%d'%(HR_DS_FACTOR),'_DS%d'%(HR_DS_FACTOR*SR_FACTOR))
                        lr_file = glob(lr_file[:lr_file.index('PSNR')]+'*')
                        assert len(lr_file)==1
                        bicubic_PSNR = calc_psnr(GT_im,np.clip(bicubic_interp(load_im(lr_file[0]),sf=SR_FACTOR),0,1))
                        scores['psnr_gain'][method][scene].append(scores['PSNR'][method][scene][-1]-bicubic_PSNR)
                    else:
                        scores['psnr_gain'][method][scene].append(scores['PSNR'][method][scene][-1]-scores['PSNR']['bicubic'][scene][-1])

def avergae_scores(scores_dict,per_scene=False):
    if per_scene:
        return dict([(m,dict([(k,(np.mean(v),np.std(v))) for k,v in scores_dict[m].items()])) for m in scores_dict])
    else:
        AVG_PER_SCENE_STDS = True
        return dict([(m,(np.mean([np.mean(v) for v in scores_dict[m].values()]),
            np.mean([np.std(v) for v in scores_dict[m].values()]) if AVG_PER_SCENE_STDS else np.std([v_ for v in scores_dict[m].values() for v_ in v])
            )) for m in scores_dict])

def find_leading(scores,metric,k):
    methods_4_comparison = [meth for meth in METHODS if meth not in ['LR',GT_method,'ours']]
    mean_advantages,mean_absolutes = {},{}
    for sc in scores[metric[0]]['ours']:
        STDs = [np.std([v for method in METHODS for v in scores[m][method][sc]]) for m in metric] if len(metric)>1 else [1]
        mean_advantages[sc] = [np.mean(np.array([np.array(scores[m]['ours'][sc])-np.array(scores[m][method][sc]) for method in methods_4_comparison]),0) for m in metric]
        mean_advantages[sc] = np.array([mean_advantages[sc][i]/STDs[i]*(-1 if metric[i]=='LPIPS' else 1) for i in range(len(metric))]).sum(0)
        mean_absolutes[sc] = [np.mean(np.array([np.array(scores[m]['ours'][sc]) for method in methods_4_comparison]),0) for m in metric]
        mean_absolutes[sc] = np.array([mean_absolutes[sc][i]/STDs[i]*(-1 if metric[i]=='LPIPS' else 1) for i in range(len(metric))]).sum(0)
    for sc in sorted(mean_advantages.keys(),key=lambda x:-max(mean_advantages[x])):
        print('\n%s:'%(sc))
        print('adavantages',np.argsort(mean_advantages[sc])[::-1][:k])
        print(['%.3f'%v for v in np.sort(mean_advantages[sc])[::-1][:k]])
        print('absolutes',np.argsort(mean_absolutes[sc])[::-1][:k])
        print(['%.3f'%v for v in np.sort(mean_absolutes[sc])[::-1][:k]])
if FIND_LEADING:
    find_leading(scores=scores,metric=['PSNR']+(['LPIPS'] if 'LPIPS' in SCORES_2_SHOW else []),k=10)
scores = dict([(k,avergae_scores(v,per_scene=PER_SCENE)) for k,v in scores.items()])
# sys.exit(0)
if SORT_SCORES:
    methods_order = dict([(k,[v[0] for v in sorted(scores[k].items(),key=lambda x:(x[1][0] if k=='LPIPS' else -x[1][0]))]) for k in scores])
else:
    methods_order = dict([(k,[m for m in METHODS if m in scores[k]]) for k in scores])

if REF_SCORES=='save':
    with open('reference_scores.pkl', 'wb') as f:
        pickle.dump(scores, f)
elif REF_SCORES=='load':
    with open('reference_scores.pkl', 'rb') as f:
        reference_scores = pickle.load(f)
for score in scores:
    if SORT_SCORES:
        print("\n%s %s:"%('Ascending' if score=='LPIPS' else 'Descending',score))
    else:
        print("\n%s (non-sorted!):"%(score))
    if REF_SCORES=='load':
        print("\t".join(["%s: %.3f +/-%.3f (%+.3f)"%(k,scores[score][k][0],scores[score][k][1],scores[score][k][0]-reference_scores[score][k][0]) for k in methods_order[score]]))
    else:
        print("\t".join(["%s: %.3f +/-%.3f"%(k,scores[score][k][0],scores[score][k][1]) for k in methods_order[score]]))
sys.exit(0)
# number0 = methods_order[-1]
# number1 = methods_order[-2]
def return_highlighting(method, score_name, score):
    if method == methods_order[score_name][-1]:
        return '\\textbf{%s}'%score
    elif method == methods_order[score_name][-2]:
        return '\\textit{%s}'%score
    else:
        return score
# def return_highlighting(idx, idy, i):
#     if idx == number0[idy]:
#         return '\\textbf{%s}'%i
#     elif idx == number1[idy]:
#         return '\\textit{%s}'%i
#     else:
#         return i
with open('reconstruction.tex','w') as f:
    f.write(
        "\\begin{table}\n\t\\centering\n\t\\begin{tabular}{lccc}\n\t\tApproach & METHODS & PSNR $(\\uparrow)$ & SSIM $(\\uparrow)$\\\\\n\t\t\\midrule"
    )
    # row = '\n\t\tApproach & Method'+'&'.join(SCORES_2_SHOW)
    # for method in METHODS_2_SHOW:
    #     row = 



# for idx, (row, method) in enumerate(zip(totalmAP, np.asarray(data_table_car)[:,0])):
#     row = [return_highlighting(idx, idy,'%.2f'%i) for idy, i in enumerate(row)]
#     row_out = ' & '.join(row)
#     print(exp_name[method.split(' ')[0]] + '& ', row_out, '\\\\')
