import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os.path as osp
from scipy.ndimage import binary_closing, binary_opening, binary_erosion, binary_dilation
from scipy.ndimage import laplace
import json
from tqdm import tqdm
from sklearn.feature_extraction.image  import extract_patches_2d
# import einops

import h5py

EDSR_PRETRAINED = "EDSR_Pretrained"
EDSR_SCRATCH = "EDSR_Scratch"
RSTT_PRETRAINED = "RSTT_Pretrained"
RSTT_SCRATCH = "RSTT_Scratch"
SRGAN_PRETRAINED = "SRGAN_Pretrained"
SRGAN_SCRATCH = "SRGAN_Scratch"
SWIN_PRETRAINED = "SWIN_Pretrained"
SWIN_SCRATCH = "SWIN_Scratch"
PRE_SR = "pre_SR"
BICUBIC = "bicubic"
NAIVE = "naive"
NAIVE_MIP = "naive_mip"
COMPARE_WITH_GT = False
PER_CHANNEL_NORMALIZATION = False

BASELINES = [
    PRE_SR,
    #BICUBIC,
    NAIVE,
    NAIVE_MIP,
    #EDSR_PRETRAINED,
    EDSR_SCRATCH,
    #RSTT_PRETRAINED,
    RSTT_SCRATCH,
    #SRGAN_PRETRAINED,
    SRGAN_SCRATCH,
    #SWIN_PRETRAINED,
    SWIN_SCRATCH,
]

MIC = "mic"
SHIP = "ship"
CHAIR = "chair"
LEGO = "lego"

DATASETS = [
    MIC,
    SHIP,
    CHAIR,
    LEGO,
]
# DATASETS = [
#     MIC,
# ]

def load_image(path):
    return cv2.imread(path)[..., [2, 1, 0]].astype(float) / 255.

def load_data(basedir):
    
    with open(osp.join(basedir, 'transforms_test.json'), 'r') as fp:
        meta = json.load(fp)

    imgs = []
    poses = []

    for frame in meta['frames']:
        # if len(imgs)>5: 
        #     print('!!WARNING!!!!!!!!')
        #     break
        fname = osp.join(basedir, frame['file_path'] + '.png')
        imgs.append(load_image(fname))
        poses.append(np.array(frame['transform_matrix']))

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    return imgs, poses, K


def load_flow(basedir, idx, stride):
    with h5py.File(f'{basedir}/motion_{stride}/{idx}.hdf5', 'r') as f:
        forward_flow = f['forward_flow'][()]

    with h5py.File(f'{basedir}/motion_{stride}/{idx+1}.hdf5', 'r') as f:
        backward_flow = f['backward_flow'][()]
    return forward_flow, backward_flow

def load_flows(basedir):

    def sort_fn(path):
        idx = int(osp.basename(path).split('.')[0])
        return idx

    files = sorted(glob.glob(f"{basedir}/motion_1/*.hdf5"), key=sort_fn)
    forward_flows = []
    backward_flows = []

    for idx in range(len(files[:-1])):
        file = files[idx]
        with h5py.File(file, 'r') as f:
            forward_flows.append(f['forward_flow'][()])
            if idx > 0: # no backward flow at index 0
                backward_flows.append(f['backward_flow'][()])
    
    # add last backward flow (no forward flow exists here)
    with h5py.File(files[-1], 'r') as f:
        backward_flows.append(f['backward_flow'][()])
    
    return forward_flows, backward_flows

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    #flow = -flow
    flow = np.copy(flow)
    flow[:,:,0] = np.arange(w) + flow[:,:,0]
    flow[:,:,1] = np.arange(h)[:, np.newaxis] + flow[:,:,1]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def compute_forward_warp_mask(backward_flow):
    h, w, c, = backward_flow.shape

    trans_pos = np.zeros_like(backward_flow)
    trans_pos[:, :, 0] = np.arange(w) + backward_flow[:, :, 0]
    trans_pos[:, :, 1] = np.arange(h)[:, np.newaxis] + backward_flow[:, :, 1]
    trans_pos_floor = np.floor(trans_pos).astype('int')
    trans_pos_floor[:, :, 0] = np.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w-1)
    trans_pos_floor[:, :, 1] = np.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h-1)

    warped_image = np.zeros(shape=(h, w, 1), dtype=np.float64)

    np.add.at(warped_image, (trans_pos_floor[:,:,1].ravel(), trans_pos_floor[:,:,0].ravel()), 1)

    return warped_image.squeeze()

def load_everything(basedir, dataset, baseline,load_ours=True):

    imgs, poses, K = load_data(basedir)

    # load our output

    def sort_fn(path):
        basename = osp.basename(path)
        idx = int(basename.split("_")[0])
        return idx
    
    if baseline == BICUBIC:
        def sort_fn_bicubic(path):
            basename = osp.basename(path)
            idx = int(basename.split("_")[1])
            return idx
        baseline_sr_paths = sorted(glob.glob(osp.join(basedir, BICUBIC, "*.png")), key=sort_fn_bicubic)
    elif baseline == PRE_SR:
        baseline_sr_paths = sorted(glob.glob(osp.join(basedir, PRE_SR, "*.png")), key=sort_fn)
    elif baseline == NAIVE:
        baseline_sr_paths = sorted(glob.glob(osp.join(basedir, "sr_model", "blind_fine", "*.png")), key=sort_fn)
    elif baseline == NAIVE_MIP:
        baseline_sr_paths = sorted(glob.glob(osp.join(basedir, "MipNerf", "blind_fine", "*.png")), key=sort_fn)
    else:
        baseline_sr_paths = sorted(glob.glob(osp.join(basedir, "baselines", f"*{baseline}_upscaleX4.png")), key=sort_fn)
    baseline_srs = [load_image(path) for path in baseline_sr_paths]

    if load_ours:
        vol_sr_paths = sorted(glob.glob(osp.join(basedir, "sr_model", "blind_SR", "*.png")), key=sort_fn)
        vol_srs = [load_image(path) for path in vol_sr_paths]
    else:
        vol_srs = None

    forward_flows, backward_flows = load_flows(basedir)

    return imgs, baseline_srs, vol_srs, forward_flows, backward_flows

def compute_errors(idx, imgs, baseline_srs, vol_srs, forward_flows, backward_flows):
    stride = 1
    forward_flow = forward_flows[idx]
    backward_flow = backward_flows[idx]
    
    # gt test data
    gt_ref, gt_forward = imgs[idx], imgs[idx + stride]
    #gt_ref = cv2.resize(gt_ref, (100, 100), interpolation=cv2.INTER_AREA).astype(float) / 255.
    gt_ref = cv2.resize(gt_ref, (400, 400), interpolation=cv2.INTER_LINEAR)
    #gt_forward = cv2.resize(gt_forward, (100, 100), interpolation=cv2.INTER_AREA).astype(float) / 255.
    gt_forward = cv2.resize(gt_forward, (400, 400), interpolation=cv2.INTER_LINEAR)

    if vol_srs is not None:
        vol_sr_ref = vol_srs[idx]
        vol_sr_forward = vol_srs[idx + stride]
    baseline_sr_ref = baseline_srs[idx]
    baseline_sr_forward = baseline_srs[idx + stride]

    mask = np.all(gt_ref == 0, axis=2)
    forward_warped_mask = compute_forward_warp_mask(backward_flow) == 0
    forward_warped_mask = binary_opening(forward_warped_mask, iterations=3)
    forward_warped_mask = binary_dilation(forward_warped_mask, iterations=1)
    mask |= forward_warped_mask


    gt_warped = warp_flow(gt_forward, forward_flow).astype(float)
    gt_warped[mask] = 0
    if vol_srs is not None:
        vol_sr_warped = warp_flow(vol_sr_forward, forward_flow).astype(float)
    # vol_sr_warped[mask] = 0
    baseline_sr_warped = warp_flow(baseline_sr_forward, forward_flow).astype(float)
    # baseline_sr_warped[mask] = 0

    # everything into 7x7 patches
    PATCH_SIZE = 7
    def im2patches(im):
        return extract_patches_2d(im,(PATCH_SIZE,PATCH_SIZE)).reshape([-1,PATCH_SIZE**2,3 if im.ndim==3 else 1])
    if COMPARE_WITH_GT:
        gt_warped = im2patches(gt_warped)
        gt_ref = im2patches(gt_ref)
    if vol_srs is not None:
        vol_sr_warped = im2patches(vol_sr_warped)
        vol_sr_ref = im2patches(vol_sr_ref)
    baseline_sr_warped = im2patches(baseline_sr_warped)
    baseline_sr_ref = im2patches(baseline_sr_ref)
    mask = im2patches(mask)
    # gt_warped = einops.rearrange(gt_warped, '(h p1) (w p2) c -> (h w) (p1 p2) c', p1=8, p2=8)
    # gt_ref = einops.rearrange(gt_ref, '(h p1) (w p2) c -> (h w) (p1 p2) c', p1=8, p2=8)
    # vol_sr_warped = einops.rearrange(vol_sr_warped, '(h p1) (w p2) c -> (h w) (p1 p2) c', p1=8, p2=8)
    # vol_sr_ref = einops.rearrange(vol_sr_ref, '(h p1) (w p2) c -> (h w) (p1 p2) c', p1=8, p2=8)
    # baseline_sr_warped = einops.rearrange(baseline_sr_warped, '(h p1) (w p2) c -> (h w) (p1 p2) c', p1=8, p2=8)
    # baseline_sr_ref = einops.rearrange(baseline_sr_ref, '(h p1) (w p2) c -> (h w) (p1 p2) c', p1=8, p2=8)
    # mask = einops.rearrange(mask, '(h p1) (w p2) -> (h w) (p1 p2)', p1=8, p2=8)
    # normalize patches to have unit variance
    # std = np.any(np.std(gt_warped, axis=1, keepdims=True) < 1e-5, axis=2)
    # mask = mask | std
    # def normalize(img):
    #     img = img - img.mean(1, keepdims=True)
    #     img = img / (np.std(1, keepdims=True) + 1e-5)
    #     return img
    # gt_warped = normalize(gt_warped)
    # gt_ref = normalize(gt_ref)
    # vol_sr_warped = normalize(vol_sr_warped)
    # vol_sr_ref = normalize(vol_sr_ref)
    # baseline_sr_warped = normalize(baseline_sr_warped)
    # baseline_sr_ref = normalize(baseline_sr_ref)
    def im_std(im):
        if PER_CHANNEL_NORMALIZATION:
            return np.std(im,1,keepdims=True)
        else:
            return np.mean(np.std(im,1,keepdims=True),2,keepdims=True)
    
    def error_fn(ref, comp,mask):
         ref_std = im_std(ref)
         comp_std = im_std(comp)
         std_threshold = 3/255
         mask = np.any(mask,1)
         mask |= np.min(ref_std,2,keepdims=True).squeeze(-1)<std_threshold
         mask |= np.min(comp_std,2,keepdims=True).squeeze(-1)<std_threshold
         mask = mask.squeeze()
         return np.abs((ref[~mask]/ref_std[~mask]-comp[~mask]/comp_std[~mask])).mean()
        #  return (abs(ref - comp)).mean(-1)


    # def error_fn(ref, comp):
    #      return (abs(ref - comp)).mean(-1)

    # error_gt = error_fn(gt_ref, gt_warped,mask)
    if vol_srs is not None:
        error_vol_sr_to_ref = error_fn(vol_sr_ref, vol_sr_warped,mask)# / (vol_laplacian + 1./ 255.)
        if COMPARE_WITH_GT:
            error_vol_sr_to_gt = error_fn(gt_ref, vol_sr_warped,mask)
    error_baseline_sr_to_ref = error_fn(baseline_sr_ref, baseline_sr_warped,mask)# / (baseline_laplacian + 1./255.)
    if COMPARE_WITH_GT:
        error_baseline_sr_to_gt = error_fn(gt_ref, baseline_sr_warped,mask)

    return {
        # 'mse_vol_sr_to_ref' : error_vol_sr_to_ref[~mask].mean(),
        # 'mse_baseline_sr_to_ref': error_baseline_sr_to_ref[~mask].mean(),
        # 'mse_vol_sr_to_gt': error_vol_sr_to_gt[~mask].mean(),
        # 'mse_baseline_sr_to_gt': error_baseline_sr_to_gt[~mask].mean()
        'mse_vol_sr_to_ref' : None if vol_srs is None else error_vol_sr_to_ref,
        'mse_baseline_sr_to_ref': error_baseline_sr_to_ref,
        'mse_vol_sr_to_gt': None if vol_srs is None or not COMPARE_WITH_GT else error_vol_sr_to_gt,
        'mse_baseline_sr_to_gt': None if not COMPARE_WITH_GT else error_baseline_sr_to_gt,
    }


if __name__ == '__main__':

    #baseline = EDSR_SCRATCH

    error_mean_dict = {}
    PRINT_4_EACH_DATASET = True
    for b_num,baseline in enumerate(BASELINES):
        global_error_list_baseline = []
        global_error_list_ours = []
        for dataset in tqdm(DATASETS):
            basedir = osp.join("/tigress/yb6751/projects/NeuralMFSR/results/AVI_computation", "iccv", dataset)
            imgs, baseline_srs, vol_srs, forward_flows, backward_flows = load_everything(basedir, dataset, baseline,load_ours=b_num==0)
            idx_range = range(len(imgs) - 1) # last image has no forward flow counterpart

            error_by_idx = lambda i : compute_errors(i, imgs, baseline_srs, vol_srs, forward_flows, backward_flows)

            errors = [error for error in map(error_by_idx, idx_range)]

            vol_errors = [error['mse_vol_sr_to_ref'] for error in errors]
            baseline_errors = [error['mse_baseline_sr_to_ref'] for error in errors]
            vol_errors_to_gt = [error['mse_vol_sr_to_gt'] for error in errors]
            baseline_errors_to_gt = [error['mse_baseline_sr_to_gt'] for error in errors]

            global_error_list_baseline.append(baseline_errors)
            global_error_list_ours.append(vol_errors)
            if PRINT_4_EACH_DATASET:
                if vol_srs is not None:
                    print('Dataset %s'%(dataset))
                    print("volumetric SR consistency error: %.3f+/-%.3f"%(np.mean(vol_errors),np.var(vol_errors)))
                    # print(f"Mean volumetric SR consistency error: {np.mean(vol_errors)}")
                print("%s consistency error: %.3f+/-%.3f"%(baseline,np.mean(baseline_errors),np.var(baseline_errors)))
                # print(f"Mean {baseline} SR consistency error: {np.mean(baseline_errors)}")
                # print(f"Mean volumetric SR consistency error (to gt): {np.mean(vol_errors_to_gt)}")
                # print(f"Mean baseline SR consistency error (to gt): {np.mean(baseline_errors_to_gt)}")
                # print("------------------------------------------")
                # if vol_srs is not None:
                #     print(f"Variance volumetric SR consistency error: {np.var(vol_errors)}")
                # print(f"Variance {baseline} SR consistency error: {np.var(baseline_errors)}")
                # print(f"Variance volumetric SR consistency error (to gt): {np.var(vol_errors_to_gt)}")
                # print(f"Variance baseline SR consistency error (to gt): {np.var(baseline_errors_to_gt)}")
                # print("------------------------------------------")
                # print("------------------------------------------")

        # error_mean_dict[baseline] = np.mean(global_error_list_baseline)
        # if vol_srs is not None:
        #     error_mean_dict["Ours"] = np.mean(global_error_list_ours)
        #     print("MSE of ours: %.3f"%(np.mean(global_error_list_ours)))
        error_mean_dict[baseline] = (np.mean(global_error_list_baseline),np.std(global_error_list_baseline))
        if vol_srs is not None:
            error_mean_dict["Ours"] = (np.mean(global_error_list_ours),np.std(global_error_list_ours))
            print("MSE of ours: %.3f"%(np.mean(global_error_list_ours)))
        print("MSE of baseline %s: %.3f+/-%.3f"%(baseline,np.mean(global_error_list_baseline),np.std(global_error_list_baseline)))

        with open('iccv.json', 'w') as fp:
            json.dump(error_mean_dict, fp, indent=4)

    # fig, (ax0, ax1) = plt.subplots(1, 2)

    # ax0.plot(vol_errors, label='Volumetric SR MSE')
    # ax0.plot(baseline_errors, label="Baseline SR MSE")
    # ax0.legend()
    # ax0.set_title('To SR ref')

    # ax1.plot(vol_errors_to_gt, label='Volumetric SR MSE')
    # ax1.plot(baseline_errors_to_gt, label="Baseline SR MSE")
    # ax1.legend()
    # ax1.set_title('To GT ref')

    plt.show()