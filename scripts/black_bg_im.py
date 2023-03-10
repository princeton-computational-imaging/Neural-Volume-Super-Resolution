import imageio
import cv2
import sys
sys.path.append('.')
from nerf_helpers import calc_resize_crop_margins
import numpy as np

# IM_PATH = '/scratch/gpfs/yb6751/datasets/Synthetic/%s/test/r_%d.png'
# HR_SF = 2
# LR_SF = 8
IM_PATH = '/scratch/gpfs/yb6751/datasets/LLFF/%s/images_8/image%03d.png'
HR_SF = 1
LR_SF = 4

SCENE = 'fern'

for im_num in [10]:
    im_path = IM_PATH%(SCENE,im_num)
    im = imageio.imread(im_path)
    org_shape = np.array(im.shape[:2])
    marg2crop = calc_resize_crop_margins(org_shape,LR_SF//HR_SF)
    if marg2crop is not None:
        im = im[marg2crop[0]:-marg2crop[0] if marg2crop[0]>0 else None,marg2crop[1]:-marg2crop[1] if marg2crop[1]>0 else None,:]
        org_shape[:2] -= 2*marg2crop
    im = cv2.resize(im, dsize=(org_shape[1]//HR_SF,org_shape[0]//HR_SF), interpolation=cv2.INTER_AREA)
    saving_name = SCENE+'_'+im_path.split('/')[-1]
    imageio.imsave(saving_name,im=im[...,:3])
    imageio.imsave(saving_name.replace('.','_LR.'),im=cv2.resize(cv2.resize(im[...,:3], dsize=(org_shape[1]//LR_SF,org_shape[0]//LR_SF), interpolation=cv2.INTER_AREA), dsize=(org_shape[1]//HR_SF,org_shape[0]//HR_SF), interpolation=cv2.INTER_NEAREST))