import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from tqdm import tqdm
import sys
sys.path.append('scripts')
from imresize_CEM import imresize

IM_PATH = '/scratch/gpfs/yb6751/projects/VolumetricEnhance/cache/nerf_synthetic/drums/test/r_0.png'
FACTOR = 4
UPSCALE = 2
PACKAGE = 'cv2' #'skimage','cv2','CEM'

def downscale(img,factor):
    im_shape = list(img.shape[:2])
    assert [v//factor==v/factor for v in im_shape]
    if PACKAGE=='skimage':
        return np.stack([rescale(img[...,i], 1/factor, anti_aliasing=True) for i in range(img.shape[2])],-1)
    elif PACKAGE=='cv2':
        return cv2.resize(img, dsize=(im_shape[1]//factor, im_shape[0]//factor), interpolation=cv2.INTER_AREA)
    elif PACKAGE=='CEM':
        return imresize(img,1/factor)


def lowpass(img,factor):
    im_shape = list(img.shape[:2])
    ds = downscale(img,factor)
    assert ds.shape[1]/ds.shape[0]==im_shape[1]/im_shape[0]
    if PACKAGE=='skimage':
        return np.stack([rescale(ds[...,i], UPSCALE*factor, anti_aliasing=True) for i in range(img.shape[2])],-1)
    elif PACKAGE=='cv2':
        return cv2.resize(ds, dsize=(UPSCALE*im_shape[1], UPSCALE*im_shape[0]), interpolation=cv2.INTER_CUBIC)
    elif PACKAGE=='CEM':
        return imresize(ds,UPSCALE*factor)


img = (imageio.imread(IM_PATH)/ 255.0).astype(np.float32)[...,:3]
low_passed = lowpass(img,FACTOR)
ds = downscale(img,FACTOR)
correlations,sampled, = [],[]
# plt.imsave('downsampled.png',np.clip(ds,0,1))
for offset in tqdm(range(UPSCALE*FACTOR)):
    sampled.append(low_passed[offset::UPSCALE*FACTOR,offset::UPSCALE*FACTOR,:])
    # plt.imsave('Offset%d.png'%(offset),np.clip(sampled[-1],0,1))
    correlations.append(np.sum((ds-np.mean(ds))*(sampled[-1]-np.mean(sampled[-1]))))
    print('Offset %d/%d: %.3f'%(offset,UPSCALE*FACTOR,correlations[-1]))
optimal_offset = np.argmax(correlations)
print('Max correlation (%.1f) obtained in %.3f offset'%(correlations[optimal_offset],optimal_offset/UPSCALE/FACTOR))
