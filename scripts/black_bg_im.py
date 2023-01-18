import imageio
import cv2

IM_PATH = '/scratch/gpfs/yb6751/datasets/Synthetic/%s/train/r_%d.png'
SCENE = 'hotdog'
HR_SF = 2
LR_SF = 8

for im_num in [0,30,50]:
    im_path = IM_PATH%(SCENE,im_num)
    im = imageio.imread(im_path)
    org_shape = im.shape[:2]
    im = cv2.resize(im, dsize=(org_shape[0]//HR_SF,org_shape[1]//HR_SF), interpolation=cv2.INTER_AREA)
    saving_name = SCENE+'_'+im_path.split('/')[-1]
    imageio.imsave(saving_name,im=im[...,:3])
    imageio.imsave(saving_name.replace('.','_LR.'),im=cv2.resize(cv2.resize(im[...,:3], dsize=(org_shape[0]//LR_SF,org_shape[1]//LR_SF), interpolation=cv2.INTER_AREA), dsize=(org_shape[0]//HR_SF,org_shape[1]//HR_SF), interpolation=cv2.INTER_NEAREST))