import os
from re import search
from glob import glob
from shutil import copyfile

DATASET_PATH = '/scratch/gpfs/yb6751/datasets/EDSR_SR_Synthetic_trainedOn259'
PATTERN = 'r_(\d)+_scratch_upscaleX4\.png'
for scene in os.listdir(DATASET_PATH):
    counter =0 
    if not os.path.isdir(os.path.join(DATASET_PATH,scene)): continue
    if os.path.isdir(os.path.join(DATASET_PATH,scene,'all_train')): continue
    os.rename(os.path.join(DATASET_PATH,scene,'train'),os.path.join(DATASET_PATH,scene,'all_train'))
    os.mkdir(os.path.join(DATASET_PATH,scene,'train'))
    for f in glob(os.path.join(DATASET_PATH,scene,'all_train','*.png')):
        if search(PATTERN,f) is not None:
            copyfile(f,f.replace('all_train','train').replace('_scratch_upscaleX4',''))
            counter += 1
    print("Converted %d files for scene %s"%(counter,scene))


# Renaming the horns scene image names:
org_names = glob('/scratch/gpfs/yb6751/datasets/LLFF/horns/images_4/*.png')
org_names = sorted([name for name in org_names],key=lambda x: int(search('(?<=DJI_20200223_)(\d)+(?=_(\d)+\.png$)',x).group(0)))
for im_num,im in enumerate(org_names):
    new_name = im.split('/')
    new_name[-1] = 'image%03d.png'%(im_num)
    new_name = '/'.join(new_name)
    os.rename(im,new_name)