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
