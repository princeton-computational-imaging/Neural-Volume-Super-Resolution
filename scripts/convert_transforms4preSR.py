import os
import json
import numpy as np

SOURCE_DIR = '/scratch/gpfs/yb6751/datasets/Synthetic'
TARGET_DIR = '/tigress/yb6751/projects/NeuralMFSR/results/ScratchSwinIR4preSR'
SCENES = ['mic','ship','chair','lego']

print('Currently not changing anything in the transformation files - just copying them to the target location as is.')
for scene in SCENES:
    for part in ['train','val']:
        with open(os.path.join(SOURCE_DIR,scene, f"transforms_{part}.json"), "r") as fp:
            json_file = json.load(fp)
        # json_file['camera_angle_x'] = 2*np.arctan(2*np.tan(json_file['camera_angle_x']/2))
        with open(os.path.join(TARGET_DIR,scene, f"transforms_{part}.json"), "w") as fp:
            fp.write(json.dumps(json_file))

