import os
import sys
from unicodedata import name
sys.path.append('.')
import load_DTU

scenes_file = '/scratch/gpfs/yb6751/datasets/rs_dtu_4/DTU/all_train.lst'


def name_mapping():
    base_dir = os.path.dirname(scenes_file)
    cat = os.path.basename(base_dir)

    with open(scenes_file, "r") as f:
        objs = sorted([(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()])

    return dict(zip([load_DTU.old_DTU_sceneID(i) for i in range(len(objs))],[o[1].split('/')[-1] for o in objs]))


if __name__ == "__main__":
    name_mapping()
