from deepdiff import DeepDiff
import sys
sys.path.append('.')
from nerf_helpers import get_config,rgetattr,rsetattr

config_B_file = '/scratch/gpfs/yb6751/projects/VolumetricEnhance/config/planes_E2E.yml'
config_A_file = '/scratch/gpfs/yb6751/projects/VolumetricEnhance/logs/E2E_Synt_Res16Sc200_14Sc800_32_LR100_400_SepValSceneTrain_1/config.yml'

def sort_scenes(config):
    # return config
    for partition in ['train','val']:
        if partition in config.dataset.dir:
            for conf in getattr(config.dataset.dir,partition):
                rsetattr(config.dataset.dir,'.'.join([partition,conf]),sorted([sc for sc in getattr(config.dataset.dir,partition)[conf] if not isinstance(sc,list)]))
    return config

cfg_A = sort_scenes(get_config(config_A_file))
cfg_B = sort_scenes(get_config(config_B_file))

config_diffs = DeepDiff(cfg_A,cfg_B)
ok = True
counter = 0

def diff2attr(diff_string):
    attr = ('.'.join(diff_string.replace("root['","").split("']['"))).replace("']","")
    if '[' in attr:
        attr = attr[:attr.find('[')]
    return attr

diff_attrs = set()
for ch_type in ['dictionary_item_removed','dictionary_item_added','values_changed','type_changes','iterable_item_removed']:
    if ch_type not in config_diffs: continue
    for diff in config_diffs[ch_type]:
        diff_attr = diff2attr(diff)
        if diff_attr in diff_attrs: continue
        diff_attrs.add(diff_attr)
        if ch_type in ['dictionary_item_added','values_changed'] and diff=="root['path']":  continue
        if ch_type=='dictionary_item_removed' and "['use_viewdirs']" in diff:  continue
        elif ch_type=='dictionary_item_added' and diff[:len("root['fine']")]=="root['fine']":  continue
        elif ch_type=='dictionary_item_removed' and "root['fine']" in str(config_diffs[ch_type]):   continue
        diff_string = '\n%d: %s, %s'%(counter,ch_type,diff)
        if ch_type=='dictionary_item_removed':  diff_string += ' %s'%rgetattr(cfg_A,diff_attr)
        elif ch_type=='dictionary_item_added':  diff_string += ' %s'%rgetattr(cfg_B,diff_attr)
        elif ch_type=='values_changed':  diff_string += ' %s -> %s'%(rgetattr(cfg_A,diff_attr),rgetattr(cfg_B,diff_attr))
        elif ch_type=='type_changes':  raise Exception
        elif ch_type=='iterable_item_removed':  raise Exception
        print(diff_string)
        counter += 1
