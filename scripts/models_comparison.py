# This script looks at the differences (in parameter space) between models trained using different image resolutions, 
# compared to the differences between models trained to rperesent different scenes. The incentive for this check was
#  the idea to learn the weights mapping from an LR-trained NeRF to an HR-trained NeRF, and use it for model super-resolution.

from cfgnode import CfgNode
import yaml
import models
import os
from train_utils import find_latest_checkpoint
import torch
import numpy as np

config_file_LR = "/tigress/yb6751/projects/VolumetricEnhance/logs/downsample_mip_0"
config_file_refined = "/tigress/yb6751/projects/VolumetricEnhance/logs/Refined_LR_mip_0"
config_file_LR2 = "/tigress/yb6751/projects/VolumetricEnhance/logs/downsample_mip_chair_0"

paths = [config_file_LR,config_file_refined,config_file_LR2]
coarse_params,fine_params = [],[]
for path in paths:
    with open(os.path.join(path,"config.yml"), "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = CfgNode(cfg_dict)
    checkpoint = find_latest_checkpoint(path)
    checkpoint = torch.load(checkpoint)
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    coarse_params.append(list(model_coarse.named_parameters()))
    model_fine = getattr(models, cfg.models.fine.type)(
        num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
        include_input_xyz=cfg.models.fine.include_input_xyz,
        include_input_dir=cfg.models.fine.include_input_dir,
        use_viewdirs=cfg.models.fine.use_viewdirs,
    )
    model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
    fine_params.append(list(model_fine.named_parameters()))

all_param_diffs = []
compared_models = [[paths[i].split('/')[-1],paths[j].split('/')[-1]] for i in range(len(coarse_params)) for j in range(i+1,len(coarse_params))]
print('\n'.join(['\t'.join([m_name[i] for m_name in compared_models]) for i in range(len(compared_models[0]))]))
for p_num in range(len(coarse_params[0])):
    params = torch.stack([coarse_params[i][p_num][1] for i in range(len(coarse_params))],0)
    diff_STDs = ((params.unsqueeze(0)-params.unsqueeze(1)).abs()).reshape([len(coarse_params),len(coarse_params),-1]).mean(2).detach().numpy()
    diff_STDs = [diff_STDs[i,j] for i in range(len(coarse_params)) for j in range(i+1,len(coarse_params))]
    all_param_diffs.append(diff_STDs)
    print('\t'.join([str(d) for d in diff_STDs]),'\t',"%d, %s "%(p_num,coarse_params[0][p_num][0]),'\t',coarse_params[0][p_num][1].shape)
print("Mean: ",'\t'.join([str(d) for d in np.stack(all_param_diffs,0).mean(0)]))