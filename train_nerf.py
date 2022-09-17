import argparse
import glob
import os
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
# import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import models
# from cfgnode import CfgNode
from load_blender import load_blender_data
from load_DTU import DVRDataset
from nerf_helpers import * 
from train_utils import eval_nerf, run_one_iter_of_nerf,find_latest_checkpoint
from mip import IntegratedPositionalEncoding
from deepdiff import DeepDiff
from copy import deepcopy

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default='',
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to load config file to resume.",
    )
    configargs = parser.parse_args()

    # Read config file.
    assert (configargs.config is None) ^ (configargs.resume is None)
    cfg = None
    if configargs.config is None:
        # assert os.path.isdir(configargs.resume)
        config_file = os.path.join(configargs.resume,"config.yml")
    else:
        config_file = configargs.config
    print('Using configuration file %s'%(config_file))
    cfg = get_config(config_file)
    print("Running experiment %s"%(cfg.experiment.id))
    SR_experiment = None
    if "super_resolution" in cfg:
        SR_experiment = "model" if "model" in cfg.super_resolution.keys() else "refine"
    planes_model = cfg.models.coarse.type=="TwoDimPlanesModel"
    downsampling_factor = cfg.dataset.get("downsampling_factor",[1])
    downsampling_factor = assert_list(downsampling_factor)

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    print('Saving logs and models into %s'%(logdir))
    if configargs.load_checkpoint=="resume":
        configargs.load_checkpoint = logdir
    else:
        if configargs.load_checkpoint=='':
            if os.path.exists(logdir):  assert len([f for f in os.listdir(logdir) if '.ckpt' in f])==0,'Folder %s already contains saved models.'%(logdir)
            os.makedirs(logdir, exist_ok=True)
            # os.makedirs(logdir)
        # Write out config parameters.
        with open(os.path.join(logdir, "config.yml"), "w") as f:
            f.write(cfg.dump())  # cfg, f, default_flow_style=False)
    if configargs.load_checkpoint!='':
        assert os.path.exists(configargs.load_checkpoint)
    writer = SummaryWriter(logdir)
    load_saved_models = SR_experiment or os.path.exists(configargs.load_checkpoint)

    if planes_model:
        plane_resolutions = assert_list(getattr(cfg.models.coarse,'plane_resolutions',[512]))
        viewdir_plane_resolution = assert_list(getattr(cfg.models.coarse,'viewdir_plane_resolution',plane_resolutions))
        plane_resolutions = [(plane_resolutions[i],viewdir_plane_resolution[i]) for i in range(len(plane_resolutions))]
        # if not isinstance(plane_resolutions,list):  plane_resolutions = [plane_resolutions]
        assert len(downsampling_factor)==1 or len(downsampling_factor)==len(plane_resolutions)
    internal_SR,record_fine = False,True
    if SR_experiment:
        assert not hasattr(cfg.dataset,"downsampling_factor")
        LR_model_folder = cfg.models.path
        if os.path.isfile(LR_model_folder):   LR_model_folder = "/".join(LR_model_folder.split("/")[:-1])
        LR_model_ds_factor = get_config(os.path.join(LR_model_folder,"config.yml")).dataset.downsampling_factor
        # downsampling_factor = 
        internal_SR = isinstance(LR_model_ds_factor,list) and len(LR_model_ds_factor)>1
        # internal_SR = isinstance(cfg.super_resolution.ds_factor,list) and len(cfg.super_resolution.ds_factor)>1
        if internal_SR:
            assert not isinstance(cfg.dataset.dir,list) or len(cfg.dataset.dir)==1
            cfg.super_resolution.ds_factor = max(LR_model_ds_factor)//min(LR_model_ds_factor)
            downsampling_factor = [cfg.super_resolution.ds_factor] # In this case the SR model should train to reconstruct images downsampled using the same factor as the SR factor. 
        else:
            sf_config = getattr(cfg.super_resolution.model,'scale_factor','linear')
            assert sf_config in ['linear','sqrt','one']
            if sf_config=='one':
                cfg.super_resolution.ds_factor = 1
            elif sf_config=='linear':
                cfg.super_resolution.ds_factor = LR_model_ds_factor
            else:
                cfg.super_resolution.ds_factor = int(np.sqrt(LR_model_ds_factor))
            # downsampling_factor = assert_list(downsampling_factor)
        if SR_experiment=="model":  consistent_SR_density = cfg.super_resolution.model.get("consistent_density",False)
    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        dataset_type = getattr(cfg.dataset,'type','synt')
        assert dataset_type in['synt','DTU']
        if dataset_type=='synt':
            train_dirs = assert_list(cfg.dataset.dir.train)
            val_only_dirs = assert_list(getattr(cfg.dataset.dir,'val',[]))
            # basedirs = [os.path.join(cfg.dataset.root,d) for d in train_dirs+val_only_dirs]
            basedirs = train_dirs+val_only_dirs
            images, poses, render_poses, hwfDs, i_split,scene_ids = [],torch.zeros([0,4,4]),[],[[],[],[],[]],[np.array([]).astype(np.int64) for i in range(3)],[]
            scene_id,scene_id_plane_resolution,val_only_scene_ids,coords_normalization = -1,{},[],{}
            i_train,i_val = OrderedDict(),OrderedDict()
            if planes_model and internal_SR:
                scene_id_plane_resolution = {0:max(plane_resolutions),1:min(plane_resolutions)}
            for basedir in tqdm(basedirs):
                for ds_num,factor in enumerate(downsampling_factor):
                    # scene_id += 1
                    scene_id = ''+basedir
                    val_only = basedir not in train_dirs
                    if val_only:    val_only_scene_ids.append(scene_id)
                    if planes_model and not internal_SR:
                        scene_id_plane_resolution[scene_id] = plane_resolutions[min(len(plane_resolutions)-1,ds_num)]
                    cur_images, cur_poses, cur_render_poses, cur_hwfDs, cur_i_split = load_blender_data(
                        os.path.join(cfg.dataset.root,basedir),
                        half_res=getattr(cfg.dataset,'half_res',False),
                        testskip=cfg.dataset.testskip,
                        downsampling_factor=cfg.super_resolution.ds_factor if internal_SR else factor,
                        val_downsampling_factor=1 if internal_SR else None,
                        cfg=cfg,
                        val_only=val_only,
                    )
                    if planes_model and not load_saved_models: # No need to calculate the per-scene normalization coefficients as those will be loaded with the saved model.
                        coords_normalization[basedir] =\
                            calc_scene_box({'camera_poses':cur_poses.numpy()[:,:3,:4],'near':cfg.dataset.near,'far':cfg.dataset.far,'H':cur_hwfDs[0],'W':cur_hwfDs[1],'f':cur_hwfDs[2]},including_dirs=cfg.nerf.use_viewdirs)
                    i_val[scene_id] = [v+len(images) for v in cur_i_split[1]]
                    i_train[scene_id] = [v+len(images) for v in cur_i_split[0]]
                    # for i in range(len(i_split)):
                    #     i_split[i] = np.concatenate((i_split[i],cur_i_split[i]+len(images)))
                    images += cur_images
                    poses = torch.cat((poses,cur_poses),0)
                    for i in range(len(hwfDs)):
                        hwfDs[i] += cur_hwfDs[i]
                    scene_ids += [scene_id for i in cur_images]
            if internal_SR: # Speicifically handling this case, where there is only one scene, and we are currently 
                # training our SR model on downsampled training images, while evaluating it on full size 
                # validation images.
                assert scene_id==0
                scene_ids = [0 if i in i_split[1] else 1 for i in range(len(scene_ids))]
            H, W, focal,ds_factor = hwfDs
        else:
            dataset = DVRDataset(path=cfg.dataset.root,stage='train_val',eval_ratio=0.1,\
                list_prefix='new_' if SR_experiment else 'all_',max_scenes=8,
                z_near=cfg.dataset.near,z_far=cfg.dataset.far)
            val_only_scene_ids = dataset.val_scene_IDs()
            scene_ids = dataset.scene_IDs()
            total_scenes_num = dataset.num_scenes() #+dataset_eval.num_scenes()
            assert len(plane_resolutions)==1

            scene_id_plane_resolution = dict(zip([dataset.DTU_sceneID(i) for i in range(total_scenes_num)],[plane_resolutions[0] for i in range(total_scenes_num)]))
            basedirs,coords_normalization = [],{}
            for id in trange(dataset.num_scenes()):
                scene_info = dataset.scene_info(id)
                scene_info.update({'near':dataset.z_near,'far':dataset.z_far})
                coords_normalization[dataset.DTU_sceneID(id)] = calc_scene_box(scene_info,including_dirs=cfg.nerf.use_viewdirs)
                basedirs.append(dataset.DTU_sceneID(id))
            i_val = dataset.i_val
            # assert all([len(i_val[basedirs[0]])==len(i_val[id]) for id in basedirs]),'Assuming all scenes have the same number of evaluation images'
            i_train = dataset.train_ims_per_scene
        # assert all([len(i_val[basedirs[0]])==len(i_val[id]) for id in basedirs]),'Assuming all scenes have the same number of evaluation images'

        EVAL_TRAINING_TOO = True
        if EVAL_TRAINING_TOO:
            val_ims_per_scene = len(i_val[basedirs[0]])
            for id in basedirs:
                if id in val_only_scene_ids:    continue
                im_freq = len(i_train[id])//val_ims_per_scene
                # i_val[id+'_train'] = [i for i,x in enumerate(scene_ids) if x==id and i in i_train][:1]
                i_val[id+'_train'] = [x for i,x in enumerate(i_train[id]) if (i+im_freq//2)%im_freq==0]
        assert all([len(i_val[basedirs[0]])==len(i_val[id]) for id in i_val.keys()]),'Assuming all scenes have the same number of evaluation images'
        i_train = [i for s in i_train.values() for i in s]

        SR_HR_im_inds,val_ims_dict = None,None
    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if getattr(cfg.nerf,"encode_position_fn",None) is not None:
        assert cfg.nerf.encode_position_fn in ["mip","positional_encoding"]
        if cfg.nerf.encode_position_fn=="mip":
            mip_encoder = IntegratedPositionalEncoding(input_dims=3,\
                multires=cfg.models.coarse.num_encoding_fn_xyz+1,include_input=cfg.models.coarse.include_input_xyz)
            def encode_position_fn(x):
                return mip_encoder(x)
        else:
            def encode_position_fn(x):
                return positional_encoding(
                    x,
                    num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
                    include_input=cfg.models.coarse.include_input_xyz,
                )

        def encode_direction_fn(x):
            return positional_encoding(
                x,
                num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
                include_input=cfg.models.coarse.include_input_dir,
            )
    else:
        encode_position_fn = None
        encode_direction_fn = None
    # if getattr(cfg.models.coarse,"use_viewdirs",False) or getattr(cfg.models.fine,"use_viewdirs",False):    assert cfg.nerf.use_viewdirs
    if planes_model:
        if hasattr(cfg.nerf.train,'viewdir_downsampling'):  assert hasattr(cfg.nerf.train,'max_plane_downsampling')
        model_coarse = models.TwoDimPlanesModel(
            use_viewdirs=cfg.nerf.use_viewdirs,
            scene_id_plane_resolution=scene_id_plane_resolution,
            coords_normalization = coords_normalization,
            dec_density_layers=getattr(cfg.models.coarse,'dec_density_layers',4),
            dec_rgb_layers=getattr(cfg.models.coarse,'dec_rgb_layers',4),
            dec_channels=getattr(cfg.models.coarse,'dec_channels',128),
            num_plane_channels=getattr(cfg.models.coarse,'num_plane_channels',48),
            rgb_dec_input=getattr(cfg.models.coarse,'rgb_dec_input','projections'),
            proj_combination=getattr(cfg.models.coarse,'proj_combination','sum'),
            plane_interp=getattr(cfg.models.coarse,'plane_interp','bilinear'),
            align_corners=getattr(cfg.models.coarse,'align_corners',True),
            interp_viewdirs=getattr(cfg.models.coarse,'interp_viewdirs',None),
            viewdir_downsampling=getattr(cfg.nerf.train,'viewdir_downsampling',True),
        )
        
    else:
        # Initialize a coarse-resolution model.
        assert not (cfg.nerf.encode_position_fn=="mip" and cfg.models.coarse.include_input_xyz),"Mip-NeRF does not use the input xyz"
        assert cfg.models.coarse.include_input_xyz==cfg.models.fine.include_input_xyz,"Assuming they are the same"
        assert cfg.models.coarse.include_input_dir==cfg.models.fine.include_input_dir,"Assuming they are the same"
        model_coarse = getattr(models, cfg.models.coarse.type)(
            num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
            include_input_xyz=cfg.models.coarse.include_input_xyz,
            include_input_dir=cfg.models.coarse.include_input_dir,
            use_viewdirs=cfg.nerf.use_viewdirs,
        )

    print("Coarse model: %d parameters"%num_parameters(model_coarse))
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        if cfg.models.fine.type=="use_same":
            model_fine = model_coarse
            print("Using the same model for coarse and fine")
        else:
            if planes_model:
                for k in cfg.models.coarse.keys():
                    if k not in cfg.models.fine:
                        setattr(cfg.models.fine,k,getattr(cfg.models.coarse,k))
                model_fine = models.TwoDimPlanesModel(
                    use_viewdirs=cfg.nerf.use_viewdirs,
                    scene_id_plane_resolution=scene_id_plane_resolution,
                    coords_normalization = coords_normalization,
                    dec_density_layers=getattr(cfg.models.fine,'dec_density_layers',4),
                    dec_rgb_layers=getattr(cfg.models.fine,'dec_rgb_layers',4),
                    dec_channels=getattr(cfg.models.fine,'dec_channels',128),
                    num_plane_channels=getattr(cfg.models.fine,'num_plane_channels',48),
                    rgb_dec_input=getattr(cfg.models.fine,'rgb_dec_input','projections'),
                    proj_combination=getattr(cfg.models.fine,'proj_combination','sum'),
                    planes=model_coarse.planes_ if getattr(cfg.models.fine,'use_coarse_planes',False) else None,
                    plane_interp=getattr(cfg.models.fine,'plane_interp','bilinear'),
                    align_corners=getattr(cfg.models.fine,'align_corners',True),
                    interp_viewdirs=getattr(cfg.models.fine,'interp_viewdirs',None),
                    viewdir_downsampling=getattr(cfg.nerf.train,'viewdir_downsampling',True),
                )
            else:
                model_fine = getattr(models, cfg.models.fine.type)(
                    num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
                    num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
                    include_input_xyz=cfg.models.fine.include_input_xyz,
                    include_input_dir=cfg.models.fine.include_input_dir,
                    use_viewdirs=cfg.nerf.use_viewdirs,
                )
            print("Fine model: %d parameters"%num_parameters(model_fine))
            model_fine.to(device)

    if SR_experiment=="model":
        if cfg.super_resolution.model.type=='EDSR':
            plane_channels = list(model_fine.planes_.values())[0].shape[1]
            SR_model = getattr(models, cfg.super_resolution.model.type)(
                # scale_factor=cfg.super_resolution.model.scale_factor,
                scale_factor=cfg.super_resolution.ds_factor,
                in_channels=plane_channels,
                out_channels=plane_channels,
                hidden_size=cfg.super_resolution.model.hidden_size,
                plane_interp=model_fine.plane_interp,
                n_blocks=cfg.super_resolution.model.n_blocks,
            )
            print("SR model: %d parameters"%(num_parameters(SR_model)))
        else:
            NUM_FUNC_TYPES,NUM_COORDS,NUM_MODEL_OUTPUTS = 2,3,4
            assert cfg.super_resolution.model.input in ["outputs","dirs_encoding","xyz_encoding"]
            if cfg.super_resolution.model.input=="outputs":
                encoding_grad_inputs = 2*[0]
            elif cfg.super_resolution.model.input=="xyz_encoding":
                encoding_grad_inputs = [cfg.models.fine.num_encoding_fn_xyz,0]
                # assert not cfg.super_resolution.model.get("xyz_input_2_dir",False),"Not taking view-directions as input, so no sense of adding xyz to them"
            elif cfg.super_resolution.model.input=="dirs_encoding":
                encoding_grad_inputs = [cfg.models.fine.num_encoding_fn_xyz,cfg.models.fine.num_encoding_fn_dir]
            if consistent_SR_density:
                SR_input_dim = [NUM_FUNC_TYPES*d for d in encoding_grad_inputs]
                if SR_input_dim[0]>0:   SR_input_dim[0] = NUM_COORDS*(SR_input_dim[0]+cfg.models.coarse.include_input_xyz)
                if SR_input_dim[1]>0:   SR_input_dim[1] = NUM_COORDS*(SR_input_dim[1]+cfg.models.coarse.include_input_dir)
                # SR_input_dim = [(d+1)*NUM_COORDS if d>0 else 0 for d in SR_input_dim]
                jacobian_numel = NUM_MODEL_OUTPUTS*sum(SR_input_dim)
                SR_input_dim[0] += 1
                SR_input_dim[1] = jacobian_numel+NUM_MODEL_OUTPUTS-SR_input_dim[0]
            else:
                raise Exception("This computation is wrong, and reaches a lower input dimension than the actual product of the dimension of the input to the NeRF model times number of its outputs, which is the size of the Jacobian, plus the number of outputs. The reason why it worked is because I removed the excessive input channels as the first step when running the SR model.")
                SR_input_dim = NUM_FUNC_TYPES*sum(encoding_grad_inputs)
                if SR_input_dim>0:  
                    SR_input_dim += 1
                    SR_input_dim *= NUM_COORDS*NUM_MODEL_OUTPUTS
                SR_input_dim += NUM_MODEL_OUTPUTS
                SR_input_dim = [SR_input_dim,0]
            # SR_input_dim += NUM_MODEL_OUTPUTS
            if cfg.super_resolution.model.type=="Conv3D":   assert cfg.super_resolution.model.input=="outputs","Currently not supporting gradients input to spatial SR model."
            SR_model = getattr(models, cfg.super_resolution.model.type)(
                input_dim= SR_input_dim,
                use_viewdirs=consistent_SR_density,
                num_layers=cfg.super_resolution.model.num_layers_xyz,
                num_layers_dir=cfg.super_resolution.model.get("num_layers_dir",1),
                hidden_size=cfg.super_resolution.model.hidden_size,
                dirs_hidden_width_ratio=1,
                xyz_input_2_dir=cfg.super_resolution.model.get("xyz_input_2_dir",False)
            )
            print("SR model: %d parameters, input dimension xyz: %d, dirs: %d"%\
                (num_parameters(SR_model),SR_input_dim[0],SR_input_dim[1]))
        SR_model.to(device)
        trainable_parameters = list(SR_model.parameters())
    else:
        # Initialize optimizer.
        trainable_parameters = list(model_coarse.parameters())
        if model_fine is not None:
            if planes_model:
                if cfg.models.fine.type!="use_same":
                    if getattr(cfg.models.fine,'use_coarse_planes',False):
                        trainable_parameters += [v for k,v in model_fine.named_parameters() if 'planes_.' not in k]
                    else:
                        trainable_parameters += list(model_fine.parameters())
            else: 
                trainable_parameters += list(model_fine.parameters())
        SR_model = None
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # Load an existing checkpoint, if a path is specified.
    start_i = 0
    if load_saved_models:
        if SR_experiment:
            saved_rgb_fine = [{} for i in i_val.keys()]
            checkpoint = find_latest_checkpoint(cfg.models.path)
            print("Using LR model %s"%(checkpoint))
            if SR_experiment=="model" and os.path.exists(configargs.load_checkpoint):
                assert os.path.isdir(configargs.load_checkpoint)
                SR_model_checkpoint = os.path.join(configargs.load_checkpoint,sorted([f for f in os.listdir(configargs.load_checkpoint) if "checkpoint" in f and f[-5:]==".ckpt"],key=lambda x:int(x[len("checkpoint"):-5]))[-1])
                start_i = int(SR_model_checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
                print("Resuming training on model %s"%(SR_model_checkpoint))
                saved_config_dict = get_config(os.path.join(configargs.load_checkpoint,"config.yml"))
                config_diffs = DeepDiff(saved_config_dict,cfg)
                for diff in [config_diffs[ch_type] for ch_type in ['dictionary_item_removed','dictionary_item_added','values_changed'] if ch_type in config_diffs]:
                    print(diff)

                SR_model.load_state_dict(torch.load(SR_model_checkpoint)["SR_model"])
        else:
            checkpoint = find_latest_checkpoint(configargs.load_checkpoint)
            start_i = int(checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
            print("Resuming training on model %s"%(checkpoint))
        checkpoint_config = get_config(os.path.join('/'.join(checkpoint.split('/')[:-1]),'config.yml'))
        config_diffs = DeepDiff(checkpoint_config.models,cfg.models)
        if SR_experiment:   assert getattr(cfg.dataset,'half_res',False)==getattr(checkpoint_config.dataset,'half_res',False),'Unsupported "half-res" mismatch'
        ok = True
        for ch_type in ['dictionary_item_removed','dictionary_item_added','values_changed','type_changes']:
            if ch_type not in config_diffs: continue
            for diff in config_diffs[ch_type]:
                if ch_type=='dictionary_item_added' and diff=="root['path']":  continue
                if ch_type=='dictionary_item_removed' and "['use_viewdirs']" in diff:  continue
                elif ch_type=='dictionary_item_added' and diff[:len("root['fine']")]=="root['fine']":  continue
                elif ch_type=='dictionary_item_removed' and "root['fine']" in str(config_diffs[ch_type]):   continue
                print(ch_type,diff)
                ok = False
        if not ok:  raise Exception('Inconsistent model config')
        checkpoint = torch.load(checkpoint)
        def update_saved_names(state_dict):
            # if any(['planes_.sc0' in k for k in state_dict.keys()]):
            #     assert len([k for k in state_dict if 'planes_.sc' in k])==len([k for k in model_coarse.state_dict() if 'planes_.sc' in k])
            return state_dict
            if any(['planes_.' in k and '.sc' not in k for k in state_dict.keys()]):
                return OrderedDict([(k.replace('planes_.','planes_.sc0_res32_D'),v) for k,v in state_dict.items()])
            else:
                return OrderedDict([(k.replace('planes.','planes_.sc0_res32_D'),v) for k,v in state_dict.items()])

        if False:
            from re import search
            def rep_name(name):
                ind = search('(?<=.sc)(\d)+(?=_D)',name).group(0)
                return name.replace('.sc'+ind,'.sc'+basedirs[int(ind)])
            checkpoint["model_coarse_state_dict"] = OrderedDict([(rep_name(k),v) if '.sc' in k else (k,v) for k,v in checkpoint["model_coarse_state_dict"].items()])
            checkpoint["model_fine_state_dict"] = OrderedDict([(rep_name(k),v) if '.sc' in k else (k,v) for k,v in checkpoint["model_fine_state_dict"].items()])
            torch.save(checkpoint,find_latest_checkpoint(cfg.models.path))

        def load_saved_parameters(model,saved_params):
            mismatch = model.load_state_dict(saved_params,strict=False)
            assert len(mismatch.missing_keys)==0 and all(['planes_.sc' in k for k in mismatch.unexpected_keys])
            if planes_model:
                model.box_coords = checkpoint["coords_normalization"]


        load_saved_parameters(model_coarse,checkpoint["model_coarse_state_dict"])
        # mismatch = model_coarse.load_state_dict(update_saved_names(checkpoint["model_coarse_state_dict"]),strict=False)
        if checkpoint["model_fine_state_dict"]:
            load_saved_parameters(model_fine,checkpoint["model_fine_state_dict"])
            # model_fine.load_state_dict(update_saved_names(checkpoint["model_fine_state_dict"]))
        if SR_experiment=="refine":
            LR_model_coarse = deepcopy(model_coarse)
            LR_model_fine = deepcopy(model_fine)
            furthest_rgb_fine,closest_rgb_fine, = [{} for i in train_dirs+val_only_dirs],[{} for i in train_dirs+val_only_dirs],
            if os.path.exists(configargs.load_checkpoint):
                checkpoint = find_latest_checkpoint(configargs.load_checkpoint)
                start_i = int(checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
                print("Resuming training on model %s"%(checkpoint))
                checkpoint = torch.load(checkpoint)
                model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
                model_fine.load_state_dict(checkpoint["model_fine_state_dict"])

        del checkpoint # Importent: Releases GPU memory occupied by loaded data.
    # # TODO: Prepare raybatch tensor if batching random rays
    spatial_padding_size = SR_model.receptive_field//2 if isinstance(SR_model,models.Conv3D) else 0
    spatial_sampling = spatial_padding_size>0 or cfg.nerf.train.get("spatial_sampling",False)

    if planes_model and SR_model is not None:
        if getattr(cfg.super_resolution,'apply_2_coarse',False):
            model_coarse.assign_SR_model(SR_model,SR_viewdir=cfg.super_resolution.SR_viewdir)
        else:
            assert getattr(cfg.super_resolution.training,'loss','both')=='fine'
        model_fine.assign_SR_model(SR_model,SR_viewdir=cfg.super_resolution.SR_viewdir)

    # downsampling_offset = lambda ds_factor: np.floor((ds_factor-1)/2)/ds_factor
    downsampling_offset = lambda ds_factor: (ds_factor-1)/(2*ds_factor)
    # if SR_experiment and '/planes_Res200Lrgb4Lden4_LR4_allScenes_0' in cfg.models.path:
    #     print('WARNING: Applying a temprary patch...')
    #     CORRECTION_OFFSET = 1
    #     downsampling_offset = lambda dummy: -CORRECTION_OFFSET
    #     images = [torch.nn.functional.pad(im[CORRECTION_OFFSET:,CORRECTION_OFFSET:,:],(0,0,0,CORRECTION_OFFSET,0,CORRECTION_OFFSET),mode='constant') for im in images]
    # if any([v/2==v//2  for v in ds_factor]):    print('*** WARNING: Using an even downsampling factor, which causes misalignment with the full scale image ***')
    # assert [v in [1,4] for v in ds_factor],'The above function (using floor) was only checked for 4 (and 1). Should verify that is the case for other factors too'

    for iter in trange(start_i,cfg.experiment.train_iters):
        # Validation
        if (
            iter % cfg.experiment.validate_every == 0
            or iter == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(iter))
            # if dataset_type=='DTU': dataset.eval(True)
            model_coarse.eval()
            if model_fine:
                model_fine.eval()
            if SR_experiment=="model":
                SR_model.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, _ = eval_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    if True:
                        val_ind = lambda dummy: (iter//cfg.experiment.validate_every)%len(i_val[basedirs[0]])
                    else: #For the previous case of having only a single training image to evaluate on:
                        val_ind = lambda set_id: (iter//cfg.experiment.validate_every)%len(i_val[basedirs[0]]) if 'validation' in set_id else 0
                    val_strings = ['blind_validation' if id in val_only_scene_ids else 'train_imgs' if '_train' in id else 'validation' for id in i_val.keys()]
                    img_indecis = [v[val_ind(val_strings[i])] for i,v in enumerate(i_val.values())]
                    if val_ims_dict is not None:
                        raise Exception('Revisit after enabling multi-scene.')
                        img_indecis += [i_val[val_ims_dict["closest_val"]],i_val[val_ims_dict["furthest_val"]]]
                        val_strings += ["closest_","furthest_"]
                    # img_idx = 0

                    coarse_loss,fine_loss,loss,psnr,rgb_coarse,rgb_fine,rgb_SR,target_ray_values = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
                    for scene_num,img_idx in enumerate(img_indecis):
                        if dataset_type=='synt':
                            img_target = images[img_idx].to(device)
                            pose_target = poses[img_idx, :3, :4].to(device)
                            cur_H,cur_W,cur_focal,cur_ds_factor = H[img_idx], W[img_idx], focal[img_idx],ds_factor[img_idx]
                        else:
                            img_target,pose_target,cur_H,cur_W,cur_focal = dataset.item(img_idx,device)
                        ray_origins, ray_directions = get_ray_bundle(
                            cur_H, cur_W, cur_focal, pose_target,padding_size=spatial_padding_size,downsampling_offset=downsampling_offset(cur_ds_factor),
                        )
                        rgb_coarse_, _, _, rgb_fine_, _, _,rgb_SR_,_,_ = eval_nerf(
                            cur_H,
                            cur_W,
                            cur_focal,
                            model_coarse,
                            model_fine,
                            ray_origins,
                            ray_directions,
                            cfg,
                            mode="validation",
                            encode_position_fn=encode_position_fn,
                            encode_direction_fn=encode_direction_fn,
                            SR_model=SR_model,
                            ds_factor_or_id=scene_ids[img_idx] if planes_model else ds_factor[img_idx],
                            spatial_margin=SR_model.receptive_field//2 if spatial_sampling else None
                        )
                        target_ray_values[val_strings[scene_num]].append(img_target[...,:3])
                        if SR_experiment:
                            if SR_experiment=="refine" or planes_model:
                                rgb_SR_ = 1*rgb_fine_
                                rgb_SR_coarse_ = 1*rgb_coarse_
                                if val_ind(val_strings[scene_num]) not in saved_rgb_fine[scene_num]:
                                    record_fine = True
                                    model_coarse.skip_SR(True)
                                    model_fine.skip_SR(True)
                                    rgb_coarse_, _, _, rgb_fine_, _, _,_,_,_ = eval_nerf(
                                        cur_H,
                                        cur_W,
                                        cur_focal,
                                        model_coarse if planes_model else LR_model_coarse,
                                        model_fine if planes_model else LR_model_fine,
                                        ray_origins,
                                        ray_directions,
                                        cfg,
                                        mode="validation",
                                        encode_position_fn=encode_position_fn,
                                        encode_direction_fn=encode_direction_fn,
                                        SR_model=SR_model,
                                        ds_factor_or_id=scene_ids[img_idx] if planes_model else ds_factor[img_idx],
                                    )
                                    model_coarse.skip_SR(False)
                                    model_fine.skip_SR(False)
                                    saved_rgb_fine[scene_num][val_ind(val_strings[scene_num])] = 1*rgb_fine_.detach()
                                else:
                                    record_fine = False
                                    rgb_fine_ = 1*saved_rgb_fine[scene_num][val_ind(val_strings[scene_num])]
                            fine_loss[val_strings[scene_num]].append(img2mse(rgb_fine_[..., :3], img_target[..., :3]).item())
                            loss[val_strings[scene_num]].append(img2mse(rgb_SR_[..., :3], img_target[..., :3]).item())
                        else:
                            coarse_loss[val_strings[scene_num]].append(img2mse(rgb_coarse_[..., :3], img_target[..., :3]).item())
                            fine_loss[val_strings[scene_num]].append(0.0)
                            if rgb_fine is not None:
                                fine_loss[val_strings[scene_num]][-1] = img2mse(rgb_fine_[..., :3], img_target[..., :3]).item()
                            loss[val_strings[scene_num]].append(coarse_loss[val_strings[scene_num]][-1] + fine_loss[val_strings[scene_num]][-1])
                        rgb_coarse[val_strings[scene_num]].append(rgb_coarse_)
                        rgb_fine[val_strings[scene_num]].append(rgb_fine_)
                        rgb_SR[val_strings[scene_num]].append(rgb_SR_)
                        psnr[val_strings[scene_num]].append(mse2psnr(loss[val_strings[scene_num]][-1]))
                    SAVE_COARSE_IMAGES = False
                    for val_set in set(val_strings):
                        font_scale = 4/downsampling_factor[0]
                        if SR_experiment:
                            writer.add_scalar("%s/SR_psnr_gain"%(val_set), np.mean([psnr[val_set][i]-mse2psnr(fine_loss[val_set][i]) for i in range(len(psnr[val_set]))]), iter)
                            writer.add_image(
                                "%s/rgb_SR"%(val_set), arange_ims(rgb_SR[val_set],str(val_ind(val_set)),psnrs=psnr[val_set],fontScale=font_scale),iter
                            )
                        writer.add_scalar("%s/fine_loss"%(val_set), np.mean(fine_loss[val_set]), iter)
                        writer.add_scalar("%s/fine_psnr"%(val_set), np.mean([mse2psnr(l) for l in fine_loss[val_set]]), iter)
                        writer.add_scalar("%s/loss"%(val_set), np.mean(loss[val_set]), iter)
                        writer.add_scalar("%s/psnr"%(val_set), np.mean(psnr[val_set]), iter)
                        writer.add_scalar("%s/coarse_loss"%(val_set), np.mean(coarse_loss[val_set]), iter)
                        if len(rgb_fine[val_set])>0:
                            if record_fine:
                                writer.add_scalar("%s/fine_loss"%(val_set), np.mean(fine_loss[val_set]), val_ind(val_set) if SR_experiment else iter)
                                writer.add_image("%s/rgb_fine"%(val_set), arange_ims(rgb_fine[val_set],str(val_ind(val_set)),psnrs=[mse2psnr(l) for l in fine_loss[val_set]],fontScale=font_scale),val_ind(val_set) if SR_experiment else iter)
                        if SAVE_COARSE_IMAGES:
                            writer.add_image(
                                "%s/rgb_coarse"%(val_set), arange_ims(rgb_coarse[val_set],str(val_ind(val_set)),psnrs=[mse2psnr(l) for l in coarse_loss[val_set]],fontScale=font_scale),iter
                            )
                        writer.add_image(
                            "%s/img_target"%(val_set), arange_ims(target_ray_values[val_set],str(val_ind(val_set)),fontScale=font_scale),iter
                        )
                        tqdm.write(
                            "%s Validation loss: "%(val_set)
                            + str(np.mean(loss[val_set]))
                            + "%s Validation PSNR: "%(val_set)
                            + str(np.mean(psnr[val_set]))
                            + "Time: "
                            + str(time.time() - start)
                        )
            
        # Training:
        # if dataset_type=='DTU': dataset.eval(False)
        if SR_experiment=="model":
            SR_model.train()
        else:
            model_coarse.train()
            if model_fine:
                model_fine.train()
            if getattr(cfg.nerf.train,'max_plane_downsampling',1)>1:
                plane_downsampling = 1 if np.random.uniform()>0.5 else 1+np.random.randint(cfg.nerf.train.max_plane_downsampling)
                model_coarse.use_downsampled_planes(plane_downsampling)
                model_fine.use_downsampled_planes(plane_downsampling)

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"]
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_bundle,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                spatial_margin=SR_model.receptive_field//2 if spatial_sampling else None
            )
        else:
            if SR_HR_im_inds is None:
                img_idx = np.random.choice(i_train)
            else:
                if np.random.uniform()<cfg.super_resolution.training.LR_ims_chance:
                    img_idx = np.random.choice(SR_LR_im_inds)
                else:
                    img_idx = np.random.choice(SR_HR_im_inds)

            if dataset_type=='synt':
                img_target = images[img_idx].to(device)
                pose_target = poses[img_idx, :3, :4].to(device)
                cur_H,cur_W,cur_focal,cur_ds_factor = H[img_idx], W[img_idx], focal[img_idx],ds_factor[img_idx]
            else:
                img_target,pose_target,cur_H,cur_W,cur_focal = dataset.item(img_idx,device)

            ray_origins, ray_directions = get_ray_bundle(cur_H, cur_W, cur_focal, pose_target,padding_size=spatial_padding_size,downsampling_offset=downsampling_offset(cur_ds_factor),)
            coords = torch.stack(
                meshgrid_xy(torch.arange(cur_H+2*spatial_padding_size).to(device), torch.arange(cur_W+2*spatial_padding_size).to(device)),
                dim=-1,
            )
            if spatial_padding_size>0 or spatial_sampling:
                patch_size = chunksize_to_2D(cfg.nerf.train.num_random_rays)
                upper_left_corner = np.random.uniform(size=[2])*(np.array([cur_H,cur_W])-patch_size)
                upper_left_corner = np.floor(upper_left_corner).astype(np.int32)
                select_inds = \
                    coords[upper_left_corner[0]:upper_left_corner[0]+patch_size+2*spatial_padding_size,\
                    upper_left_corner[1]:upper_left_corner[1]+patch_size+2*spatial_padding_size]
                select_inds = select_inds.reshape([-1,2])
                cropped_inds =\
                    coords[upper_left_corner[0]:upper_left_corner[0]+patch_size,\
                    upper_left_corner[1]:upper_left_corner[1]+patch_size]
                cropped_inds = cropped_inds.reshape([-1,2])
                target_s = img_target[cropped_inds[:, 0], cropped_inds[:, 1], :]
            else:
                coords = coords.reshape((-1, 2))
                select_inds = np.random.choice(
                    coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
                )
                select_inds = coords[select_inds]
                target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            batch_rays = torch.stack([ray_origins, ray_directions], dim=0)

            rgb_coarse, _, _, rgb_fine, _, _,rgb_SR,_,_ = run_one_iter_of_nerf(
                cur_H if not spatial_sampling else patch_size,
                cur_W if not spatial_sampling else patch_size,
                cur_focal,
                model_coarse,
                model_fine,
                batch_rays,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                SR_model=None if planes_model else SR_model,
                ds_factor_or_id=scene_ids[img_idx] if planes_model else ds_factor[img_idx],
                spatial_margin=SR_model.receptive_field//2 if spatial_sampling else None
            )
            target_ray_values = target_s

        if SR_experiment=="model" and not planes_model:
            loss = torch.nn.functional.mse_loss(
                    rgb_SR[..., :3], target_ray_values[..., :3]
                )
        else:
            coarse_loss = torch.nn.functional.mse_loss(
                rgb_coarse[..., :3], target_ray_values[..., :3]
            )
            fine_loss = None
            if rgb_fine is not None:
                fine_loss = torch.nn.functional.mse_loss(
                    rgb_fine[..., :3], target_ray_values[..., :3]
                )
            if SR_experiment=="model":
                if getattr(cfg.super_resolution.training,'loss','both')=='fine': coarse_loss = None
                elif getattr(cfg.super_resolution.training,'loss','both')=='coarse': fine_loss = None
            # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
            loss = (coarse_loss if coarse_loss is not None else 0.0) + (fine_loss if fine_loss is not None else 0.0)
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()
        # If training an SR model operating on planes, discarding super-resolved planes after updating the model:
        if planes_model and SR_experiment=='model': SR_model.weights_updated()
        optimizer.zero_grad()
        if iter % cfg.experiment.print_every == 0 or iter == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(iter)
                + " Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr)
            )
        writer.add_scalar("train/loss", loss.item(), iter)
        if SR_experiment!="model" or planes_model:
            if coarse_loss is not None:
                writer.add_scalar("train/coarse_loss", coarse_loss.item(), iter)
            if fine_loss is not None:
                writer.add_scalar("train/fine_loss", fine_loss.item(), iter)
                writer.add_scalar("train/fine_psnr", mse2psnr(fine_loss.item()), iter)
        writer.add_scalar("train/psnr", psnr, iter)

        if iter>0 and iter % cfg.experiment.save_every == 0 or iter == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": iter,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            if SR_experiment=="model":
                checkpoint_dict.update({"SR_model":SR_model.state_dict()})
            else:
                checkpoint_dict.update({"model_coarse_state_dict": model_coarse.state_dict()})
                if model_fine:  checkpoint_dict.update({"model_fine_state_dict": model_fine.state_dict()})
                if planes_model:    checkpoint_dict.update({"coords_normalization": model_fine.box_coords})
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(iter).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")



if __name__ == "__main__":
    main()
