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
from re import search

import models
# from cfgnode import CfgNode
from load_blender import load_blender_data
import load_DTU
from nerf_helpers import *
from train_utils import eval_nerf, run_one_iter_of_nerf,find_latest_checkpoint
from mip import IntegratedPositionalEncoding
from deepdiff import DeepDiff
from copy import deepcopy
from shutil import copyfile

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
    if SR_experiment:
        assert not hasattr(cfg.dataset,"downsampling_factor")
        LR_model_folder = cfg.models.path
        if os.path.isfile(LR_model_folder):   LR_model_folder = "/".join(LR_model_folder.split("/")[:-1])
        LR_model_config = get_config(os.path.join(LR_model_folder,"config.yml"))
        # set_config_defaults(source=LR_model_config.models,target=cfg.models)

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    print('Logs and models will be saved into %s'%(logdir))
    if configargs.load_checkpoint=="resume":
        configargs.load_checkpoint = logdir
    else:
        if configargs.load_checkpoint=='':
            if os.path.exists(logdir):  assert len([f for f in os.listdir(logdir) if '.ckpt' in f])==0,'Folder %s already contains saved models.'%(logdir)
            os.makedirs(logdir, exist_ok=True)
        # Write out config parameters.
        with open(os.path.join(logdir, "config.yml"), "w") as f:
            f.write(cfg.dump())  # cfg, f, default_flow_style=False)
    if configargs.load_checkpoint!='':
        assert os.path.exists(configargs.load_checkpoint)
    writer = SummaryWriter(logdir)
    load_saved_models = SR_experiment or os.path.exists(configargs.load_checkpoint)

    internal_SR, = False,
    if SR_experiment:
        set_config_defaults(source=LR_model_config.models,target=cfg.models)
        existing_LR_scenes = [f.split('.')[0][len('coarse_'):] for f in os.listdir(os.path.join(cfg.models.path,'planes')) if '.par' in f]
        internal_SR = False #isinstance(LR_model_ds_factor,list) and len(LR_model_ds_factor)>1
    else:
        existing_LR_scenes = None
    planes_model = cfg.models.coarse.type=="TwoDimPlanesModel"
    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    # Load dataset
    dataset_type = getattr(cfg.dataset,'type','synt')
    scenes_set = set()
    assert dataset_type in['synt','DTU']
    assert not getattr(cfg.dataset,'half_res',False),'Depricated. Use downsampling factor instead.'
    assert not hasattr(cfg.dataset,'downsampling_factor'),'Depricated.'
    def get_scene_id(basedir,ds_factor,plane_res):
        return '%s_DS%d_PlRes%s'%(basedir,ds_factor,'' if plane_res[0] is None else '%d_%d'%(plane_res))
    ds_factor_extraction_pattern = lambda name: '(?<='+name.split('_DS')[0]+'_DS'+')(\d)+(?=_PlRes'+name.split('_PlRes')[1]+')'
    if dataset_type=='synt':
        def get_scene_configs(config_dict):
            ds_factors,dir,plane_res = [],[],[]
            for conf,scenes in config_dict.items():
                conf = eval(conf)
                for s in scenes:
                    ds_factors.append(conf[0])
                    plane_res.append((conf[1],conf[2] if len(conf)>2 else conf[1]))
                    dir.append(s)
            return ds_factors,dir,plane_res
        downsampling_factors,train_dirs,plane_resolutions = get_scene_configs(cfg.dataset.dir.train)
        val_only_dirs = get_scene_configs(getattr(cfg.dataset.dir,'val',{}))
        downsampling_factors += val_only_dirs[0]
        plane_resolutions += val_only_dirs[2]
        val_only_dirs = val_only_dirs[1]
        basedirs = train_dirs+val_only_dirs
        images, poses, render_poses, hwfDs, i_split,scene_ids = [],torch.zeros([0,4,4]),[],[[],[],[],[]],[np.array([]).astype(np.int64) for i in range(3)],[]
        scene_id,val_only_scene_ids,coords_normalization = -1,[],{}
        scene_id_plane_resolution = {}
        ds_factor_ratio = []
        i_train,i_val = OrderedDict(),OrderedDict()
        font_scale = 4/min(downsampling_factors)
        for basedir,ds_factor,plane_res in zip(tqdm(basedirs,desc='Loading scenes'),downsampling_factors,plane_resolutions):
            scene_id = get_scene_id(basedir,ds_factor,plane_res)
            if SR_experiment:
                new_scene_id,original_ds = find_scene_match(existing_scenes=existing_LR_scenes,pattern=ds_factor_extraction_pattern(scene_id))
                ds_factor_ratio.append(int(original_ds)/int(find_scene_match([scene_id],ds_factor_extraction_pattern(scene_id))[1]))
                scene_id = new_scene_id
            scenes_set.add(scene_id)
            val_only = basedir not in train_dirs
            if val_only:    val_only_scene_ids.append(scene_id)
            if planes_model and not internal_SR:
                scene_id_plane_resolution[scene_id] = plane_res
            cur_images, cur_poses, cur_render_poses, cur_hwfDs, cur_i_split = load_blender_data(
                os.path.join(cfg.dataset.root,basedir),
                testskip=cfg.dataset.testskip,
                downsampling_factor=cfg.super_resolution.ds_factor if internal_SR else ds_factor,
                val_downsampling_factor=1 if internal_SR else None,
                cfg=cfg,
                val_only=val_only,
            )

            if planes_model and not load_saved_models: # No need to calculate the per-scene normalization coefficients as those will be loaded with the saved model.
                coords_normalization[scene_id] =\
                    calc_scene_box({'camera_poses':cur_poses.numpy()[:,:3,:4],'near':cfg.dataset.near,'far':cfg.dataset.far,'H':cur_hwfDs[0],'W':cur_hwfDs[1],'f':cur_hwfDs[2]},including_dirs=cfg.nerf.use_viewdirs)
            i_val[scene_id] = [v+len(images) for v in cur_i_split[1]]
            if not val_only:    i_train[scene_id] = [v+len(images) for v in cur_i_split[0]]
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
        dataset = load_DTU.DVRDataset(config=cfg.dataset,scene_id_func=get_scene_id,eval_ratio=0.1,
            existing_scenes2match=existing_LR_scenes,ds_factor_extraction_pattern=ds_factor_extraction_pattern)
        val_only_scene_ids = dataset.val_scene_IDs()
        scene_ids = dataset.per_im_scene_id
        scenes_set = set(scene_ids)
        font_scale = 4/min(dataset.downsampling_factors)
        scene_id_plane_resolution = dataset.scene_id_plane_resolution
        basedirs,coords_normalization = [],{}
        scene_iterator = range(dataset.num_scenes()) if load_saved_models else trange(dataset.num_scenes(),desc='Computing scene bounding boxes')
        for id in scene_iterator:
            if not load_saved_models:
                scene_info = dataset.scene_info(id)
                scene_info.update({'near':dataset.z_near,'far':dataset.z_far})
                coords_normalization[dataset.scene_IDs[id]] = calc_scene_box(scene_info,including_dirs=cfg.nerf.use_viewdirs)
            basedirs.append(dataset.scene_IDs[id])
        i_val = dataset.i_val
        i_train = dataset.train_ims_per_scene
        if SR_experiment:
            ds_factor_ratio = dataset.ds_factor_ratios
    if hasattr(cfg.dataset,'max_scenes_eval'):
        i_val,val_only_scene_ids = subsample_dataset(scenes_dict=i_val,max_scenes=cfg.dataset.max_scenes_eval,val_only_scenes=val_only_scene_ids,max_val_only_scenes=cfg.dataset.max_scenes_eval)
    scenes4which2save_ims = 1*list(i_val.keys())
    if hasattr(cfg.dataset,'max_scene_savings'):
        raise Exception('Should be fixed after adding max_scenes_eval')
        scenes4which2save_ims = [scenes4which2save_ims[i] for i in np.unique(np.round(np.linspace(0,len(scenes4which2save_ims)-1,cfg.dataset.max_scene_savings)).astype(int))]
    val_ims_per_scene = len(i_val[list(i_val.keys())[0]])
    EVAL_TRAINING_TOO = True
    if EVAL_TRAINING_TOO:
        for id in scenes_set:
            if id not in i_train:    continue
            im_freq = len(i_train[id])//val_ims_per_scene
            if id in i_val.keys(): #Avoid evaluating training images for scenes which were discarded for evaluation due to max_scenes_eval
                i_val[id+'_train'] = [x for i,x in enumerate(i_train[id]) if (i+im_freq//2)%im_freq==0]
    training_scenes = list(i_train.keys())
    evaluation_sequences = list(i_val.keys())
    val_strings = ['blind_validation' if id in val_only_scene_ids else 'train_imgs' if '_train' in id else 'validation' for id in evaluation_sequences]
    def print_scenes_list(title,scenes):
        print('%d %s scenes:\n'%(len(scenes),title))
        print(scenes)
    print_scenes_list('training',training_scenes)
    for cat in set(val_strings):
        print_scenes_list('"%s" evaluation'%(cat),[s for i,s in enumerate(evaluation_sequences) if val_strings[i]==cat])
    assert all([val_ims_per_scene==len(i_val[id]) for id in evaluation_sequences]),'Assuming all scenes have the same number of evaluation images'
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
    if planes_model:
        if hasattr(cfg.nerf.train,'viewdir_downsampling'):  assert hasattr(cfg.nerf.train,'max_plane_downsampling')
        store_planes = hasattr(cfg.nerf.train,'store_planes')
        if store_planes:    assert not getattr(cfg.nerf.train,'save_GPU_memory',False),'I think this is unnecessary.'
        model_coarse = models.TwoDimPlanesModel(
            use_viewdirs=cfg.nerf.use_viewdirs,
            scene_id_plane_resolution=None if store_planes else scene_id_plane_resolution,
            coords_normalization = None if store_planes else coords_normalization,
            dec_density_layers=getattr(cfg.models.coarse,'dec_density_layers',4),
            dec_rgb_layers=getattr(cfg.models.coarse,'dec_rgb_layers',4),
            dec_channels=getattr(cfg.models.coarse,'dec_channels',128),
            skip_connect_every=getattr(cfg.models.coarse,'skip_connect_every',None),
            num_plane_channels=getattr(cfg.models.coarse,'num_plane_channels',48),
            rgb_dec_input=getattr(cfg.models.coarse,'rgb_dec_input','projections'),
            proj_combination=getattr(cfg.models.coarse,'proj_combination','sum'),
            plane_interp=getattr(cfg.models.coarse,'plane_interp','bilinear'),
            align_corners=getattr(cfg.models.coarse,'align_corners',True),
            interp_viewdirs=getattr(cfg.models.coarse,'interp_viewdirs',None),
            viewdir_downsampling=getattr(cfg.nerf.train,'viewdir_downsampling',True),
            viewdir_proj_combination=getattr(cfg.models.coarse,'viewdir_proj_combination',None),
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
    model_coarse.optional_no_grad = null_with
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
                set_config_defaults(source=cfg.models.coarse,target=cfg.models.fine)
                model_fine = models.TwoDimPlanesModel(
                    use_viewdirs=cfg.nerf.use_viewdirs,
                    scene_id_plane_resolution=None if store_planes else scene_id_plane_resolution,
                    coords_normalization = None if store_planes else coords_normalization,
                    dec_density_layers=getattr(cfg.models.fine,'dec_density_layers',4),
                    dec_rgb_layers=getattr(cfg.models.fine,'dec_rgb_layers',4),
                    dec_channels=getattr(cfg.models.fine,'dec_channels',128),
                    skip_connect_every=getattr(cfg.models.fine,'skip_connect_every',None),
                    num_plane_channels=getattr(cfg.models.fine,'num_plane_channels',48),
                    rgb_dec_input=getattr(cfg.models.fine,'rgb_dec_input','projections'),
                    proj_combination=getattr(cfg.models.fine,'proj_combination','sum'),
                    planes=model_coarse.planes_ if (not store_planes and getattr(cfg.models.fine,'use_coarse_planes',False)) else None,
                    plane_interp=getattr(cfg.models.fine,'plane_interp','bilinear'),
                    align_corners=getattr(cfg.models.fine,'align_corners',True),
                    interp_viewdirs=getattr(cfg.models.fine,'interp_viewdirs',None),
                    viewdir_downsampling=getattr(cfg.nerf.train,'viewdir_downsampling',True),
                    viewdir_proj_combination=getattr(cfg.models.fine,'viewdir_proj_combination',None),
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


    if planes_model and getattr(cfg.nerf.train,'save_GPU_memory',False):
        model_coarse.planes2cpu()
        model_fine.planes2cpu()

    if SR_experiment=="model":
        if cfg.super_resolution.model.type=='EDSR':
            plane_channels = getattr(cfg.models.coarse,'num_plane_channels',48)
            ds_factor_ratio = list(set(ds_factor_ratio))
            assert len(ds_factor_ratio)==1 and np.round(ds_factor_ratio[0])==ds_factor_ratio[0]
            sf_config = getattr(cfg.super_resolution.model,'scale_factor','linear')
            assert sf_config in ['linear','sqrt','one']
            if sf_config=='one':
                SR_factor = 1
            elif sf_config=='linear':
                SR_factor = int(ds_factor_ratio[0])
            else:
                SR_factor = int(np.sqrt(ds_factor_ratio[0]))
            SR_model = getattr(models, cfg.super_resolution.model.type)(
                # scale_factor=cfg.super_resolution.model.scale_factor,
                scale_factor=SR_factor,
                # scale_factor=cfg.super_resolution.ds_factor,
                in_channels=plane_channels,
                out_channels=plane_channels,
                hidden_size=cfg.super_resolution.model.hidden_size,
                plane_interp=getattr(cfg.super_resolution,'plane_resize_mode',model_fine.plane_interp),
                n_blocks=cfg.super_resolution.model.n_blocks,
                input_normalization=cfg.super_resolution.get("input_normalization",False),
                consistentcy_loss_w=cfg.super_resolution.get("consistentcy_loss_w",None),
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
                xyz_input_2_dir=cfg.super_resolution.model.get("xyz_input_2_dir",False),
            )
            print("SR model: %d parameters, input dimension xyz: %d, dirs: %d"%\
                (num_parameters(SR_model),SR_input_dim[0],SR_input_dim[1]))
        SR_model.to(device)
        # trainable_parameters = list(SR_model.parameters())
        trainable_parameters = [p for k,p in SR_model.named_parameters() if 'NON_LEARNED' not in k]
    else:
        # Initialize optimizer.
        def collect_params(model,filter='all'):
            assert filter in ['all','planes','non_planes']
            params = []
            if filter!='non_planes':
                plane_params = dict(sorted([p for p in model.named_parameters() if 'planes_.sc' in p[0]],key=lambda p:p[0]))
                params.extend(plane_params.values())
            if filter!='planes':
                params.extend([p[1] for p in model.named_parameters() if 'planes_.sc' not in p[0]])
            if filter=='planes':
                return plane_params
            else:
                return params

        trainable_parameters = collect_params(model_coarse,filter='non_planes' if store_planes else 'all')
        if model_fine is not None:
            if planes_model:
                if cfg.models.fine.type!="use_same":
                    if getattr(cfg.models.fine,'use_coarse_planes',False):
                        trainable_parameters += collect_params(model_fine,filter='non_planes')
                    else:
                        trainable_parameters += collect_params(model_fine,filter='all')
            else: 
                trainable_parameters += list(model_fine.parameters())
        SR_model = None
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # Load an existing checkpoint, if a path is specified.
    start_i,eval_counter = 0,0
    best_saved,last_saved = (0,np.finfo(np.float32).max),[]
    if load_saved_models:
        if SR_experiment:
            saved_rgb_fine = dict(zip(evaluation_sequences,[{} for i in evaluation_sequences]))
            checkpoint = find_latest_checkpoint(cfg.models.path)
            print("Using LR model %s"%(checkpoint))
            if SR_experiment=="model" and os.path.exists(configargs.load_checkpoint):
                assert os.path.isdir(configargs.load_checkpoint)
                SR_model_checkpoint = os.path.join(configargs.load_checkpoint,sorted([f for f in os.listdir(configargs.load_checkpoint) if "checkpoint" in f and f[-5:]==".ckpt"],key=lambda x:int(x[len("checkpoint"):-5]))[-1])
                start_i = int(SR_model_checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
                if 'eval_counter' in torch.load(SR_model_checkpoint).keys(): eval_counter = torch.load(SR_model_checkpoint)['eval_counter']
                if 'best_saved' in torch.load(SR_model_checkpoint).keys(): best_saved = torch.load(SR_model_checkpoint)['best_saved']
                print("Resuming training on model %s"%(SR_model_checkpoint))
                saved_config_dict = get_config(os.path.join(configargs.load_checkpoint,"config.yml"))
                config_diffs = DeepDiff(saved_config_dict,cfg)
                for diff in [config_diffs[ch_type] for ch_type in ['dictionary_item_removed','dictionary_item_added','values_changed'] if ch_type in config_diffs]:
                    print(diff)

                SR_model.load_state_dict(torch.load(SR_model_checkpoint)["SR_model"])
        else:
            checkpoint = find_latest_checkpoint(configargs.load_checkpoint)
            start_i = int(checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
            if 'eval_counter' in torch.load(checkpoint).keys(): eval_counter = torch.load(checkpoint)['eval_counter']
            if 'best_saved' in torch.load(checkpoint).keys(): best_saved = torch.load(checkpoint)['best_saved']
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
            return state_dict
            if any(['planes_.' in k and '.sc' not in k for k in state_dict.keys()]):
                return OrderedDict([(k.replace('planes_.','planes_.sc0_res32_D'),v) for k,v in state_dict.items()])
            else:
                return OrderedDict([(k.replace('planes.','planes_.sc0_res32_D'),v) for k,v in state_dict.items()])

        if False:
            def rep_name(name):
                ind = search('(?<=.sc)(\d)+(?=_D)',name).group(0)
                return name.replace('.sc'+ind,'.sc'+basedirs[int(ind)])
            checkpoint["model_coarse_state_dict"] = OrderedDict([(rep_name(k),v) if '.sc' in k else (k,v) for k,v in checkpoint["model_coarse_state_dict"].items()])
            checkpoint["model_fine_state_dict"] = OrderedDict([(rep_name(k),v) if '.sc' in k else (k,v) for k,v in checkpoint["model_fine_state_dict"].items()])
            torch.save(checkpoint,find_latest_checkpoint(cfg.models.path))

        def load_saved_parameters(model,saved_params):
            mismatch = model.load_state_dict(saved_params,strict=False)
            assert (len(mismatch.missing_keys)==0 or (store_planes and all(['planes_.sc' in k for k in mismatch.missing_keys]))) and all(['planes_.sc' in k for k in mismatch.unexpected_keys])
            if planes_model and not store_planes:
                model.box_coords = checkpoint["coords_normalization"]


        load_saved_parameters(model_coarse,checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            load_saved_parameters(model_fine,checkpoint["model_fine_state_dict"])
        if store_planes and getattr(cfg.nerf.train.store_planes,'save2checkpoint',False):
            raise Exception('No longer supported.')
            if SR_experiment:   raise Exception('See what needs to be done in this case (SR+saving planes to checkpoint) if and when happens')
            planes_opt.load_from_checkpoint(checkpoint=checkpoint)
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

    spatial_padding_size = SR_model.receptive_field//2 if isinstance(SR_model,models.Conv3D) else 0
    spatial_sampling = spatial_padding_size>0 or cfg.nerf.train.get("spatial_sampling",False)
    if spatial_padding_size>0 or spatial_sampling:
        SAMPLE_PATCH_BY_CONTENT = True
        patch_size = chunksize_to_2D(cfg.nerf.train.num_random_rays)
        patch_size = int(np.ceil(spatial_sampling*patch_size))
        if SAMPLE_PATCH_BY_CONTENT:
            assert getattr(cfg.nerf.train,'spatial_patch_sampling','background_est') in ['background_est','STD']
            if getattr(cfg.nerf.train,'spatial_patch_sampling','background_est')=='STD':
                im_2_sampling_dist = image_STD_2_distribution(patch_size=patch_size)
            else:
                im_2_sampling_dist = estimated_background_2_distribution(patch_size=patch_size)

    assert isinstance(spatial_sampling,bool) or spatial_sampling>=1
    if planes_model and SR_model is not None:
        save_RAM_memory = not store_planes and len(model_coarse.planes_)>=10
        if getattr(cfg.super_resolution,'apply_2_coarse',False):
            model_coarse.assign_SR_model(SR_model,SR_viewdir=cfg.super_resolution.SR_viewdir,save_interpolated=not save_RAM_memory,set_planes=not store_planes)
        else:
            assert getattr(cfg.super_resolution.training,'loss','both')=='fine'
            model_coarse.optional_no_grad = torch.no_grad
        model_fine.assign_SR_model(SR_model,SR_viewdir=cfg.super_resolution.SR_viewdir,save_interpolated=not save_RAM_memory,set_planes=not store_planes)
    if store_planes:
        planes_folder = os.path.join(LR_model_folder if SR_experiment else logdir,'planes')
        if os.path.isdir(planes_folder):
            resave_checkpoint = False
            param_files_list = [f for f in glob.glob(planes_folder+'/*') if '.par' in f]
            if len(param_files_list)>0 and (any(['DTU' in f.split('/')[-1] for f in param_files_list]) or 'coords_normalization' not in torch.load(param_files_list[0]).keys()):
                if any(['DTU' in f.split('/')[-1] for f in param_files_list]):
                    from scripts import convert_DTU_scene_names
                    name_mapping = convert_DTU_scene_names.name_mapping()
                checkpoint_name = find_latest_checkpoint(cfg.models.path if SR_experiment else configargs.load_checkpoint)
                checkpoint = torch.load(checkpoint_name)
                for f in tqdm(param_files_list,desc='!!! Converting saved planes to new name convention !!!',):
                    cur_scene_name = f.split('/')[-1][len('coarse_'):-4]
                    params = torch.load(f)
                    if 'DTU' in cur_scene_name:
                        params['params'] = torch.nn.ParameterDict([(k.replace(cur_scene_name,name_mapping[cur_scene_name]),v) for k,v in params['params'].items()])
                    if 'coords_normalization' in checkpoint:
                        resave_checkpoint = True
                        params['coords_normalization'] = checkpoint['coords_normalization'].pop(cur_scene_name)
                    torch.save(params,f.replace(cur_scene_name,name_mapping[cur_scene_name]) if 'DTU' in cur_scene_name else f)
                    if 'DTU' in cur_scene_name:
                        os.remove(f)
            elif hasattr(checkpoint_config.models.coarse,'plane_resolutions'):
                # checkpoint = torch.load(checkpoint_name)
                for f in tqdm(param_files_list,desc='!!! Converting saved planes to new name convention !!!',):
                    cur_scene_name = f.split('/')[-1][len('coarse_'):-4]
                    params = torch.load(f)
                    if '_PlRes' not in cur_scene_name:
                        new_scene_id = get_scene_id(cur_scene_name,ds_factor=getattr(checkpoint_config.dataset,'downsampling_factor',1),
                            plane_res=(checkpoint_config.models.coarse.plane_resolutions,checkpoint_config.models.coarse.viewdir_plane_resolution))
                        params['params'] = torch.nn.ParameterDict([(k.replace(cur_scene_name,new_scene_id),v) for k,v in params['params'].items()])
                    torch.save(params,f.replace(cur_scene_name,new_scene_id) if '_PlRes' not in cur_scene_name else f)
                    if '_PlRes' not in cur_scene_name:
                        os.remove(f)
                checkpoint_config.models.coarse.pop('plane_resolutions',None)
                checkpoint_config.models.coarse.pop('viewdir_plane_resolution',None)
                checkpoint_config.dataset.pop('downsampling_factor',None)
                checkpoint_name = os.path.join(cfg.models.path if SR_experiment else configargs.load_checkpoint,'config.yml')
                with open(checkpoint_name, "w") as f:
                    f.write(checkpoint_config.dump())  # cfg, f, default_flow_style=False)
            
            if resave_checkpoint:
                assert all([s in getattr(cfg.dataset,'excluded_scenes',[]) for s in checkpoint['coords_normalization'].keys()])
                checkpoint.pop('coords_normalization',None)
                torch.save(checkpoint,checkpoint_name)
        else:
            os.mkdir(planes_folder)
        scenes_cycle_counter = Counter()
        planes_opt = models.PlanesOptimizer(optimizer_type=cfg.optimizer.type,
            scene_id_plane_resolution=scene_id_plane_resolution,options=cfg.nerf.train.store_planes,save_location=planes_folder,
            lr=getattr(cfg.optimizer,'planes_lr',cfg.optimizer.lr),model_coarse=model_coarse,model_fine=model_fine,
            use_coarse_planes=getattr(cfg.models.fine,'use_coarse_planes',False),
            init_params=not load_saved_models,optimize=not SR_experiment,training_scenes=training_scenes,
            coords_normalization=None if load_saved_models else coords_normalization,
            do_when_reshuffling=lambda:scenes_cycle_counter.step(print_str='Number of scene cycles performed: '),
            STD_factor=getattr(cfg.nerf.train,'STD_factor',0.1),
        )
    if planes_model:    assert not (hasattr(cfg.models.coarse,'plane_resolutions') or hasattr(cfg.models.coarse,'viewdir_plane_resolution')),'Depricated.'
    if SR_experiment=="model" and getattr(cfg.super_resolution,'input_normalization',False) and not os.path.exists(configargs.load_checkpoint):
        #Initializing a new SR model that uses input normalization
        SR_model.normalization_params(planes_opt.get_plane_stats(viewdir=getattr(cfg.super_resolution,'SR_viewdir',False)))

    downsampling_offset = lambda ds_factor: (ds_factor-1)/(2*ds_factor)
    saved_target_ims = dict(zip(val_strings,[set() for i in val_strings]))#set()
    if isinstance(spatial_sampling,bool):
        spatial_margin = SR_model.receptive_field//2 if spatial_sampling else None
    else:
        spatial_margin = 0
    virtual_batch_size = getattr(cfg.nerf.train,'virtual_batch_size',1)

    def evaluate():
        tqdm.write("[VAL] =======> Iter: " + str(iter))
        model_coarse.eval()
        if model_fine:
            model_fine.eval()
        if SR_experiment=="model":
            SR_model.eval()

        start = time.time()
        with torch.no_grad():
            rgb_coarse, rgb_fine = None, None
            target_ray_values = None
            if True:
                # val_ind = lambda dummy: (iter//cfg.experiment.validate_every)%val_ims_per_scene
                val_ind = lambda dummy: eval_counter%val_ims_per_scene
            else: #For the previous case of having only a single training image to evaluate on:
                val_ind = lambda set_id: (iter//cfg.experiment.validate_every)%val_ims_per_scene if 'validation' in set_id else 0
            # val_strings = ['blind_validation' if id in val_only_scene_ids else 'train_imgs' if '_train' in id else 'validation' for id in evaluation_sequences]
            img_indecis = [v[val_ind(val_strings[i])] for i,v in enumerate(i_val.values())]
            if val_ims_dict is not None:
                raise Exception('Revisit after enabling multi-scene.')
                img_indecis += [i_val[val_ims_dict["closest_val"]],i_val[val_ims_dict["furthest_val"]]]
                # val_strings += ["closest_","furthest_"]
            # img_idx = 0
            record_fine = True
            coarse_loss,fine_loss,loss,psnr,rgb_coarse,rgb_fine,rgb_SR,target_ray_values = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
            ims2save = defaultdict(list)
            for scene_num,img_idx in enumerate(tqdm(img_indecis,desc='Evaluating scenes')):
                if any([s in evaluation_sequences[scene_num] for s in scenes4which2save_ims]):  ims2save[val_strings[scene_num]].append(len(rgb_fine[val_strings[scene_num]]))
                if dataset_type=='synt':
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    cur_H,cur_W,cur_focal,cur_ds_factor = H[img_idx], W[img_idx], focal[img_idx],ds_factor[img_idx]
                else:
                    img_target,pose_target,cur_H,cur_W,cur_focal,cur_ds_factor = dataset.item(img_idx,device)
                ray_origins, ray_directions = get_ray_bundle(
                    cur_H, cur_W, cur_focal, pose_target,padding_size=spatial_padding_size,downsampling_offset=downsampling_offset(cur_ds_factor),
                )
                if store_planes:
                    planes_opt.load_scene(scene_ids[img_idx])
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
                    spatial_margin=spatial_margin,
                )
                target_ray_values[val_strings[scene_num]].append(img_target[...,:3])
                if SR_experiment:
                    if SR_experiment=="refine" or planes_model:
                        rgb_SR_ = 1*rgb_fine_
                        rgb_SR_coarse_ = 1*rgb_coarse_
                        if val_ind(val_strings[scene_num]) not in saved_rgb_fine[evaluation_sequences[scene_num]]:
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
                            saved_rgb_fine[evaluation_sequences[scene_num]][val_ind(val_strings[scene_num])] = 1*rgb_fine_.detach()
                        else:
                            record_fine = False
                            rgb_fine_ = 1*saved_rgb_fine[evaluation_sequences[scene_num]][val_ind(val_strings[scene_num])]
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
                if planes_model and SR_model is not None and (store_planes or save_RAM_memory):
                    SR_model.clear_SR_planes(all_planes=store_planes)
            SAVE_COARSE_IMAGES = False
            for val_set in set(val_strings):
                # font_scale = 4/min(downsampling_factors)
                if SR_experiment:
                    writer.add_scalar("%s/SR_psnr_gain"%(val_set), np.mean([psnr[val_set][i]-mse2psnr(fine_loss[val_set][i]) for i in range(len(psnr[val_set]))]), iter)
                    writer.add_image(
                        "%s/rgb_SR"%(val_set), arange_ims([rgb_SR[val_set][i] for i in ims2save[val_set]],str(val_ind(val_set)),psnrs=[psnr[val_set][i] for i in ims2save[val_set]],fontScale=font_scale),iter
                    )
                writer.add_scalar("%s/fine_loss"%(val_set), np.mean(fine_loss[val_set]), iter)
                writer.add_scalar("%s/fine_psnr"%(val_set), np.mean([mse2psnr(l) for l in fine_loss[val_set]]), iter)
                writer.add_scalar("%s/loss"%(val_set), np.mean(loss[val_set]), iter)
                writer.add_scalar("%s/psnr"%(val_set), np.mean(psnr[val_set]), iter)
                if len(coarse_loss[val_set])>0:
                    writer.add_scalar("%s/coarse_loss"%(val_set), np.mean(coarse_loss[val_set]), iter)
                if len(rgb_fine[val_set])>0:
                    if record_fine:
                        writer.add_scalar("%s/fine_loss"%(val_set), np.mean(fine_loss[val_set]), val_ind(val_set) if SR_experiment else iter)
                        writer.add_image("%s/rgb_fine"%(val_set), arange_ims([rgb_fine[val_set][i] for i in ims2save[val_set]],str(val_ind(val_set)),psnrs=[mse2psnr(l) for i,l in enumerate(fine_loss[val_set]) if i in ims2save[val_set]],fontScale=font_scale),val_ind(val_set) if SR_experiment else iter)
                if SAVE_COARSE_IMAGES:
                    writer.add_image(
                        "%s/rgb_coarse"%(val_set), arange_ims([rgb_coarse[val_set][i] for i in ims2save[val_set]],str(val_ind(val_set)),psnrs=[mse2psnr(l) for i,l in enumerate(coarse_loss[val_set]) if i in ims2save[val_set]],fontScale=font_scale),iter
                    )
                if val_ind(val_set) not in saved_target_ims[val_set]:
                    writer.add_image("%s/img_target"%(val_set), arange_ims([target_ray_values[val_set][i] for i in ims2save[val_set]],str(val_ind(val_set)),fontScale=font_scale),val_ind(val_set))
                    saved_target_ims[val_set].add(val_ind(val_set))
                tqdm.write(
                    "%s:\tValidation loss: "%(val_set)
                    + str(np.mean(loss[val_set]))
                    + "\tValidation PSNR: "
                    + str(np.mean(psnr[val_set]))
                    + "\tTime: "
                    + str(time.time() - start)
                )
        return loss,psnr

    def train():
        first_v_batch_iter = iter%virtual_batch_size==0
        last_v_batch_iter = iter%virtual_batch_size==(virtual_batch_size-1)
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
        if SR_HR_im_inds is None:
            if store_planes:
                img_idx = np.random.choice(available_train_inds)
            else:
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
            img_target,pose_target,cur_H,cur_W,cur_focal,cur_ds_factor = dataset.item(img_idx,device)

        ray_origins, ray_directions = get_ray_bundle(cur_H, cur_W, cur_focal, pose_target,padding_size=spatial_padding_size,downsampling_offset=downsampling_offset(cur_ds_factor),)
        coords = torch.stack(
            meshgrid_xy(torch.arange(cur_H+2*spatial_padding_size).to(device), torch.arange(cur_W+2*spatial_padding_size).to(device)),
            dim=-1,
        )
        if spatial_padding_size>0 or spatial_sampling:
            if SAMPLE_PATCH_BY_CONTENT:
                patches_vacancy_dist = im_2_sampling_dist(img_target[...,:3])
                upper_left_corner = torch.argwhere(torch.rand([])<torch.cumsum(patches_vacancy_dist.reshape([-1]),0))[0].item()
                upper_left_corner = np.array([upper_left_corner//patches_vacancy_dist.shape[1],upper_left_corner%patches_vacancy_dist.shape[1]])
            else:
                upper_left_corner = np.random.uniform(size=[2])*(np.array([cur_H,cur_W])-patch_size)
                upper_left_corner = np.floor(upper_left_corner).astype(np.int32)
            select_inds = \
                coords[upper_left_corner[1]:upper_left_corner[1]+patch_size+2*spatial_padding_size,\
                upper_left_corner[0]:upper_left_corner[0]+patch_size+2*spatial_padding_size]
            select_inds = select_inds.reshape([-1,2])
            cropped_inds =\
                coords[upper_left_corner[1]:upper_left_corner[1]+patch_size,\
                upper_left_corner[0]:upper_left_corner[0]+patch_size]
            cropped_inds = cropped_inds.reshape([-1,2])
            if spatial_sampling>1:
                selected_witin_selected = np.sort(np.random.permutation(select_inds.shape[0])[:cfg.nerf.train.num_random_rays])
                select_inds = select_inds[selected_witin_selected,:]
                cropped_inds = cropped_inds[selected_witin_selected,:]
            target_s = img_target[cropped_inds[:, 0], cropped_inds[:, 1], :]
        else:
            coords = coords.reshape((-1, 2))
            select_inds = np.random.choice(
                coords.shape[0], size=(min(img_target.shape[0]*img_target.shape[1],cfg.nerf.train.num_random_rays)), replace=False
            )
            select_inds = coords[select_inds]
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        batch_rays = torch.stack([ray_origins, ray_directions], dim=0)

        if first_v_batch_iter:
            optimizer.zero_grad()
            if store_planes:    planes_opt.zero_grad()
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
            spatial_margin=spatial_margin,
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

        writer.add_scalar("train/loss", loss.item(), iter)
        if SR_experiment=="model" and planes_model:
            SR_consistency_loss = SR_model.return_consistency_loss()
            if SR_consistency_loss is not None:
                writer.add_scalar("train/inconsistency", SR_consistency_loss.item(), iter)
                loss += cfg.super_resolution.consistentcy_loss_w*SR_consistency_loss

        loss.backward()
        psnr = mse2psnr(loss.item())
        new_drawn_scenes = None
        if store_planes:
            new_drawn_scenes = planes_opt.step(opt_step=last_v_batch_iter)
        if last_v_batch_iter:
            optimizer.step()
            # If training an SR model operating on planes, discarding super-resolved planes after updating the model:
            if planes_model and SR_experiment=='model': SR_model.clear_SR_planes()
        if SR_experiment!="model" or planes_model:
            if coarse_loss is not None:
                writer.add_scalar("train/coarse_loss", coarse_loss.item(), iter)
            if fine_loss is not None:
                writer.add_scalar("train/fine_loss", fine_loss.item(), iter)
                writer.add_scalar("train/fine_psnr", mse2psnr(fine_loss.item()), iter)
        writer.add_scalar("train/psnr", psnr, iter)
        return loss.item(),psnr,new_drawn_scenes


    training_time,last_evaluated = 0,1*start_i
    recently_saved, = time.time(),
    eval_loss_since_save,print_cycle_loss,print_cycle_psnr = [],[],[]
    evaluation_time = 0

    for iter in trange(start_i,cfg.experiment.train_iters):
        # Validation
        if isinstance(cfg.experiment.validate_every,list):
            evaluate_now = evaluation_time<=training_time*cfg.experiment.validate_every[0] or iter-last_evaluated>=cfg.experiment.validate_every[1]
        else:
            evaluate_now = iter % cfg.experiment.validate_every == 0
        evaluate_now |= iter == cfg.experiment.train_iters - 1
        # print('!!!!!!!!!!WARNING!!!!!!!!!!!')
        # evaluate_now = True
        if evaluate_now:
            last_evaluated = 1*iter
            start_time = time.time()
            loss,psnr = evaluate()
            eval_loss_since_save.extend([v for v in loss['blind_validation' if 'blind_validation' in loss else 'validation']])
            evaluation_time = time.time()-start_time
            if store_planes:    
                planes_opt.draw_scenes()
                available_train_inds = [i for i in i_train if scene_ids[i] in planes_opt.cur_scenes]
            training_time = 0
            eval_counter += 1
            
        # Training:
        start_time = time.time()
        loss,psnr,new_drawn_scenes = train()
        if new_drawn_scenes is not None:
            available_train_inds = [i for i in i_train if scene_ids[i] in new_drawn_scenes]

        print_cycle_loss.append(loss)
        print_cycle_psnr.append(psnr)
        training_time += time.time()-start_time

        if iter % cfg.experiment.print_every == 0 or iter == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(iter)
                + " Loss: "
                # + str(loss.item())
                + str(np.mean(print_cycle_loss))
                + " PSNR: "
                + str(np.mean(print_cycle_psnr))
            )
            print_cycle_loss,print_cycle_psnr = [],[]
        save_now = scenes_cycle_counter.check_and_reset() if (store_planes and not SR_experiment) else False
        save_now |= iter % cfg.experiment.save_every == 0 if isinstance(cfg.experiment.save_every,int) else (time.time()-recently_saved)/60>cfg.experiment.save_every
        save_now |= iter == cfg.experiment.train_iters - 1
        save_now &= iter>0
        # if iter>0 and iter % cfg.experiment.save_every == 0 or iter == cfg.experiment.train_iters - 1:
        if save_now:
            save_as_best = False
            if len(eval_loss_since_save)>0:
                recent_loss_avg = np.mean(eval_loss_since_save)
                if recent_loss_avg<best_saved[1]:
                    best_saved = (iter,recent_loss_avg)
                    save_as_best = True
            checkpoint_dict = {
                "iter": iter,
                "eval_counter": eval_counter,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_saved":best_saved,
                # "loss": loss,
                # "psnr": psnr,
            }
            if SR_experiment=="model":
                checkpoint_dict.update({"SR_model":SR_model.state_dict()})
            else:
                checkpoint_dict.update({"model_coarse_state_dict": model_coarse.state_dict()})
                if model_fine:  checkpoint_dict.update({"model_fine_state_dict": model_fine.state_dict()})
                if planes_model:
                    if store_planes or getattr(cfg.models.fine,'use_coarse_planes',False):
                        checkpoint_dict["model_fine_state_dict"] = OrderedDict([(k,v) for k,v in checkpoint_dict["model_fine_state_dict"].items() if 'planes_.' not in k])
                    if store_planes:
                        checkpoint_dict["model_coarse_state_dict"] = OrderedDict([(k,v) for k,v in checkpoint_dict["model_coarse_state_dict"].items() if 'planes_.' not in k])
                        if getattr(cfg.nerf.train.store_planes,'save2checkpoint',False):
                            params,opt_states = planes_opt.save_params(to_checkpoint=True)
                            checkpoint_dict.update({"plane_parameters":params,"plane_optimizer_states":opt_states})
                        else:
                            planes_opt.save_params(to_checkpoint=False)
                    else:
                        checkpoint_dict.update({"coords_normalization": model_fine.box_coords})

            ckpt_name = os.path.join(logdir, "checkpoint" + str(iter).zfill(5) + ".ckpt")
            torch.save(checkpoint_dict,ckpt_name,)
            # tqdm.write("================== Saved Checkpoint =================")
            if len(last_saved)>0:
                os.remove(last_saved.pop(0))
            last_saved.append(ckpt_name)
            if save_as_best:
                print("================Saving new best checkpoint at iteration %d, with average evaluation loss %.3e====================="%(best_saved[0],best_saved[1]))
                best_ckpt_name = os.path.join(logdir, "checkpoint.best_ckpt")
                if os.path.exists(best_ckpt_name):
                    copyfile(best_ckpt_name,best_ckpt_name.replace("_ckpt","_ckpt_old"))
                torch.save(checkpoint_dict,best_ckpt_name,)
                if os.path.exists(best_ckpt_name.replace("_ckpt","_ckpt_old")):
                    os.remove(best_ckpt_name.replace("_ckpt","_ckpt_old"))
            else:
                print("================Best checkpoint is still %d, with average evaluation loss %.3e (recent average is %.3e)====================="%(best_saved[0],best_saved[1],recent_loss_avg))
            del checkpoint_dict
            recently_saved = time.time()
            eval_loss_since_save = []

    print("Done!")



if __name__ == "__main__":
    main()
