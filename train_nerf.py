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
import socket

def main():
    if 'della-' in socket.gethostname():    print('Adjusting memory settings to running on "Della" cluster.')
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
    parser.add_argument(
        "--eval",
        type=str,
        choices=['images','video'],
        default=None,
        help="Run in evaluation mode and render images/video.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to save evaluation results.",
    )
    configargs = parser.parse_args()
    eval_mode = configargs.eval
    # Read config file.
    assert (configargs.config is None) ^ (configargs.resume is None)
    cfg = None
    if configargs.config is None:
        config_file = os.path.join(configargs.resume,"config.yml")
    else:
        config_file = configargs.config
    cfg = get_config(config_file)
    if eval_mode:
        import imageio
        config_file = os.path.join(cfg.experiment.logdir, cfg.experiment.id,"config.yml")
        results_dir = os.path.join(configargs.results_path, cfg.experiment.id)
        if not os.path.isdir(results_dir):  os.mkdir(results_dir)
        print('Evaluation outputs will be saved into %s'%(results_dir))
        cfg = get_config(config_file)
    print('Using configuration file %s'%(config_file))
    print(("Evaluating" if eval_mode else "Running") + " experiment %s"%(cfg.experiment.id))
    SR_experiment = None
    if "super_resolution" in cfg:
        SR_experiment = "model" if "model" in cfg.super_resolution.keys() else "refine"
    end2end_training = getattr(cfg.nerf.train,'train_end2end',False)
    assert end2end_training in ['HR_planes','LR_planes',False]
    rep_model_training = not SR_experiment or end2end_training
    # if SR_experiment:
    if not rep_model_training:
        LR_model_folder = cfg.models.path
        if os.path.isfile(LR_model_folder):   LR_model_folder = "/".join(LR_model_folder.split("/")[:-1])
        LR_model_config = get_config(os.path.join(LR_model_folder,"config.yml"))
        if hasattr(LR_model_config.models.coarse,'plane_resolutions'):
            planes_folder = os.path.join(LR_model_folder if SR_experiment else logdir,'planes')
            param_files_list = [f for f in glob.glob(planes_folder+'/*') if '.par' in f]
            for f in tqdm(param_files_list,desc='!!! Converting saved planes to new name convention !!!',):
                cur_scene_name = f.split('/')[-1][len('coarse_'):-4]
                params = torch.load(f)
                if '_PlRes' not in cur_scene_name:
                    ds_factor = getattr(LR_model_config.dataset,'downsampling_factor',1)*(2 if hasattr(LR_model_config.dataset,'half_res') else 1)
                    new_scene_id = models.get_scene_id(cur_scene_name,ds_factor=ds_factor,
                        plane_res=(LR_model_config.models.coarse.plane_resolutions,LR_model_config.models.coarse.viewdir_plane_resolution))
                    params['params'] = torch.nn.ParameterDict([(k.replace(cur_scene_name,new_scene_id),v) for k,v in params['params'].items()])
                torch.save(params,f.replace(cur_scene_name,new_scene_id) if '_PlRes' not in cur_scene_name else f)
                if '_PlRes' not in cur_scene_name:
                    os.remove(f)
            LR_model_config.models.coarse.pop('plane_resolutions',None)
            LR_model_config.models.coarse.pop('viewdir_plane_resolution',None)
            LR_model_config.dataset.pop('downsampling_factor',None)
            LR_model_config.dataset.pop('half_res',None)
            checkpoint_name = os.path.join(cfg.models.path,'config.yml')
            with open(checkpoint_name, "w") as f:
                f.write(LR_model_config.dump())  # cfg, f, default_flow_style=False)

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    if not eval_mode:   print('Logs and models will be saved into %s'%(logdir))
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
    if not eval_mode:   writer = SummaryWriter(logdir)

    load_saved_models = not rep_model_training or os.path.exists(configargs.load_checkpoint)

    # internal_SR, = False,
    if not rep_model_training:
        set_config_defaults(source=LR_model_config.models,target=cfg.models)
        existing_LR_scenes = [f.split('.')[0][len('coarse_'):] for f in os.listdir(os.path.join(LR_model_folder,'planes')) if '.par' in f]
        # internal_SR = False #isinstance(LR_model_ds_factor,list) and len(LR_model_ds_factor)>1
    else:
        existing_LR_scenes = None
    planes_model = cfg.models.coarse.type=="TwoDimPlanesModel"
    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    # Load dataset
    dataset_type = getattr(cfg.dataset,'type','synt')
    scenes_set = set()
    assert dataset_type in['synt','DTU']
    assert not getattr(cfg.dataset,'half_res',False),'Depricated. Use downsampling factor instead.'

    if getattr(cfg.nerf.train,'plane_dropout',0)>0: assert cfg.models.coarse.proj_combination=='avg' and cfg.models.coarse.num_planes>3

    ds_factor_extraction_pattern = lambda name: '(?<='+name.split('_DS')[0]+'_DS'+')(\d)+(?=_PlRes'+name.split('_PlRes')[1]+')'
    if dataset_type=='synt':
        def get_scene_configs(config_dict):
            ds_factors,dir,plane_res,scene_ids = [],[],[],[]
            for conf,scenes in config_dict.items():
                conf = eval(conf)
                for s in scenes:
                    ds_factors.append(conf[0])
                    plane_res.append((conf[1],conf[2] if len(conf)>2 else conf[1]))
                    dir.append(s)
                    scene_ids.append(models.get_scene_id(dir[-1],ds_factors[-1],plane_res[-1]))
            return ds_factors,dir,plane_res,scene_ids
        downsampling_factors,train_dirs,plane_resolutions,train_ids = get_scene_configs(cfg.dataset.dir.train)
        val_only_dirs = get_scene_configs(getattr(cfg.dataset.dir,'val',{}))
        downsampling_factors += val_only_dirs[0]
        plane_resolutions += val_only_dirs[2]
        val_ids = val_only_dirs[3]
        val_only_dirs = val_only_dirs[1]
        basedirs = train_dirs+val_only_dirs
        images, poses, render_poses, hwfDs, i_split,scene_ids = [],torch.zeros([0,4,4]),[],[[],[],[],[]],[np.array([]).astype(np.int64) for i in range(3)],[]
        scene_id,val_only_scene_ids,coords_normalization = -1,[],{}
        scene_id_plane_resolution = {}
        # ds_factor_ratio = []
        i_train,i_val = OrderedDict(),OrderedDict()
        font_scale = 4/min(downsampling_factors)
        for basedir,ds_factor,plane_res in zip(tqdm(basedirs,desc='Loading scenes'),downsampling_factors,plane_resolutions):
            scene_id = models.get_scene_id(basedir,ds_factor,plane_res)
            val_only = scene_id not in train_ids
            # if not rep_model_training:
            #     new_scene_id,original_ds = find_scene_match(existing_scenes=existing_LR_scenes,pattern=ds_factor_extraction_pattern(scene_id))
            #     # ds_factor_ratio.append(int(original_ds)/int(find_scene_match([scene_id],ds_factor_extraction_pattern(scene_id))[1]))
            #     scene_id = new_scene_id
            scenes_set.add(scene_id)
            if val_only:    val_only_scene_ids.append(scene_id)
            if planes_model:
                scene_id_plane_resolution[scene_id] = plane_res
            if eval_mode:
                splits2use = ['test']
            else:
                splits2use = ['val'] if val_only else ['train','val']
            cur_images, cur_poses, cur_render_poses, cur_hwfDs, cur_i_split = load_blender_data(
                os.path.join(cfg.dataset.root,basedir),
                testskip=cfg.dataset.testskip,
                downsampling_factor=ds_factor,
                val_downsampling_factor=None,
                cfg=cfg,
                splits2use=splits2use,
            )

            if planes_model and not load_saved_models: # No need to calculate the per-scene normalization coefficients as those will be loaded with the saved model.
                coords_normalization[scene_id] =\
                    calc_scene_box({'camera_poses':cur_poses.numpy()[:,:3,:4],'near':cfg.dataset.near,'far':cfg.dataset.far,'H':cur_hwfDs[0],'W':cur_hwfDs[1],'f':cur_hwfDs[2]},
                        including_dirs=cfg.nerf.use_viewdirs,adjust_elevation_range=getattr(cfg.nerf,'adjust_elevation_range',False))
            if eval_mode:
                i_val[scene_id] = [v+len(images) for v in cur_i_split[2]]
            else:
                i_val[scene_id] = [v+len(images) for v in cur_i_split[1]]
            if not val_only:    i_train[scene_id] = [v+len(images) for v in cur_i_split[0]]
            images += cur_images
            poses = torch.cat((poses,cur_poses),0)
            for i in range(len(hwfDs)):
                hwfDs[i] += cur_hwfDs[i]
            scene_ids += [scene_id for i in cur_images]
        H, W, focal,ds_factor = hwfDs
    else:
        dataset = load_DTU.DVRDataset(config=cfg.dataset,scene_id_func=models.get_scene_id,eval_ratio=0.1,
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
                coords_normalization[dataset.scene_IDs[id]] = calc_scene_box(scene_info,including_dirs=cfg.nerf.use_viewdirs,
                    adjust_elevation_range=getattr(cfg.nerf,'adjust_elevation_range',False))
            basedirs.append(dataset.scene_IDs[id])
        i_val = dataset.i_val
        i_train = dataset.train_ims_per_scene
        # if SR_experiment:
        #     ds_factor_ratio = dataset.ds_factor_ratios
    if hasattr(cfg.dataset,'max_scenes_eval') and not eval_mode:
        i_val,val_only_scene_ids = subsample_dataset(scenes_dict=i_val,max_scenes=cfg.dataset.max_scenes_eval,val_only_scenes=val_only_scene_ids,max_val_only_scenes=cfg.dataset.max_scenes_eval)
    val_ims_per_scene = len(i_val[list(i_val.keys())[0]])
    eval_training_too = not eval_mode
    available_scenes = list(scenes_set)
    if eval_training_too:
        for id in sorted(scenes_set):
            if id not in i_train:    continue
            im_freq = len(i_train[id])//val_ims_per_scene
            if id in i_val.keys(): #Avoid evaluating training images for scenes which were discarded for evaluation due to max_scenes_eval
                i_val[id+'_train'] = [x for i,x in enumerate(i_train[id]) if (i+im_freq//2)%im_freq==0]
    training_scenes = list(i_train.keys())
    scene_coupler = None
    if end2end_training:
        if not rep_model_training:
            available_scenes = []
            for conf,scenes in LR_model_config.dataset.dir.train.items():
                conf=eval(conf)
                for sc in scenes:
                    available_scenes.append(models.get_scene_id(sc,conf[0],(conf[1],conf[2] if len(conf)>2 else conf[1])))

        scene_coupler = models.SceneCoupler(available_scenes,planes_res_level=end2end_training,
            num_planes=getattr(cfg.models.coarse,'num_planes',3)+cfg.nerf.use_viewdirs,training_scenes=training_scenes)
        if end2end_training=='HR_planes':
            for sc in scene_coupler.downsample_couples:
                if sc in training_scenes:
                    i_val[sc+'_HRplane'] = 1*i_val[sc]
                    if sc+'_train' in i_val:
                        i_val[sc+'_HRplane_train'] = 1*i_val[sc+'_train']
        for sc in scene_coupler.downsample_couples.keys():  
            if end2end_training=='HR_planes' and sc in training_scenes:
                scene_id_plane_resolution.pop(scene_coupler.downsample_couples[sc])
            else:
                scene_id_plane_resolution.pop(sc)
            if not load_saved_models:
                # coords_normalization.pop(sc)
                coords_normalization[sc] = 1*coords_normalization[scene_coupler.downsample_couples[sc]]

    evaluation_sequences = list(i_val.keys())
    val_strings = []
    for id in evaluation_sequences:
        bare_id = id.replace('_train','').replace('_HRplane','')
        tags = []
        if id in val_only_scene_ids:    tags.append('blind_validation')
        elif '_train' in id:    tags.append('train_imgs')
        else:   tags.append('validation')
        if bare_id in scene_coupler.downsample_couples.values(): tags.append('LR')
        elif '_HRplane' in id:  tags.append('HRplanes')
        val_strings.append('_'.join(tags))
    # val_strings = ['blind_validation' if id in val_only_scene_ids else 'train_imgs' if '_train' in id else 'validation' for id in evaluation_sequences]
    # val_strings = [s+'_HRplanes' if '_HRplane' in sc else s for s,sc in zip(val_strings,evaluation_sequences)]
    def print_scenes_list(title,scenes):
        print('\n%d %s scenes:'%(len(scenes),title))
        print(scenes)
    if eval_mode:
        print_scenes_list('evaluation',evaluation_sequences)
    else:
        print_scenes_list('training',training_scenes)
        for cat in set(val_strings):
            print_scenes_list('"%s" evaluation'%(cat),[s for i,s in enumerate(evaluation_sequences) if val_strings[i]==cat])
    assert all([val_ims_per_scene==len(i_val[id]) for id in evaluation_sequences]),'Assuming all scenes have the same number of evaluation images'
    i_train = [i for s in i_train.values() for i in s]

    def write_scalar(name,value,iter):
        if not eval_mode:
            writer.add_scalar(name,value,iter)

    def write_image(name,images,text,iter,psnrs=[],fontscale=font_scale):
        if eval_mode:
            SAVE_IMS_ANYWAY = True
            scene_name = evaluation_sequences[int(text)]
            ims_folder_name = os.path.join(results_dir,scene_name)
            if not os.path.isdir(ims_folder_name):  os.mkdir(ims_folder_name)
            eval_name = ('blind_' if 'blind' in name else '')+name.split('_')[-1]
            if eval_mode=='images' or SAVE_IMS_ANYWAY:
                if not os.path.isdir(os.path.join(ims_folder_name,eval_name)):  os.mkdir(os.path.join(ims_folder_name,eval_name))
                for im_num,im in enumerate(images):
                    im_name = '%d%s.png'%(im_num,('_PSNR%.2f'%(psnrs[im_num])).replace('.','_') if len(psnrs)>0 else '')
                    imageio.imwrite(os.path.join(ims_folder_name,eval_name,im_name),np.array(255*torch.clamp(im,0,1).cpu()).astype(np.uint8))
            if eval_mode=='video':
                vid_path = os.path.join(ims_folder_name,'%s_%s_%s.mp4'%(eval_name,scene_name,results_dir.split('/')[-1]))
                imageio.mimwrite(vid_path, [np.array(255*torch.clamp(im,0,1).cpu()).astype(np.uint8) for im in images], fps = 30, macro_block_size = 8)  # pip install imageio-ffmpeg
        else:
            writer.add_image(name,arange_ims(images,text,psnrs=psnrs,fontScale=fontscale),iter)


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
        if getattr(cfg.nerf,'viewdir_mapping',False): assert getattr(cfg.nerf,'use_viewdirs',True)
        if hasattr(cfg.nerf.train,'viewdir_downsampling'):  assert hasattr(cfg.nerf.train,'max_plane_downsampling')
        assert not getattr(cfg.models.coarse,'force_planes_consistency',False),"Depricated"
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
            num_planes_or_rot_mats=getattr(cfg.models.coarse,'num_planes',3),
            viewdir_mapping=getattr(cfg.nerf,'viewdir_mapping',False),
            scene_coupler=scene_coupler,
            point_coords_noise=getattr(cfg.nerf.train,'point_coords_noise',0),
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
                    num_planes_or_rot_mats=model_coarse.rot_mats() if getattr(cfg.models.fine,'use_coarse_planes',False) else getattr(cfg.models.fine,'num_planes',3),
                    viewdir_mapping=getattr(cfg.nerf,'viewdir_mapping',False),
                    plane_loss_w=rgetattr(cfg,'super_resolution.plane_loss_w',None),
                    scene_coupler=scene_coupler,
                    point_coords_noise=getattr(cfg.nerf.train,'point_coords_noise',0),
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

    sr_rendering_loss_w = 1
    trainable_parameters = []
    SR_model = None
    if SR_experiment=="model":
        sr_rendering_loss_w = getattr(cfg.super_resolution,'rendering_loss',1)
        plane_channels = getattr(cfg.models.coarse,'num_plane_channels',48)
        if not eval_mode:   saved_rgb_fine = dict(zip(evaluation_sequences,[{} for i in evaluation_sequences]))
        sf_config = getattr(cfg.super_resolution.model,'scale_factor','linear')
        assert sf_config in ['linear','sqrt','one']
        # assert cfg.super_resolution.get("consistency_loss_w",0)==0 or cfg.super_resolution.get("plane_loss_w",0)==0,'Cannot use both loss terms'
        if sf_config=='one':
            SR_factor = 1
        elif sf_config=='linear':
            SR_factor = int(scene_coupler.ds_factor)
        else:
            SR_factor = int(np.sqrt(scene_coupler.ds_factor))
        SR_model = getattr(models, cfg.super_resolution.model.type)(
            scale_factor=SR_factor,
            in_channels=plane_channels*(1 if getattr(cfg.super_resolution.model,'single_plane',True) else model_fine.num_density_planes),
            out_channels=plane_channels*(1 if getattr(cfg.super_resolution.model,'single_plane',True) else model_fine.num_density_planes),
            sr_config=cfg.super_resolution,
            plane_interp=getattr(cfg.super_resolution,'plane_resize_mode',model_fine.plane_interp),
            detach_LR_planes=getattr(cfg.nerf.train,'detach_LR_planes',True)
        )
        print("SR model: %d parameters"%(num_parameters(SR_model)))
        SR_model.to(device)
        if not eval_mode:   
            trainable_parameters.append({'params':[p for k,p in SR_model.named_parameters() if 'NON_LEARNED' not in k],'lr':getattr(cfg.super_resolution,'lr',cfg.optimizer.lr)})
    if rep_model_training:
        if not eval_mode:
            # Initialize optimizer.
            def collect_params(model,filter='all'):
                assert filter in ['all','planes','non_planes']
                params = []
                if filter!='non_planes':
                    plane_params = dict(sorted([p for p in model.named_parameters() if 'planes_.sc' in p[0] and 'NON_LEARNED' not in p[0]],key=lambda p:p[0]))
                    params.extend(plane_params.values())
                if filter!='planes':
                    params.extend([p[1] for p in model.named_parameters() if all([token not in p[0] for token in ['NON_LEARNED','planes_.sc']])])
                if filter=='planes':
                    return plane_params
                else:
                    return params
            trainable_parameters_ = collect_params(model_coarse,filter='non_planes' if store_planes else 'all')
            if model_fine is not None:
                if planes_model:
                    if cfg.models.fine.type!="use_same":
                        if getattr(cfg.models.fine,'use_coarse_planes',False):
                            trainable_parameters_ += collect_params(model_fine,filter='non_planes')
                        else:
                            trainable_parameters_ += collect_params(model_fine,filter='all')
                else: 
                    trainable_parameters_ += list(model_fine.parameters())
            trainable_parameters.append({'params':trainable_parameters_,'lr':cfg.optimizer.lr})
    if not eval_mode:
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            trainable_parameters, lr=cfg.optimizer.lr
        )

    # Load an existing checkpoint, if a path is specified.
    start_i,eval_counter = 0,0
    models2save = (['representation'] if rep_model_training else [])+(['SR'] if SR_experiment=='model' else [])
    best_saved,last_saved = (0,np.finfo(np.float32).max),dict(zip(models2save,[[] for i in models2save]))
    if load_saved_models:
        if SR_experiment:
            if not rep_model_training:
                checkpoint = find_latest_checkpoint(cfg.models.path,sr=False)
                print("Using LR model %s"%(checkpoint))
            if SR_experiment=="model" and os.path.exists(configargs.load_checkpoint):
                assert os.path.isdir(configargs.load_checkpoint)
                # SR_model_checkpoint = os.path.join(configargs.load_checkpoint,sorted([f for f in os.listdir(configargs.load_checkpoint) if "SR_checkpoint" in f and f[-5:]==".ckpt"],key=lambda x:int(x[len("SR_checkpoint"):-5]))[-1])
                SR_model_checkpoint = find_latest_checkpoint(configargs.load_checkpoint,sr=True)
                start_i = int(SR_model_checkpoint.split('/')[-1][len('SR_checkpoint'):-len('.ckpt')])+1
                if 'eval_counter' in torch.load(SR_model_checkpoint).keys(): eval_counter = torch.load(SR_model_checkpoint)['eval_counter']
                if 'best_saved' in torch.load(SR_model_checkpoint).keys(): best_saved = torch.load(SR_model_checkpoint)['best_saved']
                print(("Using" if eval_mode else "Resuming training of")+" SR model %s"%(SR_model_checkpoint))
                saved_config_dict = get_config(os.path.join(configargs.load_checkpoint,"config.yml"))
                config_diffs = DeepDiff(saved_config_dict,cfg)
                for diff in [config_diffs[ch_type] for ch_type in ['dictionary_item_removed','dictionary_item_added','values_changed'] if ch_type in config_diffs]:
                    print(diff)

                SR_model.load_state_dict(torch.load(SR_model_checkpoint)["SR_model"])
        if rep_model_training:
            checkpoint = find_latest_checkpoint(configargs.load_checkpoint,sr=False)
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

        def load_saved_parameters(model,saved_params,reduced_set=False):
            mismatch = model.load_state_dict(saved_params,strict=False)
            # assert (len(mismatch.missing_keys)==0 or (store_planes and all(['planes_.sc' in k for k in mismatch.missing_keys])))\
            allowed_missing = []
            if store_planes:    allowed_missing.append('planes_.sc')
            if reduced_set: allowed_missing.append('rot_mats')
            assert (len(mismatch.missing_keys)==0 or all([any([tok in k for tok in allowed_missing]) for k in mismatch.missing_keys]))\
                and all(['planes_.sc' in k for k in mismatch.unexpected_keys])
            if planes_model and not store_planes:
                model.box_coords = checkpoint["coords_normalization"]

        checkpoint["model_coarse_state_dict"] = model_coarse.rot_mat_backward_support(checkpoint["model_coarse_state_dict"])
        load_saved_parameters(model_coarse,checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            load_saved_parameters(model_fine,checkpoint["model_fine_state_dict"],reduced_set=True)
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
            model_coarse.assign_SR_model(SR_model,SR_viewdir=cfg.super_resolution.SR_viewdir,save_interpolated=not save_RAM_memory,
                set_planes=not store_planes,plane_dropout=getattr(cfg.super_resolution,'plane_dropout',0),
                single_plane=getattr(cfg.super_resolution.model,'single_plane',True))
        else:
            assert getattr(cfg.super_resolution.training,'loss','both')=='fine'
            if not rep_model_training:  model_coarse.optional_no_grad = torch.no_grad
        model_fine.assign_SR_model(SR_model,SR_viewdir=cfg.super_resolution.SR_viewdir,save_interpolated=not save_RAM_memory,
            set_planes=not store_planes,plane_dropout=getattr(cfg.super_resolution,'plane_dropout',0),
            single_plane=getattr(cfg.super_resolution.model,'single_plane',True))
    if store_planes:
        planes_folder = os.path.join(LR_model_folder if not rep_model_training else logdir,'planes')
        if eval_mode: assert os.path.isdir(planes_folder)
        if os.path.isdir(planes_folder):
            resave_checkpoint = False
            param_files_list = [f for f in glob.glob(planes_folder+'/*') if '.par' in f]
            broken_files = [f for f in param_files_list if '.par_temp' in f and time.time()-os.path.getmtime(f)>60]
            if len(param_files_list)>0 and (any(['DTU' in f.split('/')[-1] for f in param_files_list]) or 'coords_normalization' not in torch.load(param_files_list[0]).keys()):
                if any(['DTU' in f.split('/')[-1] for f in param_files_list]):
                    from scripts import convert_DTU_scene_names
                    name_mapping = convert_DTU_scene_names.name_mapping()
                checkpoint_name = find_latest_checkpoint(cfg.models.path if not rep_model_training else configargs.load_checkpoint,sr=False)
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
            elif load_saved_models and hasattr(checkpoint_config.models.coarse,'plane_resolutions'):
                for f in tqdm(param_files_list,desc='!!! Converting saved planes to new name convention !!!',):
                    cur_scene_name = f.split('/')[-1][len('coarse_'):-4]
                    params = torch.load(f)
                    if '_PlRes' not in cur_scene_name:
                        new_scene_id = models.get_scene_id(cur_scene_name,ds_factor=getattr(checkpoint_config.dataset,'downsampling_factor',1),
                            plane_res=(checkpoint_config.models.coarse.plane_resolutions,checkpoint_config.models.coarse.viewdir_plane_resolution))
                        params['params'] = torch.nn.ParameterDict([(k.replace(cur_scene_name,new_scene_id),v) for k,v in params['params'].items()])
                    torch.save(params,f.replace(cur_scene_name,new_scene_id) if '_PlRes' not in cur_scene_name else f)
                    if '_PlRes' not in cur_scene_name:
                        os.remove(f)
                checkpoint_config.models.coarse.pop('plane_resolutions',None)
                checkpoint_config.models.coarse.pop('viewdir_plane_resolution',None)
                checkpoint_config.dataset.pop('downsampling_factor',None)
                checkpoint_name = os.path.join(cfg.models.path if not rep_model_training else configargs.load_checkpoint,'config.yml')
                with open(checkpoint_name, "w") as f:
                    f.write(checkpoint_config.dump())  # cfg, f, default_flow_style=False)
            if resave_checkpoint:
                assert all([s in getattr(cfg.dataset,'excluded_scenes',[]) for s in checkpoint['coords_normalization'].keys()])
                checkpoint.pop('coords_normalization',None)
                torch.save(checkpoint,checkpoint_name)
            if any(broken_files):
                # Salvaging a planes file whose saving was interrupted in the middle:
                assert len(broken_files)==1,'Not expecting to have more than 1 broken file'
                print('!!! Salvaging a broken planes file: %s !!!'%(broken_files[0].split('/')[-1]))
                copyfile(broken_files[0],broken_files[0].replace('.par_temp','.par'))
                os.remove(broken_files[0])
        else:
            os.mkdir(planes_folder)
        scenes_cycle_counter = Counter()
        optimize_planes = rep_model_training and not eval_mode
        planes_opt = models.PlanesOptimizer(optimizer_type=cfg.optimizer.type,
            scene_id_plane_resolution=scene_id_plane_resolution,options=cfg.nerf.train.store_planes,save_location=planes_folder,
            lr=getattr(cfg.optimizer,'planes_lr',cfg.optimizer.lr),model_coarse=model_coarse,model_fine=model_fine,
            use_coarse_planes=getattr(cfg.models.fine,'use_coarse_planes',False),
            init_params=not load_saved_models,optimize=optimize_planes,training_scenes=training_scenes,
            coords_normalization=None if load_saved_models else coords_normalization,
            do_when_reshuffling=lambda:scenes_cycle_counter.step(print_str='Number of scene cycles performed: '),
            STD_factor=getattr(cfg.nerf.train,'STD_factor',0.1),
            available_scenes=available_scenes,
        )

    if planes_model:    assert not (hasattr(cfg.models.coarse,'plane_resolutions') or hasattr(cfg.models.coarse,'viewdir_plane_resolution')),'Depricated.'
    if SR_experiment=="model" and getattr(cfg.super_resolution,'input_normalization',False) and not os.path.exists(configargs.load_checkpoint):
        #Initializing a new SR model that uses input normalization
        SR_model.normalization_params(planes_opt.get_plane_stats(viewdir=getattr(cfg.super_resolution,'SR_viewdir',False),single_plane=SR_model.single_plane))

    downsampling_offset = lambda ds_factor: (ds_factor-1)/(2*ds_factor)
    saved_target_ims = dict(zip(set(val_strings),[set() for i in set(val_strings)]))#set()
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
            if eval_mode:
                img_indecis = [v for i,v in enumerate(i_val.values())]
                eval_cycles = len(i_val)
            else:
                val_ind = lambda dummy: eval_counter%val_ims_per_scene
                img_indecis = [[v[val_ind(val_strings[i])] for i,v in enumerate(i_val.values())]]
                eval_cycles = 1

            record_fine = True
            for eval_cycle in range(eval_cycles):
                coarse_loss,fine_loss,loss,psnr,rgb_coarse,rgb_fine,rgb_SR,target_ray_values,consistency_loss,planes_SR_loss = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
                for eval_num,img_idx in enumerate(tqdm(img_indecis[eval_cycle],
                    desc='Evaluating '+('scene (%d/%d): %s'%(eval_cycle+1,eval_cycles,evaluation_sequences[eval_cycle]) if eval_mode else 'scenes'))):
                    if eval_mode:
                        val_ind = lambda dummy: eval_cycle
                        scene_num = eval_cycle
                    else:
                        scene_num = eval_num
                    sr_scene = SR_experiment and scene_ids[img_idx] in scene_coupler.downsample_couples and '_HRplane' not in evaluation_sequences[scene_num]
                    scene_coupler.toggle_used_planes_res(HR='_HRplane' in evaluation_sequences[scene_num])
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
                    if sr_scene:
                        if SR_experiment=="refine" or planes_model:
                            rgb_SR_ = 1*rgb_fine_
                            rgb_SR_coarse_ = 1*rgb_coarse_
                            if eval_mode or val_ind(val_strings[scene_num]) not in saved_rgb_fine[evaluation_sequences[scene_num]]:
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
                                if not eval_mode and not rep_model_training:
                                    saved_rgb_fine[evaluation_sequences[scene_num]][val_ind(val_strings[scene_num])] = 1*rgb_fine_.detach()
                            else:
                                record_fine = False
                                rgb_fine_ = 1*saved_rgb_fine[evaluation_sequences[scene_num]][val_ind(val_strings[scene_num])]
                        fine_loss[val_strings[scene_num]].append(img2mse(rgb_fine_[..., :3], img_target[..., :3]).item())
                        loss[val_strings[scene_num]].append(img2mse(rgb_SR_[..., :3], img_target[..., :3]).item())
                        consistency_loss_ = SR_model.return_consistency_loss()
                        if consistency_loss_ is not None:
                            consistency_loss[val_strings[scene_num]].append(consistency_loss_.item())
                        planes_SR_loss_ = model_fine.return_planes_SR_loss()
                        if planes_SR_loss_ is not None:
                            planes_SR_loss[val_strings[scene_num]].append(planes_SR_loss_.item())

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
                    if planes_model and SR_model is not None:
                        last_scene_eval = img_idx==img_indecis[eval_cycle][-1]
                        if (store_planes and (not eval_mode or last_scene_eval)) or save_RAM_memory:
                            SR_model.clear_SR_planes(all_planes=store_planes)
                            scene_coupler.downsampled_planes = {}
                SAVE_COARSE_IMAGES = False
                cur_val_sets = [val_strings[eval_cycle]] if eval_mode else set(val_strings)
                for val_set in cur_val_sets:
                    sr_scene_inds = [i for i,im in enumerate(rgb_SR[val_set]) if im is not None]
                    if len(sr_scene_inds)>0: #SR_experiment and '_HRplanes' not in val_set:
                        write_scalar("%s/SR_psnr_gain"%(val_set),
                            np.mean([psnr[val_set][i]-mse2psnr(fine_loss[val_set][i]) for i in sr_scene_inds]), iter)
                        write_image(name="%s/rgb_SR"%(val_set),
                            images=[torch.zeros_like(rgb_fine[val_set][i]) if im is None else im for i,im in enumerate(rgb_SR[val_set])],
                            text=str(val_ind(val_set)),psnrs=[None if im is None else psnr[val_set][i] for i,im in enumerate(rgb_SR[val_set])],
                            iter=iter,fontscale=font_scale/scene_coupler.ds_factor if '_LR' in val_set else font_scale)
                        if val_set in consistency_loss:
                            write_scalar("%s/SR_consistency"%(val_set), np.mean(consistency_loss[val_set]), iter)
                        if val_set in planes_SR_loss:
                            write_scalar("%s/SR_planes"%(val_set), np.mean(planes_SR_loss[val_set]), iter)
                    write_scalar("%s/fine_loss"%(val_set), np.mean(fine_loss[val_set]), iter)
                    write_scalar("%s/fine_psnr"%(val_set), np.mean([mse2psnr(l) for l in fine_loss[val_set]]), iter)
                    write_scalar("%s/loss"%(val_set), np.mean(loss[val_set]), iter)
                    write_scalar("%s/psnr"%(val_set), np.mean(psnr[val_set]), iter)
                    if len(coarse_loss[val_set])>0:
                        write_scalar("%s/coarse_loss"%(val_set), np.mean(coarse_loss[val_set]), iter)
                    if len(rgb_fine[val_set])>0:
                        if record_fine:
                            write_scalar("%s/fine_loss"%(val_set), np.mean(fine_loss[val_set]), val_ind(val_set) if not rep_model_training else iter)
                            write_image(name="%s/rgb_fine"%(val_set),images=rgb_fine[val_set],text=str(val_ind(val_set)),
                                psnrs=[mse2psnr(l) for l in fine_loss[val_set]],iter=val_ind(val_set) if not rep_model_training else iter,
                                fontscale=font_scale/scene_coupler.ds_factor if '_LR' in val_set else font_scale)
                    if SAVE_COARSE_IMAGES:
                        write_image(name="%s/rgb_coarse"%(val_set),images=rgb_coarse[val_set],text=str(val_ind(val_set)),
                            psnrs=[mse2psnr(l) for l in coarse_loss[val_set]],iter=iter,
                            fontscale=font_scale/scene_coupler.ds_factor if '_LR' in val_set else font_scale)
                    if not eval_mode and val_ind(val_set) not in saved_target_ims[val_set]:
                        write_image(name="%s/img_target"%(val_set),images=target_ray_values[val_set],
                            text=str(val_ind(val_set)),iter=val_ind(val_set),fontscale=font_scale/scene_coupler.ds_factor if '_LR' in val_set else font_scale)
                        saved_target_ims[val_set].add(val_ind(val_set))
                    if not eval_mode:
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
        if rep_model_training:
            model_coarse.train()
            if model_fine:
                model_fine.train()
            if getattr(cfg.nerf.train,'max_plane_downsampling',1)>1:
                plane_downsampling = 1 if np.random.uniform()>0.5 else 1+np.random.randint(cfg.nerf.train.max_plane_downsampling)
                model_coarse.use_downsampled_planes(plane_downsampling)
                model_fine.use_downsampled_planes(plane_downsampling)

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        if store_planes:
            img_idx = np.random.choice(available_train_inds)
        else:
            img_idx = np.random.choice(i_train)

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
        if len(scene_coupler.HR_planes)>0 and scene_ids[img_idx] in scene_coupler.downsample_couples:
            scene_coupler.toggle_used_planes_res(np.random.uniform()>=0.5)
        if hasattr(model_fine,'SR_model'):
            if end2end_training:
                model_fine.assign_LR_planes(scene=scene_ids[img_idx])
            # Handling SR planes dropout:
            model_fine.SR_planes2drop = [] if np.random.uniform()>=model_fine.SR_plane_dropout\
                 else sorted(list(np.random.choice(range(model_fine.num_density_planes),size=np.random.randint(1,model_fine.num_density_planes),replace=False)))
        else: # Handling representation model planes dropout:
            planes2drop = [] if np.random.uniform()>=getattr(cfg.nerf.train,'plane_dropout',0)\
                 else sorted(list(np.random.choice(range(model_fine.num_density_planes),size=np.random.randint(1,model_fine.num_density_planes-3),replace=False)))
            model_fine.planes2drop = planes2drop
            model_coarse.planes2drop = planes2drop
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

        coarse_loss,fine_loss = None,None
        if sr_rendering_loss_w is not None:
            if rep_model_training or getattr(cfg.super_resolution.training,'loss','both')!='fine':
                coarse_loss = torch.nn.functional.mse_loss(
                    rgb_coarse[..., :3], target_ray_values[..., :3]
                )
            if rgb_fine is not None and (rep_model_training or getattr(cfg.super_resolution.training,'loss','both')!='coarse'):
                fine_loss = torch.nn.functional.mse_loss(
                    rgb_fine[..., :3], target_ray_values[..., :3]
                )
        rendering_loss = ((coarse_loss if coarse_loss is not None else 0.0) + (fine_loss if fine_loss is not None else 0.0))
        loss = sr_rendering_loss_w*rendering_loss
        psnr = None
        if isinstance(loss,torch.Tensor):
            write_scalar("train/loss", loss.item(), iter)
            psnr = mse2psnr(loss.item())
            write_scalar("train/psnr", psnr, iter)

        if SR_experiment=="model" and planes_model:
            SR_consistency_loss = SR_model.return_consistency_loss()
            if SR_consistency_loss is not None:
                write_scalar("train/inconsistency", SR_consistency_loss.item(), iter)
                loss += cfg.super_resolution.consistency_loss_w*SR_consistency_loss
            planes_SR_loss = model_fine.return_planes_SR_loss()
            if planes_SR_loss is not None:
                write_scalar("train/planes_SR", planes_SR_loss.item(), iter)
                loss += cfg.super_resolution.plane_loss_w*planes_SR_loss

        loss.backward()
        new_drawn_scenes = None
        if store_planes:
            new_drawn_scenes = planes_opt.step(reload_planes_4_SR=end2end_training)
        if last_v_batch_iter:
            optimizer.step()
            # If training an SR model operating on planes, discarding super-resolved planes after updating the model:
            if planes_model and SR_experiment=='model':
                # SR_model.clear_SR_planes(all_planes=end2end_training)
                scene_coupler.downsampled_planes = {}
        # if SR_experiment!="model" or planes_model:
        if coarse_loss is not None:
            write_scalar("train/coarse_loss", coarse_loss.item(), iter)
        if fine_loss is not None:
            write_scalar("train/fine_loss", fine_loss.item(), iter)
            write_scalar("train/fine_psnr", mse2psnr(fine_loss.item()), iter)
        # return (rendering_loss.item() if isinstance(rendering_loss,torch.Tensor) else rendering_loss),psnr,new_drawn_scenes
        return loss.item(),psnr,new_drawn_scenes

    training_time,last_evaluated = 0,1*start_i
    recently_saved, = time.time(),
    eval_loss_since_save,print_cycle_loss,print_cycle_psnr = [],[],[]
    evaluation_time = 0
    jump_start_phase = isinstance(getattr(cfg.nerf.train,'jump_start',False),list) and start_i==0
    if jump_start_phase:
        n_jump_start_scenes = planes_opt.jump_start(config=cfg.nerf.train.jump_start)

    for iter in trange(start_i,cfg.experiment.train_iters):
        # Validation
        if isinstance(cfg.experiment.validate_every,list):
            evaluate_now = evaluation_time<=training_time*cfg.experiment.validate_every[0] or iter-last_evaluated>=cfg.experiment.validate_every[1]
        else:
            evaluate_now = iter % cfg.experiment.validate_every == 0
        evaluate_now |= iter == cfg.experiment.train_iters - 1
        if iter>0:  evaluate_now &= not jump_start_phase
        # print('!!!!!!!!!!WARNING!!!!!!!!!!!')
        # evaluate_now = True
        if evaluate_now:
            last_evaluated = 1*iter
            start_time = time.time()
            loss,psnr = evaluate()
            eval_loss_since_save.extend([v for v in loss['blind_validation' if 'blind_validation' in loss else 'validation']])
            evaluation_time = time.time()-start_time
            if store_planes and not eval_mode:
                if not jump_start_phase or iter==0:
                    planes_opt.draw_scenes(assign_LR_planes=not end2end_training)
                    new_drawn_scenes = planes_opt.cur_scenes
                    if jump_start_phase:    new_drawn_scenes = new_drawn_scenes[:n_jump_start_scenes]
                    available_train_inds = [i for i in i_train if scene_ids[i] in new_drawn_scenes]
            training_time = 0
            eval_counter += 1

        if eval_mode:   break
        # Training:
        start_time = time.time()
        loss,psnr,new_drawn_scenes = train()
        if new_drawn_scenes is not None:
            available_train_inds = [i for i in i_train if scene_ids[i] in new_drawn_scenes]

        print_cycle_loss.append(loss)
        print_cycle_psnr.append(psnr)
        training_time += time.time()-start_time

        if iter % cfg.experiment.print_every == 0 or iter == cfg.experiment.train_iters - 1:
            if iter>0 and jump_start_phase:
                if np.mean(print_cycle_loss)<=cfg.nerf.train.jump_start[1]:
                    jump_start_phase = False
                    new_drawn_scenes = planes_opt.jump_start(on=False)
                    available_train_inds = [i for i in i_train if scene_ids[i] in new_drawn_scenes]
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
        save_now = scenes_cycle_counter.check_and_reset() if (store_planes and rep_model_training) else False
        save_now |= iter % cfg.experiment.save_every == 0 if isinstance(cfg.experiment.save_every,int) else (time.time()-recently_saved)/60>cfg.experiment.save_every
        save_now |= iter == cfg.experiment.train_iters - 1
        save_now &= iter>0
        if save_now:
            save_as_best = False
            if len(eval_loss_since_save)>0:
                recent_loss_avg = np.mean(eval_loss_since_save)
                if recent_loss_avg<best_saved[1]:
                    best_saved = (iter,recent_loss_avg)
                    save_as_best = True
            if save_as_best:
                print("================Saving new best checkpoint at iteration %d, with average evaluation loss %.3e====================="%(best_saved[0],best_saved[1]))
            else:
                print("================Best checkpoint is still %d, with average evaluation loss %.3e (recent average is %.3e)====================="%(best_saved[0],best_saved[1],recent_loss_avg))
            recently_saved = time.time()
            eval_loss_since_save = []
            for model2save in models2save:
                model_filename = '%scheckpoint'%('SR_' if model2save=='SR' else '')
                checkpoint_dict = {
                    "iter": iter,
                    "eval_counter": eval_counter,
                    # "optimizer_state_dict": optimizer.state_dict(),
                    "best_saved":best_saved,
                }
                if model2save=="SR":
                    checkpoint_dict.update({"SR_model":SR_model.state_dict()})
                else:
                    tokens_2_exclude = ['planes_.','SR_model']
                    checkpoint_dict.update({"model_coarse_state_dict": model_coarse.state_dict()})
                    if model_fine:  checkpoint_dict.update({"model_fine_state_dict": model_fine.state_dict()})
                    if planes_model:
                        if store_planes or getattr(cfg.models.fine,'use_coarse_planes',False):
                            checkpoint_dict["model_fine_state_dict"] = OrderedDict([(k,v) for k,v in checkpoint_dict["model_fine_state_dict"].items() if all([token not in k for token in (tokens_2_exclude+['rot_mats'])])])
                        if store_planes:
                            checkpoint_dict["model_coarse_state_dict"] = OrderedDict([(k,v) for k,v in checkpoint_dict["model_coarse_state_dict"].items() if all([token not in k for token in tokens_2_exclude])])
                            planes_opt.save_params(to_checkpoint=False)
                        else:
                            checkpoint_dict.update({"coords_normalization": model_fine.box_coords})

                ckpt_name = os.path.join(logdir, model_filename + str(iter).zfill(5) + ".ckpt")
                torch.save(checkpoint_dict,ckpt_name,)
                if len(last_saved[model2save])>0:
                    os.remove(last_saved[model2save].pop(0))
                last_saved[model2save].append(ckpt_name)
                if save_as_best:
                    best_ckpt_name = os.path.join(logdir, model_filename+".best_ckpt")
                    if os.path.exists(best_ckpt_name):
                        copyfile(best_ckpt_name,best_ckpt_name.replace("_ckpt","_ckpt_old"))
                    torch.save(checkpoint_dict,best_ckpt_name,)
                    if os.path.exists(best_ckpt_name.replace("_ckpt","_ckpt_old")):
                        os.remove(best_ckpt_name.replace("_ckpt","_ckpt_old"))
                del checkpoint_dict

    print("Done!")



if __name__ == "__main__":
    main()
