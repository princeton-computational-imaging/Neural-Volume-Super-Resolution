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
from load_blender import load_blender_data,BlenderDataset
import load_DTU
from nerf_helpers import *
from train_utils import eval_nerf, run_one_iter_of_nerf,find_latest_checkpoint
from mip import IntegratedPositionalEncoding
from deepdiff import DeepDiff
from copy import deepcopy
from shutil import copyfile
import socket

DEBUG_MODE = False #False #True

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
        dataset_config4eval = cfg.dataset.dir.val
        config_file = os.path.join(cfg.experiment.logdir, cfg.experiment.id,"config.yml")
        results_dir = os.path.join(configargs.results_path, cfg.experiment.id)
        white_bg = getattr(cfg.nerf.validation,'white_background',False)
        if not os.path.isdir(results_dir):  os.mkdir(results_dir)
        print('Evaluation outputs will be saved into %s'%(results_dir))
        cfg = get_config(config_file)
        cfg.dataset.dir.val = dataset_config4eval
        cfg.nerf.train.pop('zero_mean_planes_w',None)
        if hasattr(cfg,'super_resolution'): 
            cfg.super_resolution.pop('consistency_loss_w',None)
            cfg.super_resolution.pop('plane_loss_w',None)
        cfg.nerf.validation.white_background = white_bg
    print('Using configuration file %s'%(config_file))
    print(("Evaluating" if eval_mode else "Running") + " experiment %s"%(cfg.experiment.id))
    SR_experiment = None
    if "super_resolution" in cfg:
        SR_experiment = "model" #if "model" in cfg.super_resolution.keys() else "refine"
    what2train = getattr(cfg.nerf.train,'what',[])
    assert all([m in ['LR_planes','HR_planes','decoder','SR'] for m in what2train])
    # if eval_mode:   assert len(what2train)==0,'Make sure this always holds'
    # end2end_training = len(what2train)==3 #getattr(cfg.nerf.train,'train_end2end',False)
    # train_planes_only = getattr(cfg.nerf.train,'planes_only',False)
    # assert not (train_planes_only and end2end_training)
    # assert end2end_training in ['HR_planes','LR_planes',False]
    # rep_model_training = (not (SR_experiment or train_planes_only)) or end2end_training
    planes_model = not hasattr(cfg.models,'coarse') or cfg.models.coarse.type=="TwoDimPlanesModel"
    rep_model_training = 'decoder' in what2train

    if planes_model and (not rep_model_training or hasattr(cfg.models,'path')):
        LR_model_folder = cfg.models.path
        if os.path.isfile(LR_model_folder):   LR_model_folder = "/".join(LR_model_folder.split("/")[:-1])
        LR_model_config = get_config(os.path.join(LR_model_folder,"config.yml"))
        set_config_defaults(source=LR_model_config.models,target=cfg.models)
        existing_LR_scenes = [f.split('.')[0][len('coarse_'):] for f in os.listdir(os.path.join(LR_model_folder,'planes')) if '.par' in f]
    else:
        existing_LR_scenes = None

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
    resume_experiment = os.path.exists(configargs.load_checkpoint)
    if configargs.load_checkpoint!='':
        assert resume_experiment
    if not eval_mode:   writer = SummaryWriter(logdir)

    load_saved_models = hasattr(cfg.models,'path') or resume_experiment
    # init_new_scenes = not load_saved_models or (train_planes_only and not resume_experiment)
    init_new_scenes = not resume_experiment and any([m in ['LR_planes','HR_planes'] for m in what2train])


    # images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    # Load dataset
    # dataset_type = getattr(cfg.dataset,'type','synt')
    # assert dataset_type in['synt','DTU','llff']

    if getattr(cfg.nerf.train,'plane_dropout',0)>0: assert cfg.models.coarse.proj_combination=='avg' and cfg.models.coarse.num_planes>3
    # assert not (getattr(cfg.nerf.train,'random_target_bg_color',False) and getattr(cfg.nerf.train,'mask_background',False))

    sr_val_scenes_with_LR = getattr(cfg.nerf.train,'sr_val_scenes_with_LR',False)
    dataset = BlenderDataset(config=cfg.dataset,scene_id_func=models.get_scene_id,add_val_scene_LR=sr_val_scenes_with_LR,
        eval_mode=eval_mode,scene_norm_coords=cfg.nerf if init_new_scenes else None)
    font_scale = 4/min(dataset.downsampling_factors)
    scene_ids = dataset.per_im_scene_id
    i_val = dataset.i_val
    i_train = dataset.i_train
    scenes_set = dataset.scenes_set
    coords_normalization = dataset.coords_normalization
    scene_id_plane_resolution = dataset.scene_id_plane_resolution
    val_only_scene_ids = dataset.val_only_scene_ids
    CURREPTED_VAL_PATCH = True
    if CURREPTED_VAL_PATCH:
        i_val = OrderedDict([(k,v) for k,v in i_val.items() if search('^(\d)+_',k) is None])
    if hasattr(cfg.dataset,'max_scenes_eval') and not eval_mode:
        i_val,val_only_scene_ids = subsample_dataset(scenes_dict=i_val,max_scenes=cfg.dataset.max_scenes_eval,val_only_scenes=val_only_scene_ids,max_val_only_scenes=cfg.dataset.max_scenes_eval)
    if not eval_mode:
        # val_ims_per_scene = len(i_val[list(i_val.keys())[0]])
        val_ims_per_scene = [len(v) for v in i_val.values()]
        assert all([max(val_ims_per_scene)%v==0 for v in val_ims_per_scene]),"Need to be able to repeat scene eval sets to have the same number of eval images for all scnenes."
        val_ims_per_scene = max(val_ims_per_scene)
        i_val = OrderedDict([(k,val_ims_per_scene//len(v)*v) for k,v in i_val.items()])
    eval_training_too = not eval_mode
    available_scenes = list(scenes_set)
    if eval_training_too:
        temp = list(i_val.keys())
        for id in temp:
            if id not in i_train:    continue
            im_freq = len(i_train[id])//val_ims_per_scene
            if id in i_val.keys(): #Avoid evaluating training images for scenes which were discarded for evaluation due to max_scenes_eval
                i_val[id+'_train'] = [i_train[id][i] for i in sorted([(i+im_freq//2)%len(i_train[id]) for i in  np.unique(np.round(np.linspace(0,len(i_train[id])-1,val_ims_per_scene)).astype(int))])]
    training_scenes = list(i_train.keys())
    # planes_updating = rep_model_training or train_planes_only
    planes_updating = any(['planes' in m for m in what2train])
    if planes_model and not planes_updating:
        available_scenes = []
        for conf,scenes in [c for p in LR_model_config.dataset.dir.values() for c in p.items()]:
            conf=eval(conf)
            for sc in interpret_scene_list(scenes):
                available_scenes.append(models.get_scene_id(sc,conf[0],(conf[1],conf[2] if len(conf)>2 else conf[1])))

    scene_coupler = models.SceneCoupler(list(set(available_scenes+val_only_scene_ids)),HR_planes='HR_planes' in what2train,
        num_pos_planes=getattr(cfg.models.coarse,'num_planes',3) if planes_model else 0,viewdir_plane=cfg.nerf.use_viewdirs,training_scenes=training_scenes,
        multi_im_res=SR_experiment=='model' or not planes_model)
    if planes_model:
        # if 'decoder' in what2train and 'SR' not in what2train:
        #     assert len(scene_coupler.downsample_couples)==0
        if 'SR' in what2train and len(what2train)==1:
            assert all([sc not in scene_coupler.downsample_couples.values() for sc in training_scenes]),'Why train on LR scenes when training only the SR model?'
        # if rep_model_training:
        #     if not end2end_training:
        #         assert len(scene_coupler.downsample_couples)==0
        #         # assert all([sc not in scene_coupler.downsample_couples.values() for sc in training_scenes]),'Why train on LR scenes when training only the SR model?'
        # else:
        #     assert train_planes_only or all([sc not in scene_coupler.downsample_couples.values() for sc in training_scenes]),'Why train on LR scenes when training only the SR model?'
    if SR_experiment=='model':
        if 'HR_planes' in what2train:
        # if end2end_training=='HR_planes':
            for sc in scene_coupler.downsample_couples:
                if sc in training_scenes:
                    i_val[sc+'_HRplane'] = 1*i_val[sc]
                    if sc+'_train' in i_val:
                        i_val[sc+'_HRplane_train'] = 1*i_val[sc+'_train']
        # for sc in scene_coupler.downsample_couples.keys():  
        for sc in scenes_set:  
            if sc not in scene_coupler.downsample_couples:  continue
            if init_new_scenes:
                # if dataset_type=='llff':
                if dataset.scene_types[sc]=='llff':
                    print("!!! Warning: Using val images to determine coord normalization on real images !!!")
                    coords_normalization[sc] = torch.stack([coords_normalization[sc],coords_normalization[scene_coupler.downsample_couples[sc]]],-1)
                    coords_normalization[sc] = torch.stack([torch.min(coords_normalization[sc][0],-1)[0],torch.max(coords_normalization[sc][1],-1)[0]],0)
                    coords_normalization[scene_coupler.downsample_couples[sc]] = 1*coords_normalization[sc]
                else:
                    coords_normalization[sc] = 1*coords_normalization[scene_coupler.downsample_couples[sc]]
            if 'HR_planes' in what2train and sc in training_scenes:
            # if end2end_training=='HR_planes' and sc in training_scenes:
                if scene_coupler.downsample_couples[sc] not in scene_id_plane_resolution:   continue
                scene_id_plane_resolution.pop(scene_coupler.downsample_couples[sc])
            else:
                if sc not in scene_id_plane_resolution:   continue
                scene_id_plane_resolution.pop(sc)

    evaluation_sequences = list(i_val.keys())
    val_strings = []
    ASSUME_LR_IF_NO_COUPLES = True
    ASSUME_LR_IF_NO_COUPLES = ASSUME_LR_IF_NO_COUPLES and (len(scene_coupler.downsample_couples)==0 and SR_experiment)
    for id in evaluation_sequences:
        bare_id = id.replace('_train','').replace('_HRplane','')
        tags = []
        if id in val_only_scene_ids:    tags.append('blind_validation')
        elif '_train' in id:    tags.append('train_imgs')
        else:   tags.append('validation')
        if '##Gauss' in bare_id:    tags.append('Gauss')
        if bare_id in scene_coupler.downsample_couples.values() or ASSUME_LR_IF_NO_COUPLES: tags.append('LR')
        elif bare_id in scene_coupler.HR_planes_LR_ims_scenes:  tags.append('downscaled')
        elif '_HRplane' in id:  tags.append('HRplanes')
        if dataset.scene_types[bare_id]=='llff': tags.append('real')
        val_strings.append('_'.join(tags))
    important_loss_terms = [tag for tag in val_strings if 'blind_validation' in tag]
    if len(important_loss_terms)==0:        important_loss_terms = [tag for tag in val_strings if ('validation' in tag and '_LR' not in tag)]
    if len(important_loss_terms)==0:        important_loss_terms = [tag for tag in val_strings if 'validation' in tag]
    important_loss_terms = set(important_loss_terms)
    def print_scenes_list(title,scenes):
        print('\n%d %s scenes:'%(len(scenes),title))
        print(scenes)
    experiment_info = dict()
    if eval_mode:
        print_scenes_list('evaluation',evaluation_sequences)
    else:
        print_scenes_list('training',training_scenes)
        for cat in set(val_strings):
            print_scenes_list('"%s" evaluation'%(cat),[s for i,s in enumerate(evaluation_sequences) if val_strings[i]==cat])
        assert all([val_ims_per_scene==len(i_val[id]) for id in evaluation_sequences]),'Assuming all scenes have the same number of evaluation images'

        running_mean_logs = ['psnr','SR_psnr_gain','planes_SR','zero_mean_planes_loss','fine_loss','fine_psnr','loss','coarse_loss','inconsistency']
        experiment_info['running_scores'] = dict([(score,dict([(cat,deque(maxlen=len(training_scenes) if cat=='train' else val_ims_per_scene)) for cat in list(set(val_strings))+['train']])) for score in running_mean_logs])
    image_sampler = ImageSampler(i_train,dataset.scene_probs)

    def write_scalar(name,new_value,iter):
        RUNNING_MEAN = True
        if not eval_mode:
            val_set,metric = name.split('/')
            experiment_info['running_scores'][metric][val_set].append(new_value)
            writer.add_scalar(name,np.mean(experiment_info['running_scores'][metric][val_set]) if RUNNING_MEAN else new_value,iter)

    def write_image(name,images,text,iter,psnrs=[],fontscale=font_scale,psnr_gains=[]):
        if eval_mode:
            SAVE_IMS_ANYWAY = True
            scene_name = evaluation_sequences[int(text)]
            white_bg = getattr(cfg.nerf.validation,'white_background',False)
            ims_folder_name = os.path.join(results_dir,('WB_' if white_bg else '')+scene_name)
            if not os.path.isdir(ims_folder_name):  os.mkdir(ims_folder_name)
            eval_name = ('blind_' if 'blind' in name else '')+name.split('_')[-1]
            if eval_mode=='images' or SAVE_IMS_ANYWAY:
                if len(psnr_gains)==0:   psnr_gains = 1*psnrs
                if not os.path.isdir(os.path.join(ims_folder_name,eval_name)):  os.mkdir(os.path.join(ims_folder_name,eval_name))
                for im_num,im in enumerate(images):
                    im_name = '%d%s.png'%(im_num,('_PSNR%.2f'%(psnr_gains[im_num])).replace('.','_') if len(psnr_gains)>0 else '')
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
        PLANE_STATS = True and DEBUG_MODE
        if PLANE_STATS: print("!!!!!!!! WARNING: Saving plane statistics !!!!!!!!")
        if getattr(cfg.nerf,'viewdir_mapping',False): assert getattr(cfg.nerf,'use_viewdirs',True)
        if hasattr(cfg.nerf.train,'viewdir_downsampling'):  assert hasattr(cfg.nerf.train,'max_plane_downsampling')
        assert not getattr(cfg.models.coarse,'force_planes_consistency',False),"Depricated"
        store_planes = hasattr(cfg.nerf.train,'store_planes')
        if store_planes:    assert not getattr(cfg.nerf.train,'save_GPU_memory',False),'I think this is unnecessary.'
        model_coarse = models.TwoDimPlanesModel(
            use_viewdirs=cfg.nerf.use_viewdirs,
            coords_normalization = None if store_planes else coords_normalization,
            dec_density_layers=getattr(cfg.models.coarse,'dec_density_layers',4),
            dec_rgb_layers=getattr(cfg.models.coarse,'dec_rgb_layers',4),
            dec_channels=getattr(cfg.models.coarse,'dec_channels',128),
            skip_connect_every=getattr(cfg.models.coarse,'skip_connect_every',None),
            num_plane_channels=getattr(cfg.models.coarse,'num_plane_channels',48),
            num_viewdir_plane_channels=getattr(cfg.models.coarse,'num_viewdir_plane_channels',None),
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
            zero_mean_planes_w=getattr(cfg.nerf.train,'zero_mean_planes_w',None),
            dec_dss_layers=getattr(cfg.models.coarse,'dec_dss_layers',None),
            plane_stats=PLANE_STATS,
        )
        
    else:
        store_planes = False
        # Initialize a coarse-resolution model.
        if cfg.nerf.encode_position_fn=="mip" and cfg.models.coarse.include_input_xyz:
            cfg.models.coarse.include_input_xyz = False
            cfg.models.fine.include_input_xyz = False
            print("!!! Warning: Not including xyz in model's input since Mip-NeRF does not use the input xyz !!!")
        # assert not (cfg.nerf.encode_position_fn=="mip" and cfg.models.coarse.include_input_xyz),"Mip-NeRF does not use the input xyz"
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
                    coords_normalization = None if store_planes else coords_normalization,
                    dec_density_layers=getattr(cfg.models.fine,'dec_density_layers',4),
                    dec_rgb_layers=getattr(cfg.models.fine,'dec_rgb_layers',4),
                    dec_channels=getattr(cfg.models.fine,'dec_channels',128),
                    skip_connect_every=getattr(cfg.models.fine,'skip_connect_every',None),
                    num_plane_channels=getattr(cfg.models.fine,'num_plane_channels',48),
                    num_viewdir_plane_channels=getattr(cfg.models.fine,'num_viewdir_plane_channels',None),
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
                    ensemble_size=getattr(cfg.models.fine,'ensemble_size',1),
                    dec_dss_layers=getattr(cfg.models.fine,'dec_dss_layers',None),
                    plane_stats=PLANE_STATS,
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

    rendering_loss_w = 1
    SR_model,SR_optimizer = None,None
    if SR_experiment=="model":
        if 'SR' not in what2train:
        # if train_planes_only:
            assert not (hasattr(cfg.super_resolution,'model') and hasattr(cfg.super_resolution.model,'path'))
            if not hasattr(cfg.super_resolution,'model'):   setattr(cfg.super_resolution,'model',CfgNode())
            rsetattr(cfg.super_resolution,'model.path',cfg.models.path)
            SR_model_config = get_config(os.path.join(cfg.super_resolution.model.path,"config.yml"))
            set_config_defaults(source=SR_model_config.super_resolution,target=cfg.super_resolution)
        rendering_loss_w = getattr(cfg.super_resolution,'rendering_loss',1)
        plane_channels = getattr(cfg.models.coarse,'num_plane_channels',48)
        if not eval_mode:   saved_rgb_fine = dict(zip(evaluation_sequences,[{} for i in evaluation_sequences]))
        sf_config = getattr(cfg.super_resolution.model,'scale_factor','linear')
        assert sf_config in ['linear','sqrt','one'] or isinstance(sf_config,int)
        if sf_config=='one':
            SR_factor = 1
        elif sf_config=='linear':
            SR_factor = int(scene_coupler.ds_factor)
        elif sf_config=='sqrt':
            SR_factor = int(np.sqrt(scene_coupler.ds_factor))
        else:
            SR_factor = sf_config
        SR_model = models.PlanesSR(
            model_arch=getattr(models, cfg.super_resolution.model.type),scale_factor=SR_factor,
            in_channels=plane_channels*(1 if getattr(cfg.super_resolution.model,'single_plane',True) else model_fine.num_density_planes),
            out_channels=plane_channels*(1 if getattr(cfg.super_resolution.model,'single_plane',True) else model_fine.num_density_planes),
            sr_config=cfg.super_resolution,
            plane_interp=getattr(cfg.super_resolution,'plane_resize_mode',model_fine.plane_interp),
            detach_LR_planes=getattr(cfg.nerf.train,'detach_LR_planes',True)
        )
        print("SR model: %d parameters"%(num_parameters(SR_model)))
        SR_model.to(device)
        if 'SR' in what2train:
        # if not (eval_mode or train_planes_only):   
            SR_optimizer = getattr(torch.optim, cfg.optimizer.type)(
                [p for k,p in SR_model.named_parameters() if 'NON_LEARNED' not in k],
                lr=getattr(cfg.super_resolution,'lr',cfg.optimizer.lr)
            )
        else:
            assert all([sc not in scene_coupler.downsample_couples.keys() for sc in training_scenes]),"Why train on SR scenes when training only the planes? Currently not assigning the SR model's LR planes during training."
    if rep_model_training or not planes_model:
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
    # if eval_mode or train_planes_only:
    # if eval_mode or not rep_model_training:
    if not eval_mode and (rep_model_training or not planes_model):
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            trainable_parameters_, lr=cfg.optimizer.lr
        )
    else:
        optimizer = None
    # Load an existing checkpoint, if a path is specified.
    # models2save = (['representation'] if (rep_model_training or train_planes_only) else [])+(['SR'] if SR_experiment=='model' else [])
    models2save = (['decoder'] if 'decoder' in what2train else [])+(['SR'] if SR_experiment=='model' and 'SR' in what2train else [])
    # best_saved = (0,np.finfo(np.float32).max)
    experiment_info.update({'start_i':0,'eval_counter':0,'best_loss':(0,np.finfo(np.float32).max),'last_saved':dict(zip(models2save,[[] for i in models2save]))})
    experiment_info_file = os.path.join(logdir, "exp_info.pkl")
    # experiment_info['start_i'],experiment_info['eval_counter'] = 0,0
    # experiment_info['best_loss'] = (0,np.finfo(np.float32).max)
    # experiment_info['last_saved'] = dict(zip(models2save,[[] for i in models2save]))
    params_init_path = None
    if load_saved_models:
        if resume_experiment and not eval_mode:
            legacy_info_tracking = not os.path.exists(experiment_info_file)
            if not legacy_info_tracking:
                experiment_info = safe_loading(experiment_info_file,suffix='pkl')

        # load_best = eval_mode or train_planes_only
        load_best = eval_mode or init_new_scenes
        # initialize_from_trained = configargs.load_checkpoint=='' and end2end_training
        if SR_experiment:
            if SR_experiment=="model" and ('SR' not in what2train or resume_experiment):
            # if SR_experiment=="model" and (resume_experiment or train_planes_only or initialize_from_trained):
                if resume_experiment:
                    SR_checkpoint_path = configargs.load_checkpoint
                elif getattr(cfg.super_resolution.model,'path',None) is not None:
                    SR_checkpoint_path = cfg.super_resolution.model.path
                else:
                    SR_checkpoint_path = cfg.models.path
                SR_checkpoint_path = find_latest_checkpoint(SR_checkpoint_path,sr=True,find_best=load_best or ('SR' not in what2train))
                SR_model_checkpoint = safe_loading(SR_checkpoint_path,suffix='ckpt_best' if load_best else 'ckpt')
                if legacy_info_tracking and not eval_mode and 'decoder' not in what2train:
                    experiment_info['start_i'] = int(SR_checkpoint_path.split('/')[-1][len('SR_checkpoint'):-len('.ckpt')])+1
                    experiment_info['eval_counter'] = SR_model_checkpoint['eval_counter']
                    experiment_info['best_loss'] = SR_model_checkpoint['best_loss'] if 'best_loss' in SR_model_checkpoint else SR_model_checkpoint['best_saved']
                print(("Using" if load_best else "Resuming training of")+" SR model %s"%(SR_checkpoint_path))
                saved_config_dict = get_config(os.path.join('/'.join(SR_checkpoint_path.split('/')[:-1]),"config.yml"))
                config_diffs = DeepDiff(saved_config_dict.super_resolution,cfg.super_resolution)
                for diff in [config_diffs[ch_type] for ch_type in ['dictionary_item_removed','dictionary_item_added','values_changed'] if ch_type in config_diffs]:
                    print(diff)
                if not all([any([t in k for t in ['inner_model','NON_LEARNED']]) for k in SR_model_checkpoint["SR_model"].keys()]):
                    assert not any(['inner_model' in k for k in SR_model_checkpoint["SR_model"].keys()])
                    SR_model_checkpoint["SR_model"] = OrderedDict([((k,v) if 'NON_LEARNED' in k else ('inner_model.'+k,v)) for k,v in SR_model_checkpoint["SR_model"].items()])
                SR_model.load_state_dict(SR_model_checkpoint["SR_model"])
                if SR_optimizer is not None and "SR_optimizer" in SR_model_checkpoint:
                    SR_optimizer.load_state_dict(SR_model_checkpoint['SR_optimizer'])
                del SR_model_checkpoint

        # # if rep_model_training or (train_planes_only and not init_new_scenes):
        # if rep_model_training or not planes_model:
        # # if rep_model_training or train_planes_only:
        if configargs.load_checkpoint=='' or 'decoder' not in what2train: # Initializing a representation model with a pre-trained model
            checkpoint_filename = find_latest_checkpoint(cfg.models.path,sr=False,find_best=load_best or 'decoder' not in what2train)
            print("Initializing model training from model %s"%(checkpoint_filename))
            checkpoint = safe_loading(checkpoint_filename,suffix='ckpt_best' if load_best else 'ckpt')
        else:
            checkpoint_filename = find_latest_checkpoint(configargs.load_checkpoint,sr=False,find_best=load_best or 'decoder' not in what2train)
            checkpoint = safe_loading(checkpoint_filename,suffix='ckpt_best' if load_best else 'ckpt')
            if legacy_info_tracking and not eval_mode:
                experiment_info['start_i'] = int(checkpoint_filename.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
                experiment_info['eval_counter'] = checkpoint['eval_counter']
                # best_saved = checkpoint['best_saved']
                experiment_info['best_loss'] = checkpoint['best_loss'] if 'best_loss' in checkpoint else checkpoint['best_saved']
            print("Resuming training on model %s"%(checkpoint_filename))
        checkpoint_config = get_config(os.path.join('/'.join(checkpoint_filename.split('/')[:-1]),'config.yml'))
        config_diffs = DeepDiff(checkpoint_config.models,cfg.models)
        ok = True
        for ch_type in ['dictionary_item_removed','dictionary_item_added','values_changed','type_changes']:
            if ch_type not in config_diffs: continue
            for diff in config_diffs[ch_type]:
                if ch_type in ['dictionary_item_added','values_changed'] and diff=="root['path']":  continue
                if ch_type=='dictionary_item_removed' and "['use_viewdirs']" in diff:  continue
                elif ch_type=='dictionary_item_added':
                    if diff[:len("root['fine']")]=="root['fine']":  continue
                    if diff=="root['use_existing_planes']": continue
                elif ch_type=='dictionary_item_removed' and "root['fine']" in str(config_diffs[ch_type]):   continue
                print(ch_type,diff)
                ok = False
        if not (ok or eval_mode): # Not abort the run in eval_mode because then the configuration is copied from the existing one anyway. They can differ if the model configuration is changed by the code at some stage.
            raise Exception('Inconsistent model config')
        # if not (ok or train_planes_only):  raise Exception('Inconsistent model config')

        def load_saved_parameters(model,saved_params,reduced_set=False):
            if not all([search('density_dec\.(\d)+\.(\d)+\.',p) is not None for p in saved_params if 'density_dec' in p]):
                saved_params = OrderedDict([(k if 'NON_LEARNED' in k else k.replace('.','.0.',1),v) for k,v in saved_params.items()])
            mismatch = model.load_state_dict(saved_params,strict=False)
            allowed_missing = []
            if store_planes:    allowed_missing.append('planes_.sc')
            if reduced_set: allowed_missing.append('rot_mats')
            assert (len(mismatch.missing_keys)==0 or all([any([tok in k for tok in allowed_missing]) for k in mismatch.missing_keys]))\
                and all(['planes_.sc' in k for k in mismatch.unexpected_keys])
            if planes_model and not store_planes:
                model.box_coords = checkpoint["coords_normalization"]

        if planes_model:
            checkpoint["model_coarse_state_dict"] = model_coarse.rot_mat_backward_support(checkpoint["model_coarse_state_dict"])
        load_saved_parameters(model_coarse,checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            load_saved_parameters(model_fine,checkpoint["model_fine_state_dict"],reduced_set=True)
        if "optimizer" in checkpoint and optimizer is not None:
            print("Loading optimizer's checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer"])
        del checkpoint # Importent: Releases GPU memory occupied by loaded data.

    spatial_padding_size = SR_model.receptive_field//2 if isinstance(SR_model,models.Conv3D) else 0
    spatial_sampling = spatial_padding_size>0 or cfg.nerf.train.get("spatial_sampling",False)
    if spatial_padding_size>0 or spatial_sampling or sr_val_scenes_with_LR:
        SAMPLE_PATCH_BY_CONTENT = True
        patch_size = chunksize_to_2D(cfg.nerf.train.num_random_rays)
        optional_size_divider = scene_coupler.ds_factor if sr_val_scenes_with_LR else 1
        if SAMPLE_PATCH_BY_CONTENT:
            assert getattr(cfg.nerf.train,'spatial_patch_sampling','background_est') in ['background_est','STD']
            if getattr(cfg.nerf.train,'spatial_patch_sampling','background_est')=='STD':
                im_2_sampling_dist = image_STD_2_distribution(patch_size=patch_size//optional_size_divider)
            else:
                im_2_sampling_dist = estimated_background_2_distribution(patch_size=patch_size//optional_size_divider)
                
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
    run_time_signature = time.time()
    if store_planes:
        planes_folder = os.path.join(LR_model_folder if not rep_model_training else logdir,'planes')
        if eval_mode: assert os.path.isdir(planes_folder)
        if not os.path.isdir(planes_folder):
            os.mkdir(planes_folder)
        scenes_cycle_counter = Counter()
        optimize_planes = any(['planes' in m for m in what2train]) and not eval_mode
        lr_scheduler = getattr(cfg.optimizer,'lr_scheduler',None)
        if getattr(cfg.models,'use_existing_planes',False):
            use_frozen_planes = os.path.join(cfg.models.path,'planes')
        else:
            use_frozen_planes = ''
        if lr_scheduler is not None:
            lr_scheduler['patience'] = int(np.ceil(lr_scheduler['patience']/cfg.experiment.print_every))
        planes_opt = models.PlanesOptimizer(optimizer_type=cfg.optimizer.type,
            scene_id_plane_resolution=scene_id_plane_resolution,options=cfg.nerf.train.store_planes,save_location=planes_folder,
            lr=getattr(cfg.optimizer,'planes_lr',cfg.optimizer.lr),model_coarse=model_coarse,model_fine=model_fine,
            use_coarse_planes=getattr(cfg.models.fine,'use_coarse_planes',False),
            init_params=init_new_scenes,optimize=optimize_planes,training_scenes=training_scenes,
            coords_normalization=None if not init_new_scenes else coords_normalization,
            do_when_reshuffling=lambda:scenes_cycle_counter.step(print_str='Number of scene cycles performed: '),
            STD_factor=getattr(cfg.nerf.train,'STD_factor',0.1),
            available_scenes=available_scenes,planes_rank_ratio=getattr(cfg.models.coarse,'planes_rank_ratio',None),
            copy_params_path=params_init_path,run_time_signature=run_time_signature,
            lr_scheduler=lr_scheduler,use_frozen_planes=use_frozen_planes,
        )

    if planes_model:    assert not (hasattr(cfg.models.coarse,'plane_resolutions') or hasattr(cfg.models.coarse,'viewdir_plane_resolution')),'Depricated.'
    if SR_experiment=="model" and getattr(cfg.super_resolution,'input_normalization',False) and not resume_experiment:
        #Initializing a new SR model that uses input normalization
        SR_model.normalization_params(planes_opt.get_plane_stats(viewdir=getattr(cfg.super_resolution,'SR_viewdir',False),single_plane=SR_model.single_plane))

    downsampling_offset = lambda ds_factor: (ds_factor-1)/(2*ds_factor)
    saved_target_ims = dict(zip(set(val_strings),[set() for i in set(val_strings)]))#set()
    if isinstance(spatial_sampling,bool):
        spatial_margin = SR_model.receptive_field//2 if spatial_sampling else None
    else:
        spatial_margin = 0
    virtual_batch_size = getattr(cfg.nerf.train,'virtual_batch_size',1)
    if sr_val_scenes_with_LR:
        def downsample_rendered_pixels(pixels):
            return model_fine.downsample_plane(pixels.permute(1,0).reshape([1,3,patch_size,patch_size])).reshape([3,-1]).permute(1,0)

    def mse_loss(x,y,weights=None):
        if weights is None:
            return torch.nn.functional.mse_loss(x,y)
        else:
            loss = torch.nn.functional.mse_loss(x,y,reduction='none').mean(1)
            return (loss*weights).mean()

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
                val_ind = lambda dummy: experiment_info['eval_counter']%val_ims_per_scene
                img_indecis = [[v[val_ind(val_strings[i])] for i,v in enumerate(i_val.values())]]
                eval_cycles = 1

            record_fine = True
            for eval_cycle in range(eval_cycles):
                coarse_loss,fine_loss,loss,psnr,rgb_coarse,rgb_fine,rgb_SR,target_ray_values,consistency_loss,planes_SR_loss,zero_mean_planes_loss =\
                    defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
                for eval_num,img_idx in enumerate(tqdm(img_indecis[eval_cycle],
                    desc='Evaluating '+('scene (%d/%d): %s'%(eval_cycle+1,eval_cycles,evaluation_sequences[eval_cycle]) if eval_mode else 'scenes'))):
                    if eval_mode:
                        val_ind = lambda dummy: eval_cycle
                        scene_num = eval_cycle
                    else:
                        scene_num = eval_num
                    sr_scene = SR_experiment and scene_ids[img_idx] in scene_coupler.downsample_couples and '_HRplane' not in evaluation_sequences[scene_num]
                    HR_plane_LR_im = scene_ids[img_idx] in scene_coupler.HR_planes_LR_ims_scenes
                    scene_coupler.toggle_used_planes_res(HR='_HRplane' in evaluation_sequences[scene_num])
                    if True: #dataset_type=='DTU' or CONSTRUCT_DATASET:
                        img_target,pose_target,cur_H,cur_W,cur_focal,cur_ds_factor = dataset.item(img_idx,device)
                    else:
                        img_target = images[img_idx].to(device)
                        pose_target = poses[img_idx, :3, :4].to(device)
                        cur_H,cur_W,cur_focal,cur_ds_factor = H[img_idx], W[img_idx], focal[img_idx],ds_factor[img_idx]
                    if HR_plane_LR_im:
                        cur_H *= scene_coupler.ds_factor
                        cur_W *= scene_coupler.ds_factor
                        cur_focal *= scene_coupler.ds_factor
                    ray_origins, ray_directions = get_ray_bundle(
                        cur_H, cur_W, cur_focal, pose_target,padding_size=spatial_padding_size,downsampling_offset=downsampling_offset(cur_ds_factor),
                    )
                    if store_planes and (not eval_mode or eval_num==0):
                        planes_opt.load_scene(scene_ids[img_idx],load_best=eval_mode)
                        planes_opt.cur_id = scene_ids[img_idx]
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
                        ds_factor_or_id=scene_ids[img_idx],
                        # ds_factor_or_id=scene_ids[img_idx] if planes_model else ds_factor[img_idx],
                        spatial_margin=spatial_margin,
                        scene_config=cfg.dataset[dataset.scene_types[scene_ids[img_idx]]],
                    )
                    target_ray_values[val_strings[scene_num]].append(img_target[...,:3])
                    if planes_model:
                        zero_mean_planes_loss_ = model_coarse.return_zero_mean_planes_loss()
                        if zero_mean_planes_loss_ is not None:
                            zero_mean_planes_loss[val_strings[scene_num]].append(zero_mean_planes_loss_.item())
                    if sr_scene:
                        if SR_experiment=="refine" or planes_model:
                            rgb_SR_ = 1*rgb_fine_
                            if HR_plane_LR_im:
                                rgb_SR_ = model_fine.downsample_plane(rgb_SR_.permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0)
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
                                    ds_factor_or_id=scene_ids[img_idx],
                                    # ds_factor_or_id=scene_ids[img_idx] if planes_model else ds_factor[img_idx],
                                    scene_config=cfg.dataset[dataset.scene_types[scene_ids[img_idx]]],
                                )
                                model_coarse.skip_SR(False)
                                model_fine.skip_SR(False)
                                if not eval_mode and not planes_updating:
                                    saved_rgb_fine[evaluation_sequences[scene_num]][val_ind(val_strings[scene_num])] = 1*rgb_fine_.detach()
                            else:
                                record_fine = False
                                rgb_fine_ = 1*saved_rgb_fine[evaluation_sequences[scene_num]][val_ind(val_strings[scene_num])]
                            if HR_plane_LR_im:
                                rgb_fine_ = model_fine.downsample_plane(rgb_fine_.permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0)

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
                            planes_opt.generated_planes.clear()
                            planes_opt.downsampled_planes.clear()
                SAVE_COARSE_IMAGES = False
                cur_val_sets = [val_strings[eval_cycle]] if eval_mode else set(val_strings)
                for val_set in cur_val_sets:
                    sr_scene_inds = [i for i,im in enumerate(rgb_SR[val_set]) if im is not None]
                    if len(sr_scene_inds)>0: #SR_experiment and '_HRplanes' not in val_set:
                        # SR_psnr_gain = np.mean([psnr[val_set][i]-mse2psnr(fine_loss[val_set][i]) for i in sr_scene_inds])
                        SR_psnr_gain = [psnr[val_set][i]-mse2psnr(fine_loss[val_set][i]) for i in sr_scene_inds]
                        if not eval_mode:
                            SR_psnr_gain = np.mean(SR_psnr_gain)
                        write_scalar("%s/SR_psnr_gain"%(val_set),SR_psnr_gain,iter)
                        write_image(name="%s/rgb_SR"%(val_set),
                            images=[torch.zeros_like(rgb_fine[val_set][i]) if im is None else im for i,im in enumerate(rgb_SR[val_set])],
                            text=str(val_ind(val_set)),psnrs=[None if im is None else SR_psnr_gain[i] if eval_mode else psnr[val_set][i] for i,im in enumerate(rgb_SR[val_set])],
                            iter=iter,fontscale=font_scale/scene_coupler.ds_factor if ('_LR' in val_set or '_downscaled' in val_set) else font_scale)
                        if val_set in consistency_loss:
                            write_scalar("%s/inconsistency"%(val_set), np.mean(consistency_loss[val_set]), iter)
                        if val_set in planes_SR_loss:
                            write_scalar("%s/planes_SR"%(val_set), np.mean(planes_SR_loss[val_set]), iter)
                    if val_set in zero_mean_planes_loss:
                        write_scalar("%s/zero_mean_planes_loss"%(val_set), np.mean(zero_mean_planes_loss[val_set]), iter)
                    # write_scalar("%s/fine_loss"%(val_set), np.mean(fine_loss[val_set]), iter)
                    write_scalar("%s/fine_psnr"%(val_set), np.mean([mse2psnr(l) for l in fine_loss[val_set]]), iter)
                    write_scalar("%s/loss"%(val_set), np.mean(loss[val_set]), iter)
                    write_scalar("%s/psnr"%(val_set), np.mean(psnr[val_set]), iter)
                    if len(coarse_loss[val_set])>0:
                        write_scalar("%s/coarse_loss"%(val_set), np.mean(coarse_loss[val_set]), iter)
                    if len(rgb_fine[val_set])>0:
                        if record_fine:
                            # recorded_iter = val_ind(val_set) if not planes_updating else iter
                            recorded_iter = iter if (planes_updating or not planes_model) else val_ind(val_set) 
                            write_scalar("%s/fine_loss"%(val_set), np.mean(fine_loss[val_set]), recorded_iter)
                            if eval_mode and evaluation_sequences[val_ind(val_set)] in scene_coupler.downsample_couples.values():
                                write_image(name="%s/rgb_bicubic"%(val_set),
                                    images=[torch.from_numpy(bicubic_interp(im.cpu().numpy(),sf=scene_coupler.ds_factor)) for im in rgb_fine[val_set]],
                                    text=str(val_ind(val_set)),
                                    psnrs=[],iter=recorded_iter,
                                    fontscale=font_scale/scene_coupler.ds_factor if ('_LR' in val_set or '_downscaled' in val_set) else font_scale)    
                                write_image(name="%s/rgb_LR"%(val_set),
                                    images=[im.unsqueeze(2).unsqueeze(1).repeat([1,scene_coupler.ds_factor,1,scene_coupler.ds_factor,1]).reshape([im.shape[0]*scene_coupler.ds_factor,im.shape[1]*scene_coupler.ds_factor,-1]) for im in rgb_fine[val_set]],
                                    text=str(val_ind(val_set)),
                                    psnrs=[],iter=recorded_iter,
                                    fontscale=font_scale/scene_coupler.ds_factor if ('_LR' in val_set or '_downscaled' in val_set) else font_scale)    
                            write_image(name="%s/rgb_fine"%(val_set),images=rgb_fine[val_set],text=str(val_ind(val_set)),
                                psnrs=[mse2psnr(l) for l in fine_loss[val_set]],iter=recorded_iter,
                                fontscale=font_scale/scene_coupler.ds_factor if ('_LR' in val_set or '_downscaled' in val_set) else font_scale)
                    if SAVE_COARSE_IMAGES:
                        write_image(name="%s/rgb_coarse"%(val_set),images=rgb_coarse[val_set],text=str(val_ind(val_set)),
                            psnrs=[mse2psnr(l) for l in coarse_loss[val_set]],iter=iter,
                            fontscale=font_scale/scene_coupler.ds_factor if ('_LR' in val_set or '_downscaled' in val_set) else font_scale)
                    if not eval_mode and val_ind(val_set) not in saved_target_ims[val_set]:
                        write_image(name="%s/img_target"%(val_set),images=target_ray_values[val_set],
                            text=str(val_ind(val_set)),iter=val_ind(val_set),fontscale=font_scale/scene_coupler.ds_factor if ('_LR' in val_set or '_downscaled' in val_set) else font_scale)
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
        # if SR_experiment=="model" and not train_planes_only:
        if SR_experiment=="model" and 'SR' in what2train:
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
        # if store_planes:
        # img_idx = np.random.choice(available_train_inds)
        img_idx = image_sampler.sample()
        # else:
        #     img_idx = np.random.choice(i_train)
        cur_scene_id = scene_ids[img_idx]
        hr_planes_iter = len(scene_coupler.HR_planes)>0 and cur_scene_id in scene_coupler.downsample_couples and np.random.uniform()>=0.5
        # if len(scene_coupler.HR_planes)>0 and cur_scene_id in scene_coupler.downsample_couples:
        scene_coupler.toggle_used_planes_res(hr_planes_iter)
        sr_iter = cur_scene_id in scene_coupler.downsample_couples and not hr_planes_iter
        HR_plane_LR_im = cur_scene_id in scene_coupler.HR_planes_LR_ims_scenes
        # if dataset_type=='synt':
        if True: #dataset_type=='DTU' or CONSTRUCT_DATASET:
            img_target,pose_target,cur_H,cur_W,cur_focal,cur_ds_factor = dataset.item(img_idx,device)
        else:
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            cur_H,cur_W,cur_focal,cur_ds_factor = H[img_idx], W[img_idx], focal[img_idx],ds_factor[img_idx]
        if HR_plane_LR_im:
            cur_H *= scene_coupler.ds_factor
            cur_W *= scene_coupler.ds_factor
            cur_focal *= scene_coupler.ds_factor
        ray_origins, ray_directions = get_ray_bundle(cur_H, cur_W, cur_focal, pose_target,padding_size=spatial_padding_size,downsampling_offset=downsampling_offset(cur_ds_factor),)
        coords = torch.stack(
            meshgrid_xy(torch.arange(cur_H+2*spatial_padding_size).to(device), torch.arange(cur_W+2*spatial_padding_size).to(device)),
            dim=-1,
        )
        if spatial_padding_size>0 or spatial_sampling or HR_plane_LR_im:
            if SAMPLE_PATCH_BY_CONTENT:
                patches_vacancy_dist = im_2_sampling_dist(img_target[...,:3])
                upper_left_corner = torch.argwhere(torch.rand([])<torch.cumsum(patches_vacancy_dist.reshape([-1]),0))[0].item()
                upper_left_corner = np.array([upper_left_corner//patches_vacancy_dist.shape[1],upper_left_corner%patches_vacancy_dist.shape[1]])
            else:
                upper_left_corner = np.random.uniform(size=[2])*(np.array([cur_H,cur_W])-patch_size)
                upper_left_corner = np.floor(upper_left_corner).astype(np.int32)
            cropped_inds =\
                coords[upper_left_corner[1]:upper_left_corner[1]+patch_size//optional_size_divider,\
                upper_left_corner[0]:upper_left_corner[0]+patch_size//optional_size_divider]
            cropped_inds = cropped_inds.reshape([-1,2])
            if HR_plane_LR_im:
                upper_left_corner *= scene_coupler.ds_factor
            select_inds = \
                coords[upper_left_corner[1]:upper_left_corner[1]+patch_size+2*spatial_padding_size,\
                upper_left_corner[0]:upper_left_corner[0]+patch_size+2*spatial_padding_size]
            select_inds = select_inds.reshape([-1,2])
            if spatial_sampling>1:
                raise Exception('Unsupported')
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
            if optimizer is not None:
                optimizer.zero_grad()
            if SR_optimizer is not None:
                SR_optimizer.zero_grad()
        if store_planes:
            planes_opt.cur_id = scene_ids[img_idx]
            planes_opt.zero_grad()
        if hasattr(model_fine,'SR_model') and sr_iter:
            # if 'SR' in what2train:
            # if end2end_training:
            if optimize_planes:
                model_fine.assign_LR_planes(scene=cur_scene_id)
            # Handling SR planes dropout:
            model_fine.SR_planes2drop = [] if np.random.uniform()>=model_fine.SR_plane_dropout\
                 else sorted(list(np.random.choice(range(model_fine.num_density_planes),size=np.random.randint(1,model_fine.num_density_planes),replace=False)))
        else: # Handling representation model planes dropout:
            planes2drop = [] if np.random.uniform()>=getattr(cfg.nerf.train,'plane_dropout',0)\
                 else sorted(list(np.random.choice(range(model_fine.num_density_planes),size=np.random.randint(1,model_fine.num_density_planes-3),replace=False)))
            model_fine.planes2drop = planes2drop
            model_coarse.planes2drop = planes2drop
        rgb_coarse, _, acc_coarse, rgb_fine, _, acc_fine,rgb_SR,_,_ = run_one_iter_of_nerf(
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
            ds_factor_or_id=scene_ids[img_idx],
            # ds_factor_or_id=cur_scene_id if planes_model else ds_factor[img_idx],
            spatial_margin=spatial_margin,
            scene_config=cfg.dataset[dataset.scene_types[cur_scene_id]],
        )
        coarse_loss_weights,fine_loss_weights = None,None
        gt_background_pixels = target_s[...,3]==0 if target_s.shape[1]==4 else None
        if getattr(cfg.nerf.train,'random_target_bg_color',False) and gt_background_pixels is not None:
            target_s[gt_background_pixels] = torch.rand_like(target_s[gt_background_pixels])
        if getattr(cfg.nerf.train,'mask_background',False) and gt_background_pixels is not None:
            coarse_loss_weights,fine_loss_weights = torch.ones_like(acc_coarse),torch.ones_like(acc_coarse)
            coarse_loss_weights[gt_background_pixels] = acc_coarse[gt_background_pixels]
            coarse_loss_weights[torch.logical_not(gt_background_pixels)] = coarse_loss_weights[torch.logical_not(gt_background_pixels)]*\
                (len(gt_background_pixels)-coarse_loss_weights[gt_background_pixels].sum())/(len(gt_background_pixels)-gt_background_pixels.sum())
            fine_loss_weights[gt_background_pixels] = acc_fine[gt_background_pixels]
            fine_loss_weights[torch.logical_not(gt_background_pixels)] = fine_loss_weights[torch.logical_not(gt_background_pixels)]*\
                (len(gt_background_pixels)-fine_loss_weights[gt_background_pixels].sum())/(len(gt_background_pixels)-gt_background_pixels.sum())
        target_ray_values = target_s
        if HR_plane_LR_im:
            rgb_coarse,rgb_fine = downsample_rendered_pixels(rgb_coarse),downsample_rendered_pixels(rgb_fine)
        coarse_loss,fine_loss = None,None
        if rendering_loss_w is not None:
            # if rep_model_training or train_planes_only or getattr(cfg.super_resolution.training,'loss','both')!='fine':
            if not planes_model or any([m in what2train for m in ['decoder','LR_planes','HR_planes']]) or getattr(cfg.super_resolution.training,'loss','both')!='fine':
                coarse_loss = mse_loss(
                    rgb_coarse, target_ray_values[..., :3],weights=coarse_loss_weights,
                )
            # if rgb_fine is not None and (rep_model_training or train_planes_only or getattr(cfg.super_resolution.training,'loss','both')!='coarse'):
            if rgb_fine is not None and (not planes_model or any([m in what2train for m in ['decoder','LR_planes','HR_planes']]) or getattr(cfg.super_resolution.training,'loss','both')!='coarse'):
                fine_loss = mse_loss(
                    rgb_fine, target_ray_values[..., :3],weights=fine_loss_weights,
                )
        rendering_loss = ((coarse_loss if coarse_loss is not None else 0.0) + (fine_loss if fine_loss is not None else 0.0))
        loss = rendering_loss_w*rendering_loss
        psnr = None
        if isinstance(loss,torch.Tensor):
            write_scalar("train/loss", loss.item(), iter)
            psnr = mse2psnr(loss.item())
            write_scalar("train/psnr", psnr, iter)

        if planes_model:
            zero_mean_planes_loss = model_coarse.return_zero_mean_planes_loss()
            if zero_mean_planes_loss is not None:
                write_scalar("train/zero_mean_planes_loss", zero_mean_planes_loss.item(), iter)
                loss += cfg.nerf.train.zero_mean_planes_w*zero_mean_planes_loss
            if SR_experiment=="model":
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
            new_drawn_scenes = planes_opt.step()
        if last_v_batch_iter:
            if optimizer is not None:
                decoder_step = True
                if 'SR' in what2train:
                # if end2end_training:
                    if getattr(cfg.nerf.train,'separate_decoder_sr',False):
                        decoder_step &= not sr_iter
                    if getattr(cfg.nerf.train,'separate_decoder_val_scenes',False):
                        decoder_step &= cur_scene_id in [s for k in scene_coupler.upsample_couples.items() for s in k]

                if decoder_step:
                    optimizer.step()
            if SR_optimizer is not None and sr_iter:
                SR_optimizer.step()
        if coarse_loss is not None:
            write_scalar("train/coarse_loss", coarse_loss.item(), iter)
        if fine_loss is not None:
            write_scalar("train/fine_loss", fine_loss.item(), iter)
            write_scalar("train/fine_psnr", mse2psnr(fine_loss.item()), iter)
        return loss.item(),psnr,new_drawn_scenes

    training_time,last_evaluated = 0,1*experiment_info['start_i']
    recently_saved, = time.time(),
    eval_loss_since_save,print_cycle_loss,print_cycle_psnr = [],[],[]
    evaluation_time = 0
    recent_loss_avg = np.nan
    jump_start_phase = isinstance(getattr(cfg.nerf.train,'jump_start',False),list) and experiment_info['start_i']==0
    if jump_start_phase:
        n_jump_start_scenes = planes_opt.jump_start(config=cfg.nerf.train.jump_start)
    for iter in trange(experiment_info['start_i'],cfg.experiment.train_iters):
        # Validation
        if isinstance(cfg.experiment.validate_every,list):
            evaluate_now = evaluation_time<=training_time*cfg.experiment.validate_every[0] or iter-last_evaluated>=cfg.experiment.validate_every[1]
        else:
            evaluate_now = iter % cfg.experiment.validate_every == 0
        evaluate_now |= iter == cfg.experiment.train_iters - 1
        if iter>0:  evaluate_now &= not jump_start_phase
        if True and DEBUG_MODE and evaluate_now:
            print('!!!!!!!!!!WARNING!!!!!!!!!!!')
            evaluate_now = False
            # planes_opt.draw_scenes(assign_LR_planes=not end2end_training)
            planes_opt.draw_scenes(assign_LR_planes=SR_experiment!="model" or not optimize_planes)
            # planes_opt.draw_scenes(assign_LR_planes='SR' not in what2train)
            image_sampler.update_active(planes_opt.cur_scenes)
        if evaluate_now:
            last_evaluated = 1*iter
            start_time = time.time()
            loss,psnr = evaluate()
            eval_loss_since_save.extend([v for term in important_loss_terms for v in loss[term]])
            evaluation_time = time.time()-start_time
            if store_planes and not eval_mode:
                if not jump_start_phase or iter==0:
                    # planes_opt.draw_scenes(assign_LR_planes=not end2end_training)
                    planes_opt.draw_scenes(assign_LR_planes=SR_experiment!="model" or not optimize_planes)
                    # planes_opt.draw_scenes(assign_LR_planes='SR' not in what2train)
                    new_drawn_scenes = planes_opt.cur_scenes
                    if jump_start_phase:    new_drawn_scenes = new_drawn_scenes[:n_jump_start_scenes]
                    image_sampler.update_active(new_drawn_scenes)
                    # available_train_inds = [i for i in i_train if scene_ids[i] in new_drawn_scenes]
            elif not store_planes:
                image_sampler.update_active(training_scenes)
            training_time = 0
            experiment_info['eval_counter'] += 1

        if eval_mode:   break
        # Training:
        start_time = time.time()
        loss,psnr,new_drawn_scenes = train()
        if new_drawn_scenes is not None:
            image_sampler.update_active(new_drawn_scenes)
            # available_train_inds = [i for i in i_train if scene_ids[i] in new_drawn_scenes]

        print_cycle_loss.append(loss)
        print_cycle_psnr.append(psnr)
        training_time += time.time()-start_time

        if iter % cfg.experiment.print_every == 0 or iter == cfg.experiment.train_iters - 1:
            if iter>0 and jump_start_phase:
                if np.mean(print_cycle_loss)<=cfg.nerf.train.jump_start[1]:
                    jump_start_phase = False
                    new_drawn_scenes = planes_opt.jump_start(on=False)
                    # available_train_inds = [i for i in i_train if scene_ids[i] in new_drawn_scenes]
                    image_sampler.update_active(new_drawn_scenes)

            tqdm.write(
                "[TRAIN] Iter: "
                + str(iter)
                + " Loss: "
                # + str(loss.item())
                + str(np.mean(print_cycle_loss))
                + " PSNR: "
                + str(np.mean(print_cycle_psnr))
            )
            if planes_model:
                planes_opt.lr_scheduler_step(np.mean(print_cycle_loss))
            print_cycle_loss,print_cycle_psnr = [],[]
        save_now = scenes_cycle_counter.check_and_reset() if (store_planes and rep_model_training) else False
        save_now |= iter % cfg.experiment.save_every == 0 if isinstance(cfg.experiment.save_every,int) else (time.time()-recently_saved)/60>cfg.experiment.save_every
        save_now |= iter == cfg.experiment.train_iters - 1
        save_now &= iter>0
        if save_now:
            save_as_best = False
            if len(experiment_info['running_scores']['loss'][list(important_loss_terms)[0]])==val_ims_per_scene:
                recent_loss_avg = np.mean([l for term in important_loss_terms for l in experiment_info['running_scores']['loss'][term]])
                if recent_loss_avg<experiment_info['best_loss'][1]:
                    experiment_info['best_loss'] = (iter,recent_loss_avg)
                    save_as_best = True
            if save_as_best:
                print("================Saving new best checkpoint at iteration %d, with average evaluation loss %.3e====================="%(experiment_info['best_loss'][0],experiment_info['best_loss'][1]))
            else:
                print("================Best checkpoint is still %d, with average evaluation loss %.3e (recent average is %.3e)====================="%(experiment_info['best_loss'][0],experiment_info['best_loss'][1],recent_loss_avg))
            recently_saved = time.time()
            eval_loss_since_save = []
            for model2save in models2save:
                model_filename = '%scheckpoint'%('SR_' if model2save=='SR' else '')
                checkpoint_dict = {}
                # checkpoint_dict = {
                #     "iter": iter,
                #     "eval_counter": eval_counter,
                #     "best_loss":best_loss,
                # }
                if model2save=="SR":
                    checkpoint_dict.update({"SR_model":SR_model.state_dict()})
                    if SR_optimizer is not None:
                        checkpoint_dict.update({"SR_optimizer": SR_optimizer.state_dict()})
                else:
                    tokens_2_exclude = ['planes_.','SR_model']
                    checkpoint_dict.update({"model_coarse_state_dict": model_coarse.state_dict()})
                    if model_fine:  checkpoint_dict.update({"model_fine_state_dict": model_fine.state_dict()})
                    if planes_model:
                        if store_planes or getattr(cfg.models.fine,'use_coarse_planes',False):
                            checkpoint_dict["model_fine_state_dict"] = OrderedDict([(k,v) for k,v in checkpoint_dict["model_fine_state_dict"].items() if all([token not in k for token in (tokens_2_exclude+['rot_mats'])])])
                        if store_planes:
                            checkpoint_dict["model_coarse_state_dict"] = OrderedDict([(k,v) for k,v in checkpoint_dict["model_coarse_state_dict"].items() if all([token not in k for token in tokens_2_exclude])])
                            if optimize_planes and save_as_best:
                                planes_opt.save_params(as_best=True)
                        else:
                            checkpoint_dict.update({"coords_normalization": model_fine.box_coords})
                    if optimizer is not None:
                        checkpoint_dict.update({"optimizer": optimizer.state_dict()})

                # ckpt_name = os.path.join(logdir, model_filename + str(iter).zfill(5) + ".ckpt")
                ckpt_name = os.path.join(logdir, model_filename + "%s.ckpt")
                # if (model2save=='decoder' and 'decoder' in what2train) or (model2save=='SR' and 'SR' in what2train):
                safe_saving(ckpt_name%(str(iter).zfill(5)),content=checkpoint_dict,suffix='ckpt',best=False,run_time_signature=run_time_signature)
                if len(experiment_info['last_saved'][model2save])>0:
                    os.remove(experiment_info['last_saved'][model2save].pop(0))
                experiment_info['last_saved'][model2save].append(ckpt_name%(str(iter).zfill(5)))
                if save_as_best:
                    safe_saving(ckpt_name%(''),content=checkpoint_dict,suffix='ckpt',best=True,run_time_signature=run_time_signature)
                del checkpoint_dict
            experiment_info['start_i'] = iter+1
            safe_saving(experiment_info_file,content=experiment_info,suffix='pkl',run_time_signature=run_time_signature)

    print("Done!")



if __name__ == "__main__":
    main()
