import argparse
import os
import time
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from re import search
import models
from load_blender import load_blender_data,BlenderDataset
from nerf_helpers import *
from train_utils import eval_nerf, run_one_iter_of_nerf,find_latest_checkpoint
from mip import IntegratedPositionalEncoding
from deepdiff import DeepDiff
from copy import deepcopy
from shutil import copyfile
import datetime
import sys

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
    
    # Read config file:
    assert (configargs.config is None) ^ (configargs.resume is None)
    cfg = None
    if configargs.config is None:
        config_file = os.path.join(configargs.resume,"config.yml")
    else:
        config_file = configargs.config
    cfg = get_config(config_file)
    planes_model = not hasattr(cfg.models,'coarse') or cfg.models.coarse.type=="TwoDimPlanesModel" # Whether using our feature-plane model. Set to False when running the Mip-NeRF baseline
    if eval_mode: # When running in evaluation mode, overriding some configuration settings with those used for training:
        import imageio
        dataset_config4eval = cfg.dataset
        config_file = os.path.join(cfg.experiment.logdir, cfg.experiment.id,"config.yml")
        results_dir = os.path.join(configargs.results_path, cfg.experiment.id)
        if not os.path.isdir(results_dir):  os.mkdir(results_dir)
        print('Evaluation outputs will be saved into %s'%(results_dir))
        if planes_model:
            cfg = get_config(config_file)
            cfg.dataset = dataset_config4eval
    print('Using configuration file %s'%(config_file))
    print(("Evaluating" if eval_mode else "Running") + " experiment %s"%(cfg.experiment.id))
    im_inconsistency_loss_w = getattr(cfg.nerf.train,'im_inconsistency_loss_w',None)
    what2train = getattr(cfg.nerf.train,'what',[])
    assert all([m in ['LR_planes','decoder','SR'] for m in what2train])
    decoder_training = 'decoder' in what2train

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    if not eval_mode:   print('Logs and models will be saved into %s'%(logdir))

    if configargs.load_checkpoint=="resume":
        configargs.load_checkpoint = logdir
    else:
        if configargs.load_checkpoint=='':
            if os.path.exists(logdir):
                assert len([f for f in os.listdir(logdir) if '.ckpt' in f])==0,'Folder %s already contains saved models.'%(logdir)
            os.makedirs(logdir, exist_ok=True)
        # Write out config parameters.
        with open(os.path.join(logdir, "config%s.yml"%('_Eval' if eval_mode else '')), "w") as f:
            f.write(cfg.dump())
    resume_experiment = os.path.exists(configargs.load_checkpoint)
    if configargs.load_checkpoint!='':
        assert resume_experiment,'Experiment to resume was not found in %s'%(configargs.load_checkpoint)

    # Using a pre-trained model:
    pretrained_model_folder = getattr(cfg.models,'path',None)
    if planes_model and (not decoder_training or pretrained_model_folder):
        if os.path.isfile(pretrained_model_folder):   pretrained_model_folder = "/".join(pretrained_model_folder.split("/")[:-1])
        pretrained_model_config = get_config(os.path.join(pretrained_model_folder,"config.yml"))
        set_config_defaults(source=pretrained_model_config.models,target=cfg.models)
    if not eval_mode:   writer = SummaryWriter(logdir) #Setting TensorBoard logger

    load_saved_models = pretrained_model_folder is not None or resume_experiment
    only_planes_update = what2train==['LR_planes']
    init_new_scenes = not resume_experiment and ('LR_planes' in what2train) and (pretrained_model_folder is None or only_planes_update)
    SR_experiment = "super_resolution" in cfg or (only_planes_update and "super_resolution" in pretrained_model_config)

    # Preparing dataset:
    dataset = BlenderDataset(config=cfg.dataset,scene_id_func=models.get_scene_id,
        eval_mode=eval_mode,scene_norm_coords=cfg.nerf if init_new_scenes else None,planes_logdir=getattr(cfg.models,'planes_path',logdir)) #,assert_LR_ver_of_val=im_inconsistency_loss
    scene_ids = dataset.per_im_scene_id
    i_val = dataset.i_val
    i_train = dataset.i_train
    scenes_set = dataset.scenes_set
    coords_normalization = dataset.coords_normalization
    scene_id_plane_resolution = dataset.scene_id_plane_resolution
    val_only_scene_ids = dataset.val_only_scene_ids
    eval_training_too = getattr(cfg.nerf.validation,'eval_train_scenes',False) and not eval_mode # Whether to add full training views to evaluation (different from the sparse ray sampling from these views during training)
    available_scenes = list(scenes_set)

    planes_updating = 'LR_planes' in what2train

    # Constructing the scene_coupler, which holds the information the pairs of matching LR-HR scenes in the dataset (including in the dataset used for training the pre-trained model, if using one):
    if planes_model and (not planes_updating or pretrained_model_folder):
        for conf,scenes in [c for p in pretrained_model_config.dataset.dir.values() for c in p.items()]:
            conf=eval(conf)
            for sc in interpret_scene_list(scenes):
                available_scenes.append(models.get_scene_id(sc,conf[0],(conf[1],conf[2] if len(conf)>2 else conf[1])))
        available_scenes = list(set(available_scenes))
    scene_coupler = models.SceneCoupler(list(set(available_scenes+val_only_scene_ids)),planes_res=''.join([m[:2] for m in what2train if '_planes' in m]),
        num_pos_planes=getattr(cfg.models.coarse,'num_planes',3) if planes_model else 0,training_scenes=list(i_train.keys()),)
        # multi_im_res=SR_experiment or not planes_model)

    # Assigning logging titles to different evaluation subsets:
    val_strings = []
    ASSUME_LR_IF_NO_COUPLES = True
    only_LR_eval = ASSUME_LR_IF_NO_COUPLES and (len(scene_coupler.downsample_couples)==0 and SR_experiment)
    for id in i_val:
        tags = []
        if id in val_only_scene_ids:    tags.append('blind_validation')
        else:   tags.append('validation')
        if '##Gauss' in id:    tags.append('Gauss')
        if id in scene_coupler.downsample_couples.values() or only_LR_eval: tags.append('LR')
        if len(dataset.module_confinements[id])>0: tags.append('Fixed_'+'_'.join(dataset.module_confinements[id]))
        if dataset.scene_types[id]=='llff': tags.append('real')
        val_strings.append('_'.join(tags))

    if hasattr(cfg.dataset,'max_scenes_eval') and not eval_mode:
        # Pruning some evaluation scenes:
        scenes2keep = subsample_dataset(max_scenes=cfg.dataset.max_scenes_eval,scene_types=val_strings,pick_first=True)
        i_val = OrderedDict([item for i,item in enumerate(i_val.items()) if i in scenes2keep])

    if not eval_mode:
        # Asserting all scenes have the same number of evaluation views:
        val_ims_per_scene = [len(v) for v in i_val.values()]
        assert all([max(val_ims_per_scene)%v==0 for v in val_ims_per_scene]),"Need to be able to repeat scene eval sets to have the same number of eval images for all scnenes."
        val_ims_per_scene = max(val_ims_per_scene)
        i_val = OrderedDict([(k,val_ims_per_scene//len(v)*v) for k,v in i_val.items()])

    if eval_training_too:
        temp = list(i_val.keys())
        for id in temp:
            if id not in i_train:    continue
            im_freq = len(i_train[id])//val_ims_per_scene
            if id in i_val.keys(): #Avoid evaluating training images for scenes which were discarded for evaluation due to max_scenes_eval
                i_val[id+'_train'] = [i_train[id][i] for i in sorted([(i+im_freq//2)%len(i_train[id]) for i in  np.unique(np.round(np.linspace(0,len(i_train[id])-1,val_ims_per_scene)).astype(int))])]

    if not eval_mode and im_inconsistency_loss_w:
        # Adding LR views corresponding to evaluation scenes to training dictionary to facilitate using the consistency loss term:
        for k in val_only_scene_ids:
            i_train.update({k:i_train[scene_coupler.downsample_couples[k]]})
            dataset.scene_probs[k] = getattr(cfg.nerf.train,'im_consistency_iters_freq')/(len(val_only_scene_ids) if getattr(cfg.dataset,'prob_assigned2scene_groups',True) else 1)
            scene_coupler.upsample_couples[scene_coupler.downsample_couples[k]] = k
    training_scenes = list(i_train.keys())
    if planes_model and 'SR' in what2train and len(what2train)==1:
        assert all([sc not in scene_coupler.downsample_couples.values() or dataset.scene_probs[sc]==0 for sc in training_scenes]),'Why train on LR scenes when training only the SR model?'
    if SR_experiment:
        for sc in scenes_set:  
            if sc not in scene_coupler.downsample_couples:  continue
            if init_new_scenes:
                # Unifying spatial coordinates normalization across scene pairs:
                if dataset.scene_types[sc]=='llff':
                    print("Note: Using validation images to determine coord normalization on real images(!)")
                    coords_normalization[sc] = torch.stack([coords_normalization[sc],coords_normalization[scene_coupler.downsample_couples[sc]]],-1)
                    coords_normalization[sc] = torch.stack([torch.min(coords_normalization[sc][0],-1)[0],torch.max(coords_normalization[sc][1],-1)[0]],0)
                    coords_normalization[scene_coupler.downsample_couples[sc]] = 1*coords_normalization[sc]
                else:
                    coords_normalization[sc] = 1*coords_normalization[scene_coupler.downsample_couples[sc]]
            # Filtering out HR scenes from the list of feature plane sets, as those use feature planes from their corresponding LR scenes:
            if sc in scene_id_plane_resolution:
                temp = scene_id_plane_resolution.pop(sc)
                if pretrained_model_folder is not None:
                    scene_id_plane_resolution[scene_coupler.downsample_couples[sc]] = (temp[0]//scene_coupler.ds_factor,temp[1])

    evaluation_sequences = list(i_val.keys())

    # Repeat logging title assignment to different evaluation subsets, after possibly prunning or adding evaluation scenes:
    val_strings = []
    for id in evaluation_sequences:
        bare_id = id.replace('_train','').replace('_HRplane','')
        tags = []
        if id in val_only_scene_ids:    tags.append('blind_validation')
        elif '_train' in id:    tags.append('train_imgs')
        else:   tags.append('validation')
        if '##Gauss' in bare_id:    tags.append('Gauss')
        if bare_id in scene_coupler.downsample_couples.values() or only_LR_eval: tags.append('LR')
        elif '_HRplane' in id:  tags.append('HRplanes')
        if len(dataset.module_confinements[bare_id])>0: tags.append('Fixed_'+'_'.join(dataset.module_confinements[bare_id]))
        if dataset.scene_types[bare_id]=='llff': tags.append('real')
        val_strings.append('_'.join(tags))

    # Which tracked values to consider when determining the best model to use for evaluation or as initialization for subsequent training:
    loss4best = 'im_inconsistency' if im_inconsistency_loss_w else 'fine_loss' if all([v not in what2train for v in ['decoder','SR']]) else 'loss'
    def tag_filter(tag_list,include=[],exclude=[]):
        return list(set([tag for tag in tag_list if all([p in tag for p in include]) and all([p not in tag for p in exclude])]))
    if im_inconsistency_loss_w:
        loss_groups4_best = tag_filter(val_strings,['blind','validation'],['_LR'])
    else:
        loss_groups4_best = tag_filter(val_strings,['validation'],['blind','_LR'])
        if len(loss_groups4_best)==0:
            loss_groups4_best = tag_filter(val_strings,['validation'],['blind'])

    # Printing scenes used for each stage and initializing score tracking:
    def print_scenes_list(title,scenes):
        print('\n%d %s scenes:'%(len(scenes),title))
        print(scenes)

    experiment_info = dict()
    if eval_mode:
        print_scenes_list('evaluation',evaluation_sequences)
    else:
        print_scenes_list('training',[sc for sc in training_scenes if dataset.scene_probs[sc]>0])
        for cat in set(val_strings):
            print_scenes_list('"%s" evaluation'%(cat),[s for i,s in enumerate(evaluation_sequences) if val_strings[i]==cat])
        assert all([val_ims_per_scene==len(i_val[id]) for id in evaluation_sequences]),'Assuming all scenes have the same number of evaluation images'
        running_mean_logs = ['psnr','SR_psnr_gain','planes_SR','fine_loss','fine_psnr','loss','coarse_loss','inconsistency','loss_sr','loss_lr','im_inconsistency']
        experiment_info['running_scores'] = dict([(score,dict([(cat,deque(maxlen=len(training_scenes) if cat=='train' else val_ims_per_scene)) for cat in list(set(val_strings))+['train']])) for score in running_mean_logs])

    image_sampler = ImageSampler(i_train,dataset.scene_probs) # Initializaing training images sampler

    # Tensorboard logger functions:
    def write_scalar(name,new_value,iterOrScNum):
        RUNNING_MEAN = True
        if eval_mode:
            if hasattr(cfg.dataset.llff,'min_eval_frames'): return
            folder_name = os.path.join(results_dir,evaluation_sequences[iterOrScNum])
            with open(os.path.join(folder_name,'metrics.txt'),'a') as f:
                f.write('%s: %f\n'%(name,np.nanmean(new_value) if isinstance(new_value,list) else new_value))
        else:
            val_set,metric = name.split('/')
            experiment_info['running_scores'][metric][val_set].append(new_value)
            writer.add_scalar(name,np.nanmean(experiment_info['running_scores'][metric][val_set]) if RUNNING_MEAN else new_value,iterOrScNum)

    def write_image(name,images,text,iter,psnrs=[],psnr_gains=[]): #fontscale=font_scale,
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
            writer.add_image(name,arange_ims(images,text,psnrs=psnrs),iter)


    # Seed experiment for reproducability:
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Not used for our feature-planes based model:
    if getattr(cfg.nerf,"encode_position_fn",None) is not None:
        assert not planes_model,"Should not require positional encoding"
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

    # Initializing coarse-resolution decoder model:
    if planes_model:
        model_coarse = models.TwoDimPlanesModel(
            use_viewdirs=cfg.nerf.use_viewdirs,
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
            viewdir_proj_combination=getattr(cfg.models.coarse,'viewdir_proj_combination',None),
            num_planes_or_rot_mats=getattr(cfg.models.coarse,'num_planes',3),
            scene_coupler=scene_coupler,
            point_coords_noise=getattr(cfg.nerf.train,'point_coords_noise',0),
            detach_LR_planes=getattr(cfg.nerf.train,'detach_LR_planes',False),
        )
        
    else: # For training a Mip-NeRF or a regular NeRF model:
        if cfg.nerf.encode_position_fn=="mip" and cfg.models.coarse.include_input_xyz:
            cfg.models.coarse.include_input_xyz = False
            cfg.models.fine.include_input_xyz = False
            print("!!! Warning: Not including xyz in model's input since Mip-NeRF does not use the input xyz !!!")

        assert cfg.models.coarse.include_input_xyz==cfg.models.fine.include_input_xyz,"Assuming they are the same"
        assert cfg.models.coarse.include_input_dir==cfg.models.fine.include_input_dir,"Assuming they are the same"
        model_coarse = getattr(models, cfg.models.coarse.type)(
            num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
            include_input_xyz=cfg.models.coarse.include_input_xyz,
            include_input_dir=cfg.models.coarse.include_input_dir,
            use_viewdirs=cfg.nerf.use_viewdirs,
        )
    model_coarse.optional_no_grad = null_with # A trick to allow using no_grad upon demand.
    print("Coarse model: %d parameters"%num_parameters(model_coarse))
    model_coarse.to(device)
    
    # Initializing a fine-resolution model, if specified:
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
                    dec_density_layers=getattr(cfg.models.fine,'dec_density_layers',4),
                    dec_rgb_layers=getattr(cfg.models.fine,'dec_rgb_layers',4),
                    dec_channels=getattr(cfg.models.fine,'dec_channels',128),
                    skip_connect_every=getattr(cfg.models.fine,'skip_connect_every',None),
                    num_plane_channels=getattr(cfg.models.fine,'num_plane_channels',48),
                    num_viewdir_plane_channels=getattr(cfg.models.fine,'num_viewdir_plane_channels',None),
                    rgb_dec_input=getattr(cfg.models.fine,'rgb_dec_input','projections'),
                    proj_combination=getattr(cfg.models.fine,'proj_combination','sum'),
                    plane_interp=getattr(cfg.models.fine,'plane_interp','bilinear'),
                    align_corners=getattr(cfg.models.fine,'align_corners',True),
                    viewdir_proj_combination=getattr(cfg.models.fine,'viewdir_proj_combination',None),
                    num_planes_or_rot_mats=model_coarse.rot_mats() if getattr(cfg.models.fine,'use_coarse_planes',True) else getattr(cfg.models.fine,'num_planes',3),
                    scene_coupler=scene_coupler,
                    point_coords_noise=getattr(cfg.nerf.train,'point_coords_noise',0),
                    detach_LR_planes=getattr(cfg.nerf.train,'detach_LR_planes',False),
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

    rendering_loss_w = 1
    SR_model,SR_optimizer = None,None
    if SR_experiment: # For experiments involving rendering HR views based on learning from LR views (not necessarily using a dedicated SR model):
        if 'SR' not in what2train:
            assert not (hasattr(cfg,'super_resolution') and hasattr(cfg.super_resolution,'model') and hasattr(cfg.super_resolution.model,'path')),'Should not specify a pre-trained SR model path if not training the SR model. Using the same path used for the decoder model in such case.'
            if not hasattr(cfg,'super_resolution'):   setattr(cfg,'super_resolution',CfgNode())
            if not hasattr(cfg.super_resolution,'model'):   setattr(cfg.super_resolution,'model',CfgNode())
            rsetattr(cfg.super_resolution,'model.path',pretrained_model_folder)
            SR_model_config = get_config(os.path.join(cfg.super_resolution.model.path,"config.yml"))
            set_config_defaults(source=SR_model_config.super_resolution,target=cfg.super_resolution)
        rendering_loss_w = getattr(cfg.super_resolution,'rendering_loss',1)
        plane_channels = getattr(cfg.models.coarse,'num_plane_channels',48)
        if not eval_mode:   saved_rgb_fine = dict(zip(evaluation_sequences,[{} for i in evaluation_sequences])) # Saving pre-SR evaluation images to avoid re-rendering them each time.

        # Determining the FEATURE PLANES super-resolution scale factor:
        sf_config = getattr(cfg.super_resolution.model,'scale_factor','linear')
        assert sf_config in ['linear','sqrt'] or isinstance(sf_config,int),'Unsupported value.'
        if sf_config=='linear':
            SR_factor = int(scene_coupler.ds_factor)
        elif sf_config=='sqrt':
            SR_factor = int(np.sqrt(scene_coupler.ds_factor))
        else:
            SR_factor = sf_config

        if cfg.super_resolution.model.type!='None':
            # Initializing the SR model (and optimizer):
            SR_model = models.PlanesSR(
                model_arch=getattr(models, cfg.super_resolution.model.type),scale_factor=SR_factor,
                in_channels=plane_channels,
                out_channels=plane_channels,
                sr_config=cfg.super_resolution,
                plane_interp=getattr(cfg.super_resolution,'plane_resize_mode',model_fine.plane_interp),
            )
            print("SR model: %d parameters"%(num_parameters(SR_model)))
            SR_model.to(device)
            if not eval_mode and 'SR' in what2train:
                SR_optimizer = getattr(torch.optim, cfg.optimizer.type)(
                    [p for k,p in SR_model.named_parameters() if 'NON_LEARNED' not in k],
                    lr=getattr(cfg.super_resolution,'lr',cfg.optimizer.lr)
                )
            else:
                assert all([sc not in scene_coupler.downsample_couples.keys() for sc in training_scenes]),"Why train on HR scenes when training only the planes? Currently not assigning the SR model's LR planes during training."

    # Collecting optimizable parameters list:
    if decoder_training or not planes_model:
        if not eval_mode:
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
            trainable_parameters_ = collect_params(model_coarse,filter='non_planes' if planes_model else 'all')
            if model_fine is not None:
                if planes_model:
                    if cfg.models.fine.type!="use_same":
                        if getattr(cfg.models.fine,'use_coarse_planes',True):
                            trainable_parameters_ += collect_params(model_fine,filter='non_planes')
                        else:
                            trainable_parameters_ += collect_params(model_fine,filter='all')
                else: 
                    trainable_parameters_ += list(model_fine.parameters())

    if not eval_mode and (decoder_training or not planes_model):
        # Initializing decoder optimizer:
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            trainable_parameters_, lr=cfg.optimizer.lr
        )
    else:
        optimizer = None

    if planes_model:
        models2save = (['decoder'] if 'decoder' in what2train else [])+(['SR'] if SR_experiment and 'SR' in what2train and SR_model is not None else [])
    else:
        models2save = ['decoder']
    experiment_info.update({'start_i':0,'eval_counter':0,'best_loss':(0,np.finfo(np.float32).max),'last_saved':dict(zip(models2save,[[] for i in models2save]))})
    experiment_info_file = os.path.join(logdir, "exp_info.pkl")

    if load_saved_models:
        # Loading saved models, either when resuming training or when initializing from pre-trained models:
        if resume_experiment and not eval_mode:
            experiment_info = deep_update(experiment_info,safe_loading(experiment_info_file,suffix='pkl'))

        load_best = eval_mode or not resume_experiment
        if SR_experiment and SR_model is not None:
            if SR_experiment and ('SR' not in what2train or resume_experiment or hasattr(cfg.super_resolution.model,'path')):
                if resume_experiment and 'SR' in what2train:
                    SR_checkpoint_path = configargs.load_checkpoint
                elif getattr(cfg.super_resolution.model,'path',None) is not None:
                    SR_checkpoint_path = cfg.super_resolution.model.path
                else:
                    SR_checkpoint_path = pretrained_model_folder
                SR_checkpoint_path = find_latest_checkpoint(SR_checkpoint_path,sr=True,find_best=load_best or ('SR' not in what2train))
                SR_model_checkpoint = safe_loading(SR_checkpoint_path,suffix='ckpt_best' if load_best else 'ckpt')
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

        if configargs.load_checkpoint=='' or (planes_model and 'decoder' not in what2train): # Initializing a representation model with a pre-trained model
            checkpoint_filename = find_latest_checkpoint(pretrained_model_folder,sr=False,find_best=load_best or 'decoder' not in what2train)
            print("Initializing model training from model %s"%(checkpoint_filename))
            checkpoint = safe_loading(checkpoint_filename,suffix='ckpt_best' if load_best else 'ckpt')
        else:
            checkpoint_filename = find_latest_checkpoint(configargs.load_checkpoint,sr=False,find_best=load_best or (planes_model and 'decoder' not in what2train))
            checkpoint = safe_loading(checkpoint_filename,suffix='ckpt_best' if load_best else 'ckpt')
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
                    if diff in ["root['use_existing_planes']","root['planes_path']"]: continue
                elif ch_type=='dictionary_item_removed' and "root['fine']" in str(config_diffs[ch_type]):   continue
                elif not planes_model and ch_type=='values_changed' and 'include_input_xyz' in diff and cfg.nerf.encode_position_fn=="mip": continue
                print(ch_type,diff)
                ok = False
        if not (ok or eval_mode): # Not aborting the run in eval_mode because then the configuration is copied from the existing one anyway. They can differ if the model configuration is changed by the code at some stage.
            raise Exception('Inconsistent model configuration.')

        def load_saved_parameters(model,saved_params,reduced_set=False):
            if not all([search('density_dec\.(\d)+\.(\d)+\.',p) is not None for p in saved_params if 'density_dec' in p]):
                saved_params = OrderedDict([(k if 'NON_LEARNED' in k else k.replace('.','.0.',1),v) for k,v in saved_params.items()])
            mismatch = model.load_state_dict(saved_params,strict=False)
            allowed_missing = []
            if planes_model:    allowed_missing.append('planes_.sc')
            if reduced_set: allowed_missing.append('rot_mats')
            assert (len(mismatch.missing_keys)==0 or all([any([tok in k for tok in allowed_missing]) for k in mismatch.missing_keys]))\
                and all(['planes_.sc' in k for k in mismatch.unexpected_keys])

        if planes_model:
            checkpoint["model_coarse_state_dict"] = model_coarse.rot_mat_backward_support(checkpoint["model_coarse_state_dict"])
        load_saved_parameters(model_coarse,checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            load_saved_parameters(model_fine,checkpoint["model_fine_state_dict"],reduced_set=True)
        if "optimizer" in checkpoint and optimizer is not None:
            print("Loading optimizer's checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer"])
        del checkpoint # Importent: Releasing GPU memory occupied by loaded data.

    if planes_model and SR_model is not None:
        if getattr(cfg.super_resolution,'apply_2_coarse',False):
            # If super-resolving feature planes to be used by the coarse-resolution decoder too (default is no):
            model_coarse.assign_SR_model(SR_model,SR_viewdir=cfg.super_resolution.SR_viewdir,)
        else:
            assert getattr(cfg.super_resolution.training,'loss','fine')=='fine',"Shouldn't use the ourput of coarse decoder for training the SR model when not feeding it with super-resolved feature planes."
            if not decoder_training:  model_coarse.optional_no_grad = torch.no_grad
        model_fine.assign_SR_model(SR_model,SR_viewdir=getattr(cfg.super_resolution,'SR_viewdir',False),)

    run_time_signature = time.time() # A trick to allow starting new training jobs that would automatically cause the old jobs to exit

    if planes_model:
        # Determining folder paths where feature planes are stored. Creating a paths hierarchy to allow using existing feature planes for some scenes while training new planes for other:
        planes_folder = []
        if planes_updating: planes_folder.append(logdir)
        if getattr(cfg.models,'planes_path',None) is not None:  planes_folder.append(getattr(cfg.models,'planes_path'))
        if pretrained_model_folder is not None: planes_folder.append(pretrained_model_folder)
        planes_folder = [os.path.join(f,'planes') for f in planes_folder]
        if eval_mode: assert os.path.isdir(planes_folder[0])
        if not os.path.isdir(planes_folder[0]):
            os.mkdir(planes_folder[0])
        if 'LR_planes' in what2train and not only_planes_update and not resume_experiment and pretrained_model_folder:
            # When starting a new non-frozen decoder and feature planes run, using a pre-trained decoder model:
            params_init_path = [os.path.join(pretrained_model_folder,'planes')]
            if getattr(cfg.models,'planes_path',None) is not None:
                params_init_path.insert(0,os.path.join(getattr(cfg.models,'planes_path'),'planes'))
        else:
            params_init_path = None
        scenes_cycle_counter = Counter() # Keeping track of cycling through the different scenes in the dataset
        optimize_planes = any(['planes' in m for m in what2train]) and not eval_mode
        lr_scheduler = getattr(cfg.optimizer,'lr_scheduler',None)
        if getattr(cfg.models,'use_existing_planes',False):
            # Using feature planes from pre-trained model, without updating them:
            use_frozen_planes = os.path.join(pretrained_model_folder,'planes')
        else:
            use_frozen_planes = ''
        if lr_scheduler is not None:
            lr_scheduler['patience'] = int(np.ceil(lr_scheduler['patience']/cfg.experiment.print_every))

        planes_opt = models.PlanesOptimizer(optimizer_type=cfg.optimizer.type,
            scene_id_plane_resolution=scene_id_plane_resolution,options=cfg.nerf.train.store_planes,save_location=planes_folder,
            lr=getattr(cfg.optimizer,'planes_lr',getattr(cfg.optimizer,'lr',None)),model_coarse=model_coarse,model_fine=model_fine,
            use_coarse_planes=getattr(cfg.models.fine,'use_coarse_planes',True),
            init_params=init_new_scenes,optimize=optimize_planes,training_scenes=training_scenes,
            coords_normalization=None if not init_new_scenes else coords_normalization,
            do_when_reshuffling=lambda:scenes_cycle_counter.step(print_str='Number of scene cycles performed: '),
            STD_factor=getattr(cfg.nerf.train,'STD_factor',0.1),
            available_scenes=available_scenes,planes_rank_ratio=getattr(cfg.models.coarse,'planes_rank_ratio',None),
            copy_params_path=params_init_path,run_time_signature=run_time_signature,
            lr_scheduler=lr_scheduler,use_frozen_planes=use_frozen_planes,
        )

    if SR_experiment and getattr(cfg.super_resolution,'input_normalization',False) and not resume_experiment:
        #Initializing a new SR model that uses input normalization
        SR_model.normalization_params(planes_opt.get_plane_stats(viewdir=getattr(cfg.super_resolution,'SR_viewdir',False)))

    downsampling_offset = lambda ds_factor: (ds_factor-1)/(2*ds_factor) # Importent: Detrmining the rendered rays offset to match the sub-pixel offset caused by downsampling the input images.
    saved_target_ims = dict(zip(set(val_strings),[set() for i in set(val_strings)])) # Keeping track of target images logged so far to avoid re-logging them.
    virtual_batch_size = getattr(cfg.nerf.train,'virtual_batch_size',1)
    if im_inconsistency_loss_w:
        def avg_downsampling(pixels):
            # For the downsampling consistency loss term (enforced w.r.t. LR views corresponding to validation scenes), downsampling the rendered HR images using simple averaging over image patches of size (scale_factor X scale_facotr), without any antialias filtering. Found this to work best.
            return torch.mean(pixels.reshape(-1,scene_coupler.ds_factor,scene_coupler.ds_factor,3),dim=(1,2))

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
        if SR_experiment and SR_model is not None:
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
                coarse_loss,fine_loss,loss,psnr,rgb_coarse,rgb_fine,rgb_SR,target_ray_values,im_inconsistency_loss,sr_scene_inds =\
                    defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
                for eval_num,img_idx in enumerate(tqdm(img_indecis[eval_cycle], # Rendering and calculating metrics for evaluation images:
                    desc='Evaluating '+('scene (%d/%d): %s'%(eval_cycle+1,eval_cycles,evaluation_sequences[eval_cycle]) if eval_mode else 'scenes'))):
                    if eval_mode:
                        val_ind = lambda dummy: eval_cycle
                        scene_num = eval_cycle
                    else:
                        scene_num = eval_num
                    cur_scene_id = scene_ids[img_idx]
                    sr_scene = (not planes_model or SR_experiment) and cur_scene_id in scene_coupler.downsample_couples
                    sr_scene_inds[val_strings[scene_num]].append(sr_scene)
                    img_target,pose_target,cur_H,cur_W,cur_focal,cur_ds_factor = dataset.item(img_idx,device)
                    ray_origins, ray_directions = get_ray_bundle(
                        cur_H, cur_W, cur_focal, pose_target,downsampling_offset=downsampling_offset(cur_ds_factor),
                    )
                    if planes_model and (not eval_mode or eval_num==0):
                        # Loading feature-planes corresponding to the evaluated scene:
                        planes_opt.load_scene(cur_scene_id,load_best=not optimize_planes)
                        planes_opt.cur_id = cur_scene_id
                    # Rendering the scene view. Using the SR model to SR the feature-planes, if applicable:
                    def render_view():
                        rgb_c, _, _, rgb_f, _, _,rgb_SR,_,_ = eval_nerf(
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
                            scene_id=cur_scene_id,
                            scene_config=cfg.dataset[dataset.scene_types[cur_scene_id]],
                        )
                        return rgb_c,rgb_f,rgb_SR

                    rgb_coarse_, rgb_fine_, rgb_SR_ = render_view()
                    target_ray_values[val_strings[scene_num]].append(img_target[...,:3])
                    loss[val_strings[scene_num]].append(None)
                    if sr_scene:
                        if planes_model:
                            rgb_SR_ = 1*rgb_fine_
                            if SR_model is not None:
                                if eval_mode or val_ind(val_strings[scene_num]) not in saved_rgb_fine[evaluation_sequences[scene_num]]:
                                    # Rendering the scene view without super-resolving the feature-planes, as reference:
                                    record_fine = True
                                    model_coarse.skip_SR(True)
                                    model_fine.skip_SR(True)
                                    rgb_coarse_, rgb_fine_, _ = render_view()
                                    model_coarse.skip_SR(False)
                                    model_fine.skip_SR(False)
                                    if not (eval_mode or planes_updating or decoder_training):
                                        # Save the reference views to avoid re-rendering them in future evaluation rounds:
                                        saved_rgb_fine[evaluation_sequences[scene_num]][val_ind(val_strings[scene_num])] = 1*rgb_fine_.detach()
                                else:
                                    record_fine = False
                                    rgb_fine_ = 1*saved_rgb_fine[evaluation_sequences[scene_num]][val_ind(val_strings[scene_num])]

                        fine_loss[val_strings[scene_num]].append(img2mse(rgb_fine_[..., :3], img_target[..., :3]).item())
                        if planes_model:
                            loss[val_strings[scene_num]][-1] = img2mse(rgb_SR_[..., :3], img_target[..., :3]).item()
                        if im_inconsistency_loss_w is None:
                            im_inconsistency_loss_ = None
                        else:
                            im_inconsistency_loss_ = calc_im_inconsistency_loss(gt_hr=img_target[..., :3].permute(2,0,1)[None,...],
                                sr=(rgb_SR_ if planes_model else rgb_fine_)[..., :3].permute(2,0,1)[None,...],ds_factor=scene_coupler.ds_factor,plane_interp='bilinear')
                        if im_inconsistency_loss_ is not None:
                            im_inconsistency_loss[val_strings[scene_num]].append(im_inconsistency_loss_.item())
                    else:
                        coarse_loss[val_strings[scene_num]].append(img2mse(rgb_coarse_[..., :3], img_target[..., :3]).item())
                        fine_loss[val_strings[scene_num]].append(0.0)
                        if rgb_fine is not None:
                            fine_loss[val_strings[scene_num]][-1] = img2mse(rgb_fine_[..., :3], img_target[..., :3]).item()

                    rgb_coarse[val_strings[scene_num]].append(rgb_coarse_)
                    rgb_fine[val_strings[scene_num]].append(rgb_fine_)
                    rgb_SR[val_strings[scene_num]].append(rgb_SR_)
                    psnr[val_strings[scene_num]].append(mse2psnr(fine_loss[val_strings[scene_num]][-1] if loss[val_strings[scene_num]][-1] is None else loss[val_strings[scene_num]][-1]))
                    if planes_model and SR_model is not None:
                        last_scene_eval = img_idx==img_indecis[eval_cycle][-1]
                        if (planes_model and (not eval_mode or last_scene_eval)):
                            SR_model.clear_SR_planes(all_planes=True)
                            planes_opt.generated_planes.clear()
                            planes_opt.downsampled_planes.clear()
                SAVE_COARSE_IMAGES = False
                cur_val_sets = [val_strings[eval_cycle]] if eval_mode else set(val_strings)
                if eval_mode:
                    if not os.path.isdir(os.path.join(results_dir,evaluation_sequences[eval_cycle])):  os.mkdir(os.path.join(results_dir,evaluation_sequences[eval_cycle]))
                    with open(os.path.join(results_dir,evaluation_sequences[eval_cycle],'metrics.txt'),'w') as f:
                        f.write('Evaluated at '+str(datetime.datetime.now())+':\n')
                        if hasattr(cfg.dataset.llff,'min_eval_frames'):
                            f.write('Not saving metrics since rendered viewing directions are interpolated and therefore do not match test images.')

                for val_set in cur_val_sets:
                    # Computing average metrics and logging them, along with the rendered images:
                    writing_index = val_ind(val_set) if eval_mode else iter
                    if sum(sr_scene_inds[val_set])>0:
                        if any([v is not None for v in rgb_SR[val_set]]):
                            SR_psnr_gain = [psnr[val_set][i]-mse2psnr(l) for i,l in enumerate(fine_loss[val_set]) if sr_scene_inds[val_set][i]]
                            if not eval_mode:
                                SR_psnr_gain = np.nanmean(SR_psnr_gain)
                            write_scalar("%s/SR_psnr_gain"%(val_set),SR_psnr_gain,writing_index)
                            write_image(name="%s/rgb_SR"%(val_set),
                                images=[torch.zeros_like(rgb_fine[val_set][i]) if im is None else im for i,im in enumerate(rgb_SR[val_set])],
                                text=str(val_ind(val_set)),psnrs=[None if im is None else SR_psnr_gain[i] if eval_mode else psnr[val_set][i] for i,im in enumerate(rgb_SR[val_set])],
                                iter=iter,)
                        if val_set in im_inconsistency_loss:
                            write_scalar("%s/im_inconsistency"%(val_set), np.nanmean(im_inconsistency_loss[val_set]), writing_index)
                    write_scalar("%s/fine_psnr"%(val_set), np.nanmean([mse2psnr(l) for l in fine_loss[val_set]]), writing_index)
                    fine_loss_is_loss = not any([v is not None for v in loss[val_set]])
                    if not fine_loss_is_loss:
                        write_scalar("%s/loss"%(val_set), np.nanmean(loss[val_set]), writing_index)
                    write_scalar("%s/psnr"%(val_set), np.nanmean(psnr[val_set]), writing_index)
                    if len(coarse_loss[val_set])>0:
                        write_scalar("%s/coarse_loss"%(val_set), np.nanmean(coarse_loss[val_set]), writing_index)
                    if len(rgb_fine[val_set])>0:
                        if record_fine:
                            recorded_iter = writing_index if (planes_updating or decoder_training or not planes_model) else val_ind(val_set) 
                            write_scalar("%s/fine_loss"%(val_set), np.nanmean(fine_loss[val_set]), recorded_iter)
                            if eval_mode and evaluation_sequences[val_ind(val_set)] in scene_coupler.downsample_couples.values():
                                write_image(name="%s/rgb_bicubic"%(val_set),
                                    images=[torch.from_numpy(bicubic_interp(im.cpu().numpy(),sf=scene_coupler.ds_factor)) for im in rgb_fine[val_set]],
                                    text=str(val_ind(val_set)),
                                    psnrs=[],iter=recorded_iter,)
                                write_image(name="%s/rgb_LR"%(val_set),
                                    images=[im.unsqueeze(2).unsqueeze(1).repeat([1,scene_coupler.ds_factor,1,scene_coupler.ds_factor,1]).reshape([im.shape[0]*scene_coupler.ds_factor,im.shape[1]*scene_coupler.ds_factor,-1]) for im in rgb_fine[val_set]],
                                    text=str(val_ind(val_set)),
                                    psnrs=[],iter=recorded_iter,)
                            write_image(name="%s/rgb_fine"%(val_set),images=rgb_fine[val_set],text=str(val_ind(val_set)),
                                psnrs=[mse2psnr(l) for l in fine_loss[val_set]],iter=recorded_iter,)
                    if SAVE_COARSE_IMAGES:
                        write_image(name="%s/rgb_coarse"%(val_set),images=rgb_coarse[val_set],text=str(val_ind(val_set)),
                            psnrs=[mse2psnr(l) for l in coarse_loss[val_set]],iter=iter,)
                    if not eval_mode and val_ind(val_set) not in saved_target_ims[val_set]:
                        write_image(name="%s/img_target"%(val_set),images=target_ray_values[val_set],text=str(val_ind(val_set)),iter=val_ind(val_set))
                        saved_target_ims[val_set].add(val_ind(val_set))
                    if not eval_mode:
                        tqdm.write(
                            "%s:\tValidation loss: "%(val_set)
                            + str(np.nanmean(fine_loss[val_set]) if fine_loss_is_loss else np.nanmean(loss[val_set]))
                            + "\tValidation PSNR: "
                            + str(np.nanmean(psnr[val_set]))
                            + "\tTime: "
                            + str(time.time() - start)
                        )
        return fine_loss if fine_loss_is_loss else loss

    def train():
        first_v_batch_iter = iter%virtual_batch_size==0
        last_v_batch_iter = iter%virtual_batch_size==(virtual_batch_size-1)
        if SR_experiment and 'SR' in what2train and SR_model is not None:
            SR_model.train()
        if decoder_training:
            model_coarse.train()
            if model_fine:
                model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        cur_scene_id,img_idx = image_sampler.sample() # Sampling a training image according to scene sampling probabilities
        sr_iter = cur_scene_id in scene_coupler.downsample_couples # Is this view rendered in HR
        img_target,pose_target,cur_H,cur_W,cur_focal,cur_ds_factor = dataset.item(img_idx,device)
        im_consistency_iter = im_inconsistency_loss_w and cur_scene_id in val_only_scene_ids
        if im_consistency_iter:
            # For image consistency iterations, rendering in HR (although GT target image is LR):
            cur_H *= scene_coupler.ds_factor
            cur_W *= scene_coupler.ds_factor
            cur_focal *= scene_coupler.ds_factor
            cur_ds_factor //= scene_coupler.ds_factor

        # Calculating ray origins, directions and corresponding pixel coordiantes for all view rays:
        ray_origins, ray_directions = get_ray_bundle(cur_H, cur_W, cur_focal, pose_target,downsampling_offset=downsampling_offset(cur_ds_factor),)
        if im_consistency_iter:
            coords = torch.stack(
                meshgrid_xy(torch.arange(img_target.shape[0]).to(device), torch.arange(img_target.shape[1]).to(device)),
                dim=-1,
            )
        else:
            coords = torch.stack(
                meshgrid_xy(torch.arange(cur_H).to(device), torch.arange(cur_W).to(device)),
                dim=-1,
            )
        coords = coords.reshape((-1, 2))
        # Randomly sampling rays to render:
        if im_consistency_iter:
            # To render patches of size (ds_factor X ds_factor), first drawing patches' upper left corner coordinates:
            select_inds = np.random.choice(
                img_target.shape[0]*img_target.shape[1], size=(min(img_target.shape[0]*img_target.shape[1],cfg.nerf.train.num_random_rays//(scene_coupler.ds_factor**2))), replace=False
            )
            select_inds = coords[select_inds]
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
            select_inds = scene_coupler.ds_factor*select_inds[:,None,None,:]
            # Extending the selected corners to full patch coordinates:
            select_inds = torch.cat([select_inds[...,:1]+torch.arange(scene_coupler.ds_factor).reshape(1,-1,1,1).type(coords.type()).repeat(1,1,scene_coupler.ds_factor,1),
                select_inds[...,1:]+torch.arange(scene_coupler.ds_factor).reshape(1,1,-1,1).type(coords.type()).repeat(1,scene_coupler.ds_factor,1,1)],-1).reshape(-1,2)
        else:
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
        if planes_model:
            planes_opt.cur_id = cur_scene_id
            planes_opt.zero_grad()
        if hasattr(model_fine,'SR_model') and sr_iter and optimize_planes:
            # Loading current feature planes to SR model to allow super-resolving them:
            model_fine.assign_LR_planes(scene=cur_scene_id)
        # Rendering the chosen rays:
        rgb_coarse, _, acc_coarse, rgb_fine, _, acc_fine,rgb_SR,_,_ = run_one_iter_of_nerf(
            cur_H,
            cur_W,
            cur_focal,
            model_coarse,
            model_fine,
            batch_rays,
            cfg,
            mode="train",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            scene_id=cur_scene_id,
            scene_config=cfg.dataset[dataset.scene_types[cur_scene_id]],
        )
        coarse_loss_weights,fine_loss_weights = None,None
        gt_background_pixels = target_s[...,3]==0 if target_s.shape[1]==4 else None

        target_ray_values = target_s
        if im_consistency_iter:
            # For image consistency iterations, averaging each rendered patch into a single pixel to be compared with the corresponding pixel in the GT LR image
            rgb_coarse,rgb_fine = avg_downsampling(rgb_coarse),avg_downsampling(rgb_fine)
        coarse_loss,fine_loss = None,None
        if rendering_loss_w is not None:
            if not planes_model or any([m in what2train for m in ['decoder','LR_planes',]]) or getattr(cfg.super_resolution.training,'loss','both')!='fine':
                coarse_loss = mse_loss(
                    rgb_coarse, target_ray_values[..., :3],weights=coarse_loss_weights,
                )
            if rgb_fine is not None and (not planes_model or any([m in what2train for m in ['decoder','LR_planes']]) or getattr(cfg.super_resolution.training,'loss','both')!='coarse'):
                fine_loss = mse_loss(
                    rgb_fine, target_ray_values[..., :3],weights=fine_loss_weights,
                )
        rendering_loss = ((coarse_loss if coarse_loss is not None else 0.0) + (fine_loss if fine_loss is not None else 0.0))
        psnr = None
        if isinstance(rendering_loss,torch.Tensor):
            if im_consistency_iter:
                write_scalar("train/im_inconsistency", rendering_loss.item(), iter)
            else:
                write_scalar("train/loss", rendering_loss.item(), iter)
                write_scalar("train/loss_%s"%('sr' if cur_scene_id in scene_coupler.downsample_couples else 'lr'), rendering_loss.item(), iter)
                psnr = mse2psnr(rendering_loss.item())
                write_scalar("train/psnr", psnr, iter)
        loss = (im_inconsistency_loss_w if im_consistency_iter else rendering_loss_w)*rendering_loss

        loss.backward()
        new_drawn_scenes = None
        if planes_model:
            new_drawn_scenes = planes_opt.step()
        if last_v_batch_iter:
            if optimizer is not None:
                decoder_step = 'decoder' not in dataset.module_confinements[cur_scene_id]
                if 'SR' in what2train:
                    if getattr(cfg.nerf.train,'separate_decoder_sr',False):
                        decoder_step &= not sr_iter
                if decoder_step:
                    optimizer.step()
            if SR_optimizer is not None and sr_iter and 'SR' not in dataset.module_confinements[cur_scene_id]:
                SR_optimizer.step()
        if not im_consistency_iter:
            if coarse_loss is not None:
                write_scalar("train/coarse_loss", coarse_loss.item(), iter)
            if fine_loss is not None:
                write_scalar("train/fine_loss", fine_loss.item(), iter)
                write_scalar("train/fine_psnr", mse2psnr(fine_loss.item()), iter)
        return loss.item(),psnr,new_drawn_scenes

    training_time,last_evaluated = 0,1*experiment_info['start_i']
    recently_saved, = time.time(),
    print_cycle_loss,print_cycle_psnr = [],[],
    evaluation_time = 0
    recent_loss_avg = np.nan

    # Begin training loop (or run a single iteration in the eval_mode case):
    for iter in trange(experiment_info['start_i'],cfg.experiment.train_iters):
        # Determining whether to run evaluation:
        if isinstance(cfg.experiment.validate_every,list):
            evaluate_now = evaluation_time<=training_time*cfg.experiment.validate_every[0] or iter-last_evaluated>=cfg.experiment.validate_every[1]
        else:
            evaluate_now = iter % cfg.experiment.validate_every == 0
        evaluate_now |= iter == cfg.experiment.train_iters - 1

        if evaluate_now: # Run evaluation:
            last_evaluated = 1*iter
            start_time = time.time()
            loss = evaluate()
            evaluation_time = time.time()-start_time
            if planes_model and not eval_mode:
                planes_opt.draw_scenes(assign_LR_planes=not SR_experiment or not optimize_planes)
                new_drawn_scenes = planes_opt.cur_scenes
                image_sampler.update_active(new_drawn_scenes)
            elif not planes_model:
                image_sampler.update_active(training_scenes)
            training_time = 0
            experiment_info['eval_counter'] += 1
        if eval_mode:   break

        # Training:
        start_time = time.time()
        loss,psnr,new_drawn_scenes = train()
        if new_drawn_scenes is not None:
            image_sampler.update_active(new_drawn_scenes)

        if psnr is not None:
            print_cycle_loss.append(loss)
            print_cycle_psnr.append(psnr)
        training_time += time.time()-start_time

        if iter % cfg.experiment.print_every == 0 or iter == cfg.experiment.train_iters - 1: # Print training metrics:
            tqdm.write("[TRAIN] Iter: " + str(iter) + " Loss: " + str(np.mean(print_cycle_loss)) + " PSNR: " + str(np.mean(print_cycle_psnr)))
            if planes_model:
                planes_opt.lr_scheduler_step(np.mean(print_cycle_loss))
            print_cycle_loss,print_cycle_psnr = [],[]

        # Determine whether to save models and feature planes:
        save_now = scenes_cycle_counter.check_and_reset() if (planes_model and decoder_training) else False
        save_now |= iter % cfg.experiment.save_every == 0 if isinstance(cfg.experiment.save_every,int) else (time.time()-recently_saved)/60>cfg.experiment.save_every
        save_now |= iter == cfg.experiment.train_iters - 1

        if save_now:
            save_as_best,quit_training = False,False
            if len(experiment_info['running_scores'][loss4best][loss_groups4_best[0]])==val_ims_per_scene:
                recent_loss_avg = np.mean([l for term in loss_groups4_best for l in experiment_info['running_scores'][loss4best][term]])
                if recent_loss_avg<experiment_info['best_loss'][1]:
                    experiment_info['best_loss'] = (iter,recent_loss_avg)
                    save_as_best = True
                elif getattr(cfg.experiment,'no_improvement_iters',None) is not None:
                    if iter-experiment_info['best_loss'][0]>=len(training_scenes)*getattr(cfg.experiment,'no_improvement_iters'): quit_training = True
            if save_as_best:
                print("================Saving new best checkpoint at iteration %d, with average evaluation loss %.3e====================="%(experiment_info['best_loss'][0],experiment_info['best_loss'][1]))
            else:
                print("================Best checkpoint is still %d, with average evaluation loss %.3e (recent average is %.3e)====================="%(experiment_info['best_loss'][0],experiment_info['best_loss'][1],recent_loss_avg))
            recently_saved = time.time()
            if planes_model and optimize_planes and save_as_best:
                planes_opt.save_params(as_best=True)
            for model2save in models2save:
                model_filename = '%scheckpoint'%('SR_' if model2save=='SR' else '')
                checkpoint_dict = {}
                if model2save=="SR":
                    checkpoint_dict.update({"SR_model":SR_model.state_dict()})
                    if SR_optimizer is not None:
                        checkpoint_dict.update({"SR_optimizer": SR_optimizer.state_dict()})
                else:
                    tokens_2_exclude = ['planes_.','SR_model']
                    checkpoint_dict.update({"model_coarse_state_dict": model_coarse.state_dict()})
                    if model_fine:  checkpoint_dict.update({"model_fine_state_dict": model_fine.state_dict()})
                    if planes_model:
                        checkpoint_dict["model_fine_state_dict"] = OrderedDict([(k,v) for k,v in checkpoint_dict["model_fine_state_dict"].items() if all([token not in k for token in (tokens_2_exclude+['rot_mats'])])])
                        checkpoint_dict["model_coarse_state_dict"] = OrderedDict([(k,v) for k,v in checkpoint_dict["model_coarse_state_dict"].items() if all([token not in k for token in tokens_2_exclude])])
                    if optimizer is not None:
                        checkpoint_dict.update({"optimizer": optimizer.state_dict()})

                ckpt_name = os.path.join(logdir, model_filename + "%s.ckpt")
                safe_saving(ckpt_name%(str(iter).zfill(5)),content=checkpoint_dict,suffix='ckpt',best=False,run_time_signature=run_time_signature)
                if len(experiment_info['last_saved'][model2save])>0:
                    chkp2remove = experiment_info['last_saved'][model2save].pop(0)
                    if os.path.exists(chkp2remove): os.remove(chkp2remove)
                experiment_info['last_saved'][model2save].append(ckpt_name%(str(iter).zfill(5)))
                if save_as_best:
                    safe_saving(ckpt_name%(''),content=checkpoint_dict,suffix='ckpt',best=True,run_time_signature=run_time_signature)
                del checkpoint_dict
            experiment_info['start_i'] = iter+1
            safe_saving(experiment_info_file,content=experiment_info,suffix='pkl',run_time_signature=run_time_signature)
            if quit_training:
                sys.exit('Done training, after no improvement in %s:%s was observed in the last %d iterations.'%(loss_groups4_best,loss4best,iter-experiment_info['best_loss'][0]))

    print("Done!")



if __name__ == "__main__":
    main()
