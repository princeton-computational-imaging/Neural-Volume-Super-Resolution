import argparse
import glob
import os
import time

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import models
from cfgnode import CfgNode
from load_blender import load_blender_data
from nerf_helpers import (get_ray_bundle, img2mse, meshgrid_xy, mse2psnr,
                          positional_encoding,chunksize_to_2D)
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
    with open(config_file, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    print("Running experiment %s"%(cfg.experiment.id))
    SR_experiment = None
    if "super_resolution" in cfg:
        SR_experiment = "model" if "model" in cfg.super_resolution.keys() else "refine"
    if SR_experiment:
        LR_model_folder = cfg.models.path
        if os.path.isfile(LR_model_folder):   LR_model_folder = "/".join(LR_model_folder.split("/")[:-1])
        with open(os.path.join(LR_model_folder,"config.yml"), "r") as f:
            cfg.super_resolution.ds_factor = CfgNode(yaml.load(f, Loader=yaml.FullLoader)).dataset.downsampling_factor
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
        images, poses, render_poses, hwfDs, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            downsampling_factor=cfg.dataset.get("downsampling_factor",1),
            cfg=cfg
        )
        i_train, i_val, i_test = i_split
        SR_HR_im_inds,val_ims_dict = None,None
        if isinstance(i_train,tuple):
            SR_HR_im_inds = i_train[0]
            SR_LR_im_inds = i_train[1]
            val_ims_dict = i_val[1]
            i_val = i_val[0]

        H, W, focal,ds_factor = hwfDs
        # H, W = int(H), int(W)
        # hwf = [H, W, focal]

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
        # encode_position_fn = lambda: None
        # encode_direction_fn = lambda: None
        encode_position_fn = None
        encode_direction_fn = None
    
    if cfg.models.coarse.type=="TwoDimPlanesModel":
        model_coarse = models.TwoDimPlanesModel(
            use_viewdirs=cfg.models.coarse.use_viewdirs,
            plane_resolutions=getattr(cfg.models.coarse,'plane_resolutions',512),
            scene_geometry = {'camera_poses':poses.numpy()[:,:3,:4],'near':cfg.dataset.near,'far':cfg.dataset.far,'H':H,'W':W,'f':focal},
            dec_density_layers=getattr(cfg.models.coarse,'dec_density_layers',4),
            dec_rgb_layers=getattr(cfg.models.coarse,'dec_rgb_layers',4),
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
            use_viewdirs=cfg.models.coarse.use_viewdirs,
        )
    def num_parameters(model):
        return sum([p.numel() for p in model.parameters()])

    print("Coarse model: %d parameters"%num_parameters(model_coarse))
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        if cfg.models.fine.type=="use_same":
            model_fine = model_coarse
            print("Using the same model for coarse and fine")
        else:
            if cfg.models.fine.type=="TwoDimPlanesModel":
                model_fine = models.TwoDimPlanesModel(
                    use_viewdirs=cfg.models.fine.use_viewdirs,
                    plane_resolutions=getattr(cfg.models.fine,'plane_resolutions',512),
                    scene_geometry = {'camera_poses':poses.numpy()[:,:3,:4],'near':cfg.dataset.near,'far':cfg.dataset.far,'H':H,'W':W,'f':focal},
                    dec_density_layers=getattr(cfg.models.fine,'dec_density_layers',4),
                    dec_rgb_layers=getattr(cfg.models.fine,'dec_rgb_layers',4),
                )
            else:
                model_fine = getattr(models, cfg.models.fine.type)(
                    num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
                    num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
                    include_input_xyz=cfg.models.fine.include_input_xyz,
                    include_input_dir=cfg.models.fine.include_input_dir,
                    use_viewdirs=cfg.models.fine.use_viewdirs,
                )
            print("Fine model: %d parameters"%num_parameters(model_fine))
            model_fine.to(device)

    if SR_experiment=="model":
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
        # SR_HR_im_inds = cfg.super_resolution.get("dataset",{}).get("train_im_inds",None)
        # if SR_HR_im_inds is not None:
        #     if not isinstance(SR_HR_im_inds,list):
        #         pass
        #     SR_LR_im_inds = [i for i in i_train if i not in SR_HR_im_inds]
    else:
        # Initialize optimizer.
        # SR_HR_im_inds = None
        trainable_parameters = list(model_coarse.parameters())
        if model_fine is not None:
            trainable_parameters += list(model_fine.parameters())
        SR_model = None
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # Load an existing checkpoint, if a path is specified.
    start_i = 0
    if configargs.load_checkpoint=="resume":  configargs.load_checkpoint = logdir
    if SR_experiment or os.path.exists(configargs.load_checkpoint):
        if SR_experiment:#=="model" or not os.path.exists(configargs.load_checkpoint):
            checkpoint = find_latest_checkpoint(cfg.models.path)
            print("Using LR model %s"%(checkpoint))
            if SR_experiment=="model" and os.path.exists(configargs.load_checkpoint):
                assert os.path.isdir(configargs.load_checkpoint)
                SR_model_checkpoint = os.path.join(configargs.load_checkpoint,sorted([f for f in os.listdir(configargs.load_checkpoint) if "checkpoint" in f and f[-5:]==".ckpt"],key=lambda x:int(x[len("checkpoint"):-5]))[-1])
                start_i = int(SR_model_checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
                print("Resuming training on model %s"%(SR_model_checkpoint))
                with open(os.path.join(configargs.load_checkpoint,"config.yml"),"r") as f:
                    saved_config_dict = CfgNode(yaml.load(f, Loader=yaml.FullLoader))
                    config_diffs = DeepDiff(saved_config_dict,cfg)
                    for diff in [config_diffs[ch_type] for ch_type in ['dictionary_item_removed','dictionary_item_added','values_changed'] if ch_type in config_diffs]:
                        print(diff)

                SR_model.load_state_dict(torch.load(SR_model_checkpoint)["SR_model"])
        else:
            checkpoint = find_latest_checkpoint(configargs.load_checkpoint)
            start_i = int(checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
            print("Resuming training on model %s"%(checkpoint))
        checkpoint = torch.load(checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        if SR_experiment=="refine":
            LR_model_coarse = deepcopy(model_coarse)
            LR_model_fine = deepcopy(model_fine)
            furthest_rgb_fine,closest_rgb_fine = None,None
            if os.path.exists(configargs.load_checkpoint):
                checkpoint = find_latest_checkpoint(configargs.load_checkpoint)
                start_i = int(checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
                print("Resuming training on model %s"%(checkpoint))
                checkpoint = torch.load(checkpoint)
                model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
                model_fine.load_state_dict(checkpoint["model_fine_state_dict"])

        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # # TODO: Prepare raybatch tensor if batching random rays
    spatial_padding_size = SR_model.receptive_field//2 if isinstance(SR_model,models.Conv3D) else 0
    spatial_sampling = spatial_padding_size>0 or cfg.nerf.train.get("spatial_sampling",False)
    for iter in trange(start_i,cfg.experiment.train_iters):
        # Validation
        if (
            iter % cfg.experiment.validate_every == 0
            or iter == cfg.experiment.train_iters - 1
        ):
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
                    img_indecis,val_strings = [np.random.choice(i_val)],[""]
                    if val_ims_dict is not None:
                        img_indecis += [i_val[val_ims_dict["closest_val"]],i_val[val_ims_dict["furthest_val"]]]
                        val_strings += ["closest_","furthest_"]
                    # img_idx = 0
                    for val_num,img_idx in enumerate(img_indecis):
                        img_target = images[img_idx].to(device)
                        pose_target = poses[img_idx, :3, :4].to(device)
                        ray_origins, ray_directions = get_ray_bundle(
                            H[img_idx], W[img_idx], focal[img_idx], pose_target,padding_size=spatial_padding_size
                        )
                        rgb_coarse, _, _, rgb_fine, _, _,rgb_SR,_,_ = eval_nerf(
                            H[img_idx],
                            W[img_idx],
                            focal[img_idx],
                            model_coarse,
                            model_fine,
                            ray_origins,
                            ray_directions,
                            cfg,
                            mode="validation",
                            encode_position_fn=encode_position_fn,
                            encode_direction_fn=encode_direction_fn,
                            SR_model=SR_model,
                            ds_factor=ds_factor[img_idx],
                            spatial_margin=SR_model.receptive_field//2 if spatial_sampling else None
                        )
                        target_ray_values = img_target
                        if SR_experiment:
                            if SR_experiment=="refine":
                                rgb_SR = 1*rgb_fine
                                rgb_SR_coarse = 1*rgb_coarse
                                if val_num==0 or (val_num==2 and furthest_rgb_fine is None) or (val_num==1 and closest_rgb_fine is None):
                                    rgb_coarse, _, _, rgb_fine, _, _,_,_,_ = eval_nerf(
                                        H[img_idx],
                                        W[img_idx],
                                        focal[img_idx],
                                        LR_model_coarse,
                                        LR_model_fine,
                                        ray_origins,
                                        ray_directions,
                                        cfg,
                                        mode="validation",
                                        encode_position_fn=encode_position_fn,
                                        encode_direction_fn=encode_direction_fn,
                                        SR_model=SR_model,
                                        ds_factor=ds_factor[img_idx],
                                    )
                                    if val_num==1:  closest_rgb_fine = 1*rgb_fine.detach()
                                    elif val_num==2:  furthest_rgb_fine = 1*rgb_fine.detach()
                                else:
                                    rgb_fine = 1*closest_rgb_fine if val_num==1 else 1*furthest_rgb_fine
                            fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3]).item()
                            loss = img2mse(rgb_SR[..., :3], target_ray_values[..., :3])
                            writer.add_image(
                                "validation/%srgb_SR"%(val_strings[val_num]), cast_to_image(rgb_SR[..., :3]),iter
                            )
                            if val_num==0:
                                writer.add_scalar("validation/%sfine_loss"%(val_strings[val_num]), fine_loss, iter)
                                writer.add_scalar("validation/%sfine_psnr"%(val_strings[val_num]), mse2psnr(fine_loss), iter)
                        else:
                            coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                            fine_loss = 0.0
                            if rgb_fine is not None:
                                fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                                writer.add_scalar("validation/%sfine_loss"%(val_strings[val_num]), fine_loss.item(), iter)
                            loss = coarse_loss + fine_loss
                            writer.add_scalar("validation/%scoarse_loss"%(val_strings[val_num]), coarse_loss.item(), iter)
                            writer.add_image(
                                "validation/%srgb_coarse"%(val_strings[val_num]), cast_to_image(rgb_coarse[..., :3]),iter
                            )
                        psnr = mse2psnr(loss.item())
                        if SR_experiment:
                            writer.add_scalar("validation/%sSR_psnr_gain"%(val_strings[val_num]), psnr-mse2psnr(fine_loss), iter)
                        if val_num==0:
                            writer.add_scalar("validation/%sloss"%(val_strings[val_num]), loss.item(), iter)
                            writer.add_scalar("validation/%spsnr"%(val_strings[val_num]), psnr, iter)
                        if rgb_fine is not None:
                            if val_num==0 or iter==0:
                                writer.add_image(
                                    "validation/%srgb_fine"%(val_strings[val_num]), cast_to_image(rgb_fine[..., :3]),iter
                                )
                        if val_num==0 or iter==0:
                            writer.add_image(
                                "validation/%simg_target"%(val_strings[val_num]), cast_to_image(target_ray_values[..., :3]),iter
                            )
                        if val_num==0:
                            tqdm.write(
                                "Validation loss: "
                                + str(loss.item())
                                + " Validation PSNR: "
                                + str(psnr)
                                + "Time: "
                                + str(time.time() - start)
                            )
            
        # Training:
        if SR_experiment=="model":
            SR_model.train()
        else:
            model_coarse.train()
            if model_fine:
                model_fine.train()

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

            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(H[img_idx], W[img_idx], focal[img_idx], pose_target,padding_size=spatial_padding_size)
            coords = torch.stack(
                meshgrid_xy(torch.arange(H[img_idx]+2*spatial_padding_size).to(device), torch.arange(W[img_idx]+2*spatial_padding_size).to(device)),
                dim=-1,
            )
            if spatial_padding_size>0 or spatial_sampling:
                patch_size = chunksize_to_2D(cfg.nerf.train.num_random_rays)
                upper_left_corner = np.random.uniform(size=[2])*(np.array([H[img_idx],W[img_idx]])-patch_size)
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
                H[img_idx] if not spatial_sampling else patch_size,
                W[img_idx] if not spatial_sampling else patch_size,
                focal[img_idx],
                model_coarse,
                model_fine,
                batch_rays,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                SR_model=SR_model,
                ds_factor=ds_factor[img_idx],
                spatial_margin=SR_model.receptive_field//2 if spatial_sampling else None
            )
            target_ray_values = target_s

        if SR_experiment=="model":
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
            # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
            loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()
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
        if SR_experiment!="model":
            writer.add_scalar("train/coarse_loss", coarse_loss.item(), iter)
            if rgb_fine is not None:
                writer.add_scalar("train/fine_loss", fine_loss.item(), iter)
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
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(iter).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    main()
