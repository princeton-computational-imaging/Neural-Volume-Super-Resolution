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
                          positional_encoding)
from train_utils import eval_nerf, run_one_iter_of_nerf


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    print("Running experiment %s"%(cfg.experiment.id))
    SR_experiment = "super_resolution" in cfg
    if SR_experiment:
        with open(os.path.join(cfg.models.path,"config.yml"), "r") as f:
            cfg.super_resolution.ds_factor = CfgNode(yaml.load(f, Loader=yaml.FullLoader)).dataset.downsampling_factor
        consistent_SR_density = cfg.super_resolution.model.get("consistent_density",False)
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
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            downsampling_factor=cfg.dataset.get("downsampling_factor",1),
            cfg=cfg
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
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

    # Initialize a coarse-resolution model.
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
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        print("Fine model: %d parameters"%num_parameters(model_fine))
        model_fine.to(device)

    if SR_experiment:
        NUM_FUNC_TYPES,NUM_COORDS,NUM_MODEL_OUTPUTS = 2,3,4
        assert cfg.super_resolution.model.input in ["xyz","dirs_encoding","xyz_encoding"]
        if cfg.super_resolution.model.input=="xyz":
            encoding_grad_inputs = 2*[0]
        elif cfg.super_resolution.model.input=="xyz_encoding":
            encoding_grad_inputs = [cfg.models.fine.num_encoding_fn_xyz,0]
        elif cfg.super_resolution.model.input=="dirs_encoding":
            encoding_grad_inputs = [cfg.models.fine.num_encoding_fn_xyz,cfg.models.fine.num_encoding_fn_dir]
        if consistent_SR_density:
            SR_input_dim = [NUM_FUNC_TYPES*d for d in encoding_grad_inputs]
            SR_input_dim = [(d+1)*NUM_COORDS if d>0 else 0 for d in SR_input_dim]
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
        SR_HR_im_inds = cfg.super_resolution.get("dataset",{}).get("train_im_inds",None)
        if SR_HR_im_inds is not None:
            SR_LR_im_inds = [i for i in i_train if i not in SR_HR_im_inds]
    else:
        # Initialize optimizer.
        SR_HR_im_inds = None
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

    def find_latest_checkpoint(ckpt_path):
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path,sorted([f for f in os.listdir(ckpt_path) if "checkpoint" in f and f[-5:]==".ckpt"],key=lambda x:int(x[len("checkpoint"):-5]))[-1])
        return ckpt_path

    # Load an existing checkpoint, if a path is specified.
    start_i = 0
    if SR_experiment or os.path.exists(configargs.load_checkpoint):
        if SR_experiment:
            checkpoint = find_latest_checkpoint(cfg.models.path)
            print("Using LR model %s"%(checkpoint))
            if os.path.exists(configargs.load_checkpoint):
                assert os.path.isdir(configargs.load_checkpoint)
                SR_model_checkpoint = os.path.join(configargs.load_checkpoint,sorted([f for f in os.listdir(configargs.load_checkpoint) if "checkpoint" in f and f[-5:]==".ckpt"],key=lambda x:int(x[len("checkpoint"):-5]))[-1])
                start_i = int(SR_model_checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
                print("Resuming training on model %s"%(SR_model_checkpoint))
                SR_model.load_state_dict(torch.load(SR_model_checkpoint)["SR_model"])
        else:
            checkpoint = find_latest_checkpoint(configargs.load_checkpoint)
            # if os.path.isdir(configargs.load_checkpoint):
            #     checkpoint = os.path.join(configargs.load_checkpoint,sorted([f for f in os.listdir(configargs.load_checkpoint) if "checkpoint" in f and f[-5:]==".ckpt"],key=lambda x:int(x[len("checkpoint"):-5]))[-1])
            start_i = int(checkpoint.split('/')[-1][len('checkpoint'):-len('.ckpt')])+1
            print("Resuming training on model %s"%(checkpoint))
        checkpoint = torch.load(checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # # TODO: Prepare raybatch tensor if batching random rays

    for i in trange(start_i,cfg.experiment.train_iters):
        if SR_experiment:
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
            ray_origins, ray_directions = get_ray_bundle(H[img_idx], W[img_idx], focal[img_idx], pose_target)
            coords = torch.stack(
                meshgrid_xy(torch.arange(H[img_idx]).to(device), torch.arange(W[img_idx]).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            )
            select_inds = coords[select_inds]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

            rgb_coarse, _, _, rgb_fine, _, _,rgb_SR,_,_ = run_one_iter_of_nerf(
                H[img_idx],
                W[img_idx],
                focal[img_idx],
                model_coarse,
                model_fine,
                batch_rays,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                SR_model=SR_model,
            )
            target_ray_values = target_s

        if SR_experiment:
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
        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr)
            )
        writer.add_scalar("train/loss", loss.item(), i)
        if not SR_experiment:
            writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
            if rgb_fine is not None:
                writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_fine.eval()

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
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(
                        H[img_idx], W[img_idx], focal[img_idx], pose_target
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
                    )
                    target_ray_values = img_target
                if SR_experiment:
                    loss = img2mse(rgb_SR[..., :3], target_ray_values[..., :3])
                    writer.add_image(
                        "validation/rgb_SR", cast_to_image(rgb_SR[..., :3])
                    )
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3]).item()
                    writer.add_scalar("validation/fine_loss", fine_loss, i)
                    writer.add_scalar("validataion/fine_psnr", mse2psnr(fine_loss), i)
                else:
                    coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                    fine_loss = 0.0
                    if rgb_fine is not None:
                        fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                        writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                    loss = coarse_loss + fine_loss
                    writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                    writer.add_image(
                        "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3])
                    )
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validataion/psnr", psnr, i)
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3])
                    )
                writer.add_image(
                    "validation/img_target", cast_to_image(target_ray_values[..., :3])
                )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + "Time: "
                    + str(time.time() - start)
                )

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                # "model_coarse_state_dict": model_coarse.state_dict(),
                # "model_fine_state_dict": None
                # if not model_fine
                # else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            if SR_experiment:
                checkpoint_dict.update({"SR_model":SR_model.state_dict()})
            else:
                checkpoint_dict.update({"model_coarse_state_dict": model_coarse.state_dict()})
                if model_fine:  checkpoint_dict.update({"model_fine_state_dict": model_fine.state_dict()})
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
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
