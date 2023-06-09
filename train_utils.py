import torch
import socket
from nerf_helpers import get_minibatches, ndc_rays,spatial_batch_merge
from nerf_helpers import sample_pdf_2 as sample_pdf
from volume_rendering_utils import volume_render_radiance_field

# from functorch import vjp#make_functional_with_buffers, vmap, grad
import mip
import numpy as np
from torch.nn.functional import pad
import os
import models
from re import search

def run_network(network_fn, pts, ray_batch, chunksize, embed_fn,\
     embeddirs_fn,scene_id,return_input_grads=False,mip_nerf=False,z_vals=None):

    pts_shape = list(pts.shape)
    if mip_nerf:
        ro, rd, near, far, viewdir = torch.split(ray_batch,[3,3,1,1,3],dim=-1)
        dx = int(search('(?<=_DS)(\d)+(?=$)',scene_id).group(0))*0.00135
        # Get t_vals in world to aply mipnerf
        radii = dx * 2 / np.sqrt(12.)
        means,covs = mip.cast_rays(z_vals,ro,rd,radii,None)
        pts_flat = embed_fn((means,covs))
        pts_shape[1] = pts_flat.shape[1]
        embedded = pts_flat.reshape((-1, pts_flat.shape[-1]))
    else:
        pts_flat = pts.reshape((-1, pts_shape[-1]))
        embedded = embed_fn(pts_flat)
    if return_input_grads:
        grads = []
        def cotangents(rows,ones_col):
            return torch.stack([torch.ones([rows]) if col==ones_col else torch.zeros([rows]) for col in range(4)],1).to(pts_flat.device)

    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts_shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)


    # chunksize = embedded.shape[0]
    if chunksize is None:
        batches = [embedded]
    else:
        batches = get_minibatches(embedded, chunksize=chunksize)
    preds = []
    for batch in batches:
        if return_input_grads:
            cur_pred,vjp_fn = vjp(network_fn,batch)
            preds.append(cur_pred)
            grads.append(torch.stack([vjp_fn(cotangents(cur_pred.shape[0],col))[0] for col in range(4)],1)[...,:return_input_grads])
        else:
            preds.append(network_fn(batch))
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts_shape[:-1]) + [radiance_field.shape[-1]]
    )
    if return_input_grads:
        return radiance_field,torch.cat(grads,0).reshape(list(pts_shape[:-1])+list(grads[0].shape[1:]))
    else:
        return radiance_field


def identity_encoding(x):
    return x


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    scene_id,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
):
    # TESTED
    mip_nerf = getattr(options.nerf,'encode_position_fn',None)=="mip"
    if encode_position_fn is None:
        encode_position_fn = identity_encoding
    if encode_direction_fn is None:
        encode_direction_fn = identity_encoding

    with model_coarse.optional_no_grad():
        ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
        bounds = ray_batch[..., 6:8].reshape(list(ray_batch.shape)[:-1]+[1, 2])
        near, far = bounds[..., 0], bounds[..., 1]

        # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
        # when not enabling "ndc".
        t_vals = torch.linspace(0.0, 1.0, getattr(options.nerf, mode).num_coarse+mip_nerf).to(ro)
        if not getattr(options.nerf, mode).lindisp:
            z_vals = near * (1.0 - t_vals) + far * t_vals
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
        z_vals = z_vals.expand(list(ray_batch.shape)[:-1]+[getattr(options.nerf, mode).num_coarse+mip_nerf])

        if getattr(options.nerf, mode).perturb:
            # Get intervals between samples.
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
            lower = torch.cat((z_vals[..., :1], mids), dim=-1)
            # Stratified samples in those intervals.
            t_rand = torch.rand(z_vals.shape).to(ro)
            z_vals = lower + (upper - lower) * t_rand
        # pts -> (num_rays, N_samples, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
        # SR_CHUNK_REDUCE = 2
        chunksize = getattr(options.nerf, mode).chunksize
        radiance_field = run_network(
            model_coarse,
            pts,
            ray_batch,
            chunksize,
            encode_position_fn,
            encode_direction_fn,
            mip_nerf=mip_nerf,
            z_vals=z_vals,
            scene_id=scene_id,
        )

        (
            rgb_coarse,
            disp_coarse,
            acc_coarse,
            weights,
            depth_coarse,
        ) = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            mip_nerf=mip_nerf,
        )

    # TODO: Implement importance sampling, and finer network.
    rgb_fine, disp_fine, acc_fine,rgb_SR,disp_SR,acc_SR = None, None, None, None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        z_average = lambda x:   0.5 * (x[..., 1:] + x[..., :-1])
        z_vals_mid = z_average(z_vals)
        if mip_nerf:    z_vals_mid = z_average(z_vals_mid)
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine+mip_nerf,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
        num_grads_2_return = 0
        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            None if (hasattr(model_fine,'SR_model') and model_fine.SR_model.training) else getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
            return_input_grads=num_grads_2_return,
            mip_nerf=mip_nerf,
            z_vals=z_vals,
            scene_id=scene_id,
        )

        rgb_fine, disp_fine, acc_fine, _, _ = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            mip_nerf=mip_nerf,
        )

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine,rgb_SR, disp_SR, acc_SR


def run_one_iter_of_nerf(
    H,
    W,
    focal,
    model_coarse,
    model_fine,
    batch_rays,
    options,
    scene_id,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    scene_config={},
):
    if encode_position_fn is None:
        encode_position_fn = identity_encoding
    if encode_direction_fn is None:
        encode_direction_fn = identity_encoding

    if isinstance(model_coarse,models.TwoDimPlanesModel):
        model_coarse.set_cur_scene_id(scene_id)
        model_fine.set_cur_scene_id(scene_id)
    ray_origins = batch_rays[0]
    ray_directions = batch_rays[1]
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.reshape((-1, 3))
    if scene_config.no_ndc is False:
        ro, rd = ndc_rays(H, W, focal, 1.0, ray_origins, ray_directions)
        ro = ro.reshape((-1, 3))
        rd = rd.reshape((-1, 3))
    else:
        ro = ray_origins.reshape((-1, 3))
        rd = ray_directions.reshape((-1, 3))
    near = scene_config.near * torch.ones_like(rd[..., :1])
    far = scene_config.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    chunk_size = getattr(options.nerf, mode).chunksize
    if isinstance(model_coarse,models.TwoDimPlanesModel):
        chunk_size = int(chunk_size/(model_coarse.num_density_planes/3))
    if hasattr(model_fine,'SR_model'):
        chunk_size //= 10
    elif hasattr(options.nerf,'encode_position_fn') and options.nerf.encode_position_fn=="mip":
        chunk_size //= 4 # For some reason I get memory problems when using MipNeRF, so I'm using this arbitrary factor of 4.
    batches = get_minibatches(rays, chunksize=chunk_size)
    batch_shapes = [tuple(b.shape[:-1]) for b in batches]
    rgb_coarse, disp_coarse, acc_coarse = [], [], []
    rgb_fine, disp_fine, acc_fine,rgb_SR, disp_SR, acc_SR = None, None, None,None, None, None
    def append2list(item,to_list):
        if item is not None:
            if to_list is None:
                to_list = [item]
            else:
                to_list.append(item)
        return to_list

    for b_num,batch in enumerate(batches):
        rc, dc, ac, rf, df, af,r_SR,d_SR,a_SR = predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            mode=mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            scene_id=scene_id,
        )
        rgb_coarse.append(rc)
        disp_coarse.append(dc)
        acc_coarse.append(ac)
        rgb_fine = append2list(rf,rgb_fine)
        disp_fine = append2list(df,disp_fine)
        acc_fine = append2list(af,acc_fine)
        rgb_SR = append2list(r_SR,rgb_SR)
        disp_SR = append2list(d_SR,disp_SR)
        acc_SR = append2list(a_SR,acc_SR)

    def cat_list(list_2_cat):
        if list_2_cat is None:  return None
        return torch.cat(list_2_cat, dim=0)

    rgb_coarse_ = cat_list(rgb_coarse)
    disp_coarse_ = cat_list(disp_coarse)
    acc_coarse_ = cat_list(acc_coarse)
    rgb_fine_ = cat_list(rgb_fine)
    disp_fine_ = cat_list(disp_fine)
    acc_fine_ = cat_list(acc_fine)
    rgb_SR_ = cat_list(rgb_SR)
    disp_SR_ = cat_list(disp_SR)
    acc_SR_ = cat_list(acc_SR)

    return rgb_coarse_, disp_coarse_, acc_coarse_, rgb_fine_, disp_fine_, acc_fine_, rgb_SR_, disp_SR_, acc_SR_


def eval_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    options,
    scene_id,
    mode="validation",
    encode_position_fn=None,
    encode_direction_fn=None,
    scene_config={},
):
    r"""Evaluate a NeRF by synthesizing a full image (as opposed to train mode, where
    only a handful of rays/pixels are synthesized).
    """
    if encode_position_fn is None:
        encode_position_fn = identity_encoding
    if encode_direction_fn is None:
        encode_direction_fn = identity_encoding
    # original_shape = ray_origins.shape
    ray_origins = ray_origins.reshape((1, -1, 3))
    ray_directions = ray_directions.reshape((1, -1, 3))
    batch_rays = torch.cat((ray_origins, ray_directions), dim=0)
    rgb_coarse, _, _, rgb_fine, _, _,rgb_SR,_,_ = run_one_iter_of_nerf(
        height,
        width,
        focal_length,
        model_coarse,
        model_fine,
        batch_rays,
        options,
        mode="validation",
        encode_position_fn=encode_position_fn,
        encode_direction_fn=encode_direction_fn,
        scene_id=scene_id,
        scene_config=scene_config,
    )
    rgb_coarse = rgb_coarse.reshape([height,width,-1])
    if rgb_fine is not None:
        rgb_fine = rgb_fine.reshape([height,width,-1])
    if rgb_SR is not None:
        rgb_SR = rgb_SR.reshape([height,width,-1])

    return rgb_coarse, None, None, rgb_fine, None, None, rgb_SR, None,None

def find_latest_checkpoint(ckpt_path,sr,find_best=False):
    if os.path.isdir(ckpt_path):
        if find_best:
            # pattern = ("^SR_checkpoint" if sr else "^checkpoint")+"\.ckpt_best"
            pattern = ("^SR_checkpoint" if sr else "^checkpoint")+"(\d)*\.ckpt_best"
            ckpt_path = os.path.join(ckpt_path,[f for f in os.listdir(ckpt_path) if search(pattern,f) is not None][0])
        else:
            pattern = "(?<="+("^SR_checkpoint" if sr else "^checkpoint")+")(\d)+(?=\.ckpt$)"
            ckpt_path = os.path.join(ckpt_path,sorted([f for f in os.listdir(ckpt_path) if search(pattern,f) is not None],
                key=lambda x:int(search(pattern,x).group(0)))[-1])
        return ckpt_path
    else:
        return None
