import torch

from nerf_helpers import get_minibatches, ndc_rays
from nerf_helpers import sample_pdf_2 as sample_pdf
from volume_rendering_utils import volume_render_radiance_field

from functorch import vjp#make_functional_with_buffers, vmap, grad

def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn,return_input_grads=False):

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    # embeded_xyz = embed_fn(pts_flat)
    embedded = embed_fn(pts_flat)
    if return_input_grads:
        grads = []
        def cotangents(rows,ones_col):
            return torch.stack([torch.ones([rows]) if col==ones_col else torch.zeros([rows]) for col in range(4)],1).to(pts_flat.device)

    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

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
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    if return_input_grads:
        return radiance_field,torch.cat(grads,0).reshape(list(pts.shape[:-1])+list(grads[0].shape[1:]))
    else:
        return radiance_field


def identity_encoding(x):
    return x


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    SR_model=None,
):
    # TESTED

    if encode_position_fn is None:
        encode_position_fn = identity_encoding
    if encode_direction_fn is None:
        encode_direction_fn = identity_encoding

    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].reshape((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(0.0, 1.0, getattr(options.nerf, mode).num_coarse).to(ro)
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

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

    # num_coarse = getattr(options.nerf, mode).num_coarse
    # far_ = options.dataset.far
    # near_ = options.dataset.near
    # z_vals = torch.linspace(near_, far_, num_coarse).to(ro)
    # noise_shape = list(ro.shape[:-1]) + [num_coarse]
    # z_vals = z_vals + torch.rand(noise_shape).to(ro) * (far_ - near_) / num_coarse
    # pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
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
    )

    # TODO: Implement importance sampling, and finer network.
    rgb_fine, disp_fine, acc_fine,rgb_SR,disp_SR,acc_SR = None, None, None, None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
        num_grads_2_return = 0
        if SR_model is not None:
            if options.super_resolution.model.input=="xyz_encoding":    
                num_grads_2_return = 3*(2*options.models.fine.num_encoding_fn_xyz+1)
            elif options.super_resolution.model.input=="dirs_encoding":
                num_grads_2_return = 3*(2*(options.models.fine.num_encoding_fn_dir+options.models.fine.num_encoding_fn_xyz)+2)
        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            # getattr(options.nerf, mode).chunksize//(1 if SR_model is None else 100),
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
            return_input_grads=num_grads_2_return,
            # return_input_grads={"order_by_outputs":options.super_resolution.model.get("consistent_density",False),\
            #     "num_grads_2_return":num_grads_2_return},
        )
        if SR_model is not None:
            SR_inputs = []
            if num_grads_2_return>0:
                if options.super_resolution.model.get("consistent_density",False):
                    num_xyz_coords = 3*(2*options.models.fine.num_encoding_fn_xyz+1)
                    SR_inputs.append(torch.cat((
                        radiance_field[1][:,:,-1,:num_xyz_coords],
                        radiance_field[1][:,:,:-1,:].reshape([radiance_field[1].shape[0],radiance_field[1].shape[1],-1]),
                        radiance_field[1][:,:,-1,num_xyz_coords:],
                    ),-1))
                else:
                    SR_inputs.append(radiance_field[1].reshape([radiance_field[1].shape[0],radiance_field[1].shape[1],-1]))
                radiance_field = radiance_field[0]
            if options.super_resolution.model.get("consistent_density",False):
                SR_inputs.insert(0,radiance_field[...,-1:])
                SR_inputs.append(radiance_field[...,:-1])
            else:
                SR_inputs.insert(0,radiance_field)
            SR_inputs = torch.cat(SR_inputs,-1)

        rgb_fine, disp_fine, acc_fine, _, _ = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
        )
        if SR_model is not None:
            SR_input_shape = list(SR_inputs.shape)
            residual = SR_model(SR_inputs.reshape([SR_input_shape[0]*SR_input_shape[1],-1]))
            radiance_field = radiance_field + options.super_resolution.model.get("weight",1)*residual.reshape(SR_input_shape[:-1]+[4])
            if False:
                # Plotting STD of SR input channels across rays:
                from matplotlib import pyplot as plt
                plt.clf()
                legends = []
                STDs2plot = SR_inputs.std(1).mean(0).cpu().numpy()
                OMIT_FIRST = 0
                plt.plot(STDs2plot[:4]);  legends = ["RGBsigma"]
                for i in range(4):
                    plt.plot(STDs2plot[4+i*num_grads_2_return+OMIT_FIRST:4+(i+1)*num_grads_2_return])
                    legends.append(i)
                plt.legend(legends)
                plt.savefig("SR_inputs_STD.png")

            rgb_SR, disp_SR, acc_SR, _, _ = volume_render_radiance_field(
                radiance_field,
                z_vals,
                rd,
                radiance_field_noise_std=getattr(
                    options.nerf, mode
                ).radiance_field_noise_std,
                white_background=getattr(options.nerf, mode).white_background,
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
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    SR_model=None,
):
    if encode_position_fn is None:
        encode_position_fn = identity_encoding
    if encode_direction_fn is None:
        encode_direction_fn = identity_encoding

    ray_origins = batch_rays[0]
    ray_directions = batch_rays[1]
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.reshape((-1, 3))
    ray_shapes = ray_directions.shape  # Cache now, to restore later.
    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(H, W, focal, 1.0, ray_origins, ray_directions)
        ro = ro.reshape((-1, 3))
        rd = rd.reshape((-1, 3))
    else:
        ro = ray_origins.reshape((-1, 3))
        rd = ray_directions.reshape((-1, 3))
    # near = options.nerf.near * torch.ones_like(rd[..., :1])
    # far = options.nerf.far * torch.ones_like(rd[..., :1])
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize//(32 if SR_model is not None else 1))
    # TODO: Init a list, keep appending outputs to that list,
    # concat everything in the end.
    rgb_coarse, disp_coarse, acc_coarse = [], [], []
    rgb_fine, disp_fine, acc_fine,rgb_SR, disp_SR, acc_SR = None, None, None,None, None, None
    def append2list(item,to_list):
        if item is not None:
            if to_list is None:
                to_list = [item]
            else:
                to_list.append(item)
        return to_list

    for batch in batches:
        rc, dc, ac, rf, df, af,r_SR,d_SR,a_SR = predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            SR_model=SR_model,
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

    rgb_coarse_ = torch.cat(rgb_coarse, dim=0)
    disp_coarse_ = torch.cat(disp_coarse, dim=0)
    acc_coarse_ = torch.cat(acc_coarse, dim=0)

    def cat_list(list_2_cat):
        return None if list_2_cat is None else torch.cat(list_2_cat, dim=0)

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
    mode="validation",
    encode_position_fn=None,
    encode_direction_fn=None,
    SR_model=None,
):
    r"""Evaluate a NeRF by synthesizing a full image (as opposed to train mode, where
    only a handful of rays/pixels are synthesized).
    """
    if encode_position_fn is None:
        encode_position_fn = identity_encoding
    if encode_direction_fn is None:
        encode_direction_fn = identity_encoding
    original_shape = ray_origins.shape
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
        SR_model=SR_model,
    )
    rgb_coarse = rgb_coarse.reshape(original_shape)
    if rgb_fine is not None:
        rgb_fine = rgb_fine.reshape(original_shape)
    if rgb_SR is not None:
        rgb_SR = rgb_SR.reshape(original_shape)

    return rgb_coarse, None, None, rgb_fine, None, None, rgb_SR, None,None
