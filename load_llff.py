import os
from nerf_helpers import im_resize,calc_resize_crop_margins
import imageio
import numpy as np
import torch
from scipy.interpolate import interp1d
# Implementation from:
# https://github.com/yenchenlin/nerf-pytorch/blob/master/load_llff.py
# Slightly modified version of LLFF data loading code
#  see https://github.com/Fyusion/LLFF for original


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, "images_{}".format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, "images_{}x{}".format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, "images")
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [
        f
        for f in imgs
        if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
    ]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = "images_{}".format(r)
            resizearg = "{}%".format(100.0 / r)
        else:
            name = "images_{}x{}".format(r[1], r[0])
            resizearg = "{}x{}".format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print("Minifying", r, basedir)

        os.makedirs(imgdir)
        check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split(".")[-1]
        args = " ".join(
            ["mogrify", "-resize", resizearg, "-format", "png", "*.{}".format(ext)]
        )
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != "png":
            check_output("rm {}/*.{}".format(imgdir, ext), shell=True)
            print("Removed duplicates")
        print("Done")


def _load_data(basedir, factor=None,base_factor=1,max_factor=1, width=None, height=None, load_imgs=True,min_eval_frames=None,):

    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    if min_eval_frames is not None:
        min_eval_frames = int(np.ceil(min_eval_frames/(len(poses_arr)-1))*(len(poses_arr)-1)+1)
        repeat_ims = (min_eval_frames-1)//(len(poses_arr)-1)
        temp = 1*poses_arr
        poses_arr = interp1d(np.arange(len(poses_arr)),poses_arr,axis=0)(np.linspace(start=0,stop=len(poses_arr)-1,num=min_eval_frames))
        poses_arr[::repeat_ims,:] = temp
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    while not os.path.isdir(os.path.join(basedir,"images%s"%('_%d'%(base_factor) if base_factor>1 else ''))):
        assert base_factor>=1
        base_factor //= 2
    images_subdir = "images%s"%('_%d'%(base_factor) if base_factor>1 else '')
    assert factor//base_factor==factor/base_factor
    img0 = [
        os.path.join(basedir, images_subdir, f)
        for f in sorted(os.listdir(os.path.join(basedir, images_subdir)))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    sh = imageio.imread(img0).shape

    sfx = ""
    imgdir = os.path.join(basedir, images_subdir + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, "does not exist, returning")
        return

    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    if min_eval_frames is not None:
        imgfiles = [f_ for f in imgfiles for f_ in ([f]+(repeat_ims-1)*[None])][:-repeat_ims+1]
    # else:
    #     repeat_ims = None
    if poses.shape[-1] != len(imgfiles):
        print(
            "Mismatch between imgs {} and poses {} !!!!".format(
                len(imgfiles), poses.shape[-1]
            )
        )
        return

    sh = np.array(imageio.imread(imgfiles[0]).shape)
    marg2crop = calc_resize_crop_margins(sh,max_factor//base_factor)
    if marg2crop is not None:
        sh[:2] -= 2*marg2crop
    sh = (sh[0]//(factor//base_factor),sh[1]//(factor//base_factor),sh[2])
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

    if load_imgs:
        def imread(f):
            if f.endswith("png"):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)

        imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
        if marg2crop is not None:
            imgs = [im[marg2crop[0]:-marg2crop[0] if marg2crop[0]>0 else None,marg2crop[1]:-marg2crop[1] if marg2crop[1]>0 else None,:] for im in imgs]
        if factor!=base_factor:
            imgs = [im_resize(im, scale_factor=factor//base_factor) for im in imgs]
        imgs = np.stack(imgs, -1)
    else:
        imgs = imgfiles

    return poses, bds, imgs,(base_factor,marg2crop)#,repeat_ims


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):

    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):
    def add_row_to_homogenize_transform(p):
        r"""Add the last row to homogenize 3 x 4 transformation matrices."""
        return np.concatenate(
            [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
        )

    # p34_to_44 = lambda p: np.concatenate(
    #     [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    # )

    p34_to_44 = add_row_to_homogenize_transform

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def load_llff_data(
    basedir, factor=8,base_factor=1,max_factor=1, recenter=True, bd_factor=0.75, spherify=False, path_zflat=False,load_imgs=True,min_eval_frames=None,
):

    poses, bds, imgs,load_params = _load_data(
        basedir, factor=factor,base_factor=base_factor,max_factor=max_factor,load_imgs=load_imgs,min_eval_frames=min_eval_frames,
    )  # factor=8 downsamples original imgs by 8x
    # print("Loaded", basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    if load_imgs:
        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        c2w = poses_avg(poses)
        # print("recentered", c2w.shape)
        # print(c2w[:3, :4])

        # Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = 0.8
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.0
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
        )

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    # print("Data:")
    # print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    # print("HOLDOUT view is", i_test)

    # images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    # return torch.from_numpy(images.astype(np.float32)) if load_imgs else images, torch.from_numpy(poses), bds, render_poses, i_test,None if load_imgs else load_params
    return torch.from_numpy(images.astype(np.float32)) if load_imgs else images, torch.from_numpy(poses), bds, render_poses, i_test,load_params