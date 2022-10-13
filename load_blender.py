import json
import os

# import cv2
import imageio
import numpy as np
import torch
from nerf_helpers import im_resize

def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, debug=False,
        downsampling_factor=1,val_downsampling_factor=None,cfg=None,splits2use=['train','val']):
    assert cfg is not None,"As of now, expecting to get the entire configuration"
    train_im_inds = None
    if cfg.get("super_resolution",None) is not None:
        train_im_inds = cfg.super_resolution.get("dataset",{}).get("train_im_inds",None)
    assert downsampling_factor==1 or train_im_inds is None,"Should not use a global downsampling_factor when training an SR model, only when learning the LR representation"
    # if downsampling_factor!=1: assert half_res,"Assuming half_res is True"
    if val_downsampling_factor is None:
        val_downsampling_factor = downsampling_factor
    splits = ["train", "val", "test"]
    assert all([s in splits for s in splits2use])
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    H,W,focal,ds_factor = [],[],[],[]
    ds_f_nums = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        # if val_only and s=='train':
        if s not in splits2use:
            meta["frames"] = []
        imgs = []
        poses = []
        if s=='val':
            skip = testskip
        else:
        # if s == "train" or testskip == 0:
            skip = 1
        # else:
        #     skip = testskip

        camera_angle_x = float(meta["camera_angle_x"])
        focal_over_W = 0.5 / np.tan(0.5 * camera_angle_x)
        total_split_frames = len(meta["frames"])
        for f_num,frame in enumerate(meta["frames"][::skip]):
            # if f_num>=2:
            #     print("!!!!!WARNING!!!!!!!")
            #     break
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            img = (imageio.imread(fname)/ 255.0).astype(np.float32)
            if s=='val':
                per_im_ds_factor = 1*val_downsampling_factor
            else:
                per_im_ds_factor = 1*downsampling_factor
            if s=="train" and train_im_inds is not None and (f_num not in train_im_inds if isinstance(train_im_inds,list) else f_num>train_im_inds*total_split_frames):
                per_im_ds_factor = cfg.super_resolution.ds_factor
                ds_f_nums.append(f_num)
            # for factor in per_im_ds_factor:
            H.append(img.shape[0])
            W.append(img.shape[1])
            if half_res:
                per_im_ds_factor *= 2
            H[-1] //= (per_im_ds_factor)
            W[-1] //= (per_im_ds_factor)
            # resized_img = torch.from_numpy(cv2.resize(img, dsize=(img.shape[1]//per_im_ds_factor, img.shape[0]//per_im_ds_factor), interpolation=cv2.INTER_AREA))
            resized_img = torch.from_numpy(im_resize(img, scale_factor=per_im_ds_factor))

            focal.append(focal_over_W*W[-1])
            ds_factor.append(per_im_ds_factor)
            imgs.append(resized_img)
            poses.append(np.array(frame["transform_matrix"]))
        # imgs = (np.array(imgs) / 255.0).astype(np.float32)
        poses = np.array(poses).reshape([-1,4,4]).astype(np.float32)
        counts.append(counts[-1] + len(imgs))
        all_imgs.append(imgs)
        all_poses.append(poses)

    imgs = [im for s in all_imgs for im in s]
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    if len(ds_f_nums)>0:
        i_split[0] = ([i for i in i_split[0] if i not in ds_f_nums],ds_f_nums)
        pose_dists = np.sum((np.stack([all_poses[0][i] for i in i_split[0][0]],0)[:,None,...]-all_poses[1][None,...])**2,axis=(2,3))
        closest_val = np.argmin(pose_dists)%pose_dists.shape[1]
        furthest_val = np.argmax(np.min(pose_dists,0))
        i_split[1] = (i_split[1],{"closest_val":closest_val,"furthest_val":furthest_val})
    # imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    # H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta["camera_angle_x"])
    # focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    # In debug mode, return extremely tiny images
    assert not debug,"No longer supported, after introducing downsampling options"
    if debug:
        H = H // 32
        W = W // 32
        focal = focal / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        poses = torch.from_numpy(poses)
        return imgs, poses, render_poses, [H, W, focal], i_split

    poses = torch.from_numpy(poses)

    return imgs, poses, render_poses, [H, W, focal,ds_factor], i_split
