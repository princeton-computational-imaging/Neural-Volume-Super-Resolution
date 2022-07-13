import json
import os

import cv2
import imageio
import numpy as np
import torch


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


def load_blender_data(basedir, half_res=False, testskip=1, debug=False,downsampling_factor=1,cfg=None):
    assert cfg is not None,"As of now, expecting to get the entire configuration"
    train_im_inds = None
    if cfg.get("super_resolution",None) is not None:
        train_im_inds = cfg.super_resolution.get("dataset",{}).get("train_im_inds",None)
    assert downsampling_factor==1 or train_im_inds is None,"Should not use a global downsampling_factor when training an SR model, only when learning the LR representation"
    if downsampling_factor!=1: assert half_res,"Assuming half_res is True"
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    H,W,focal = [],[],[]
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        camera_angle_x = float(meta["camera_angle_x"])
        focal_over_W = 0.5 / np.tan(0.5 * camera_angle_x)
        for f_num,frame in enumerate(meta["frames"][::skip]):
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            img = (imageio.imread(fname)/ 255.0).astype(np.float32)
            H.append(img.shape[0])
            W.append(img.shape[1])
            per_im_ds_factor = 1*downsampling_factor
            if s=="train" and train_im_inds is not None and f_num not in train_im_inds:
                per_im_ds_factor = cfg.super_resolution.ds_factor
            if half_res:
                H[-1] //= (2*per_im_ds_factor)
                W[-1] //= (2*per_im_ds_factor)
                img = torch.from_numpy(cv2.resize(img, dsize=(400//per_im_ds_factor, 400//per_im_ds_factor), interpolation=cv2.INTER_AREA))

            focal.append(focal_over_W*W[-1])
            imgs.append(img)
            poses.append(np.array(frame["transform_matrix"]))
        # imgs = (np.array(imgs) / 255.0).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + len(imgs))
        all_imgs.append(imgs)
        all_poses.append(poses)

    imgs = [im for s in all_imgs for im in s]
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

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
    if half_res and False:
        # TODO: resize images using INTER_AREA (cv2)
        H = H // (2*downsampling_factor)
        W = W // (2*downsampling_factor)
        focal = focal / (2.0*downsampling_factor)
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(400//downsampling_factor, 400//downsampling_factor), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)

    poses = torch.from_numpy(poses)

    return imgs, poses, render_poses, [H, W, focal], i_split
