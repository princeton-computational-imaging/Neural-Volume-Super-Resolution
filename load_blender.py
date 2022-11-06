import json
import os
from tqdm import tqdm
from re import search
# import cv2
import imageio
import numpy as np
import torch
from nerf_helpers import im_resize,calc_scene_box,interpret_scene_list
# from models import get_scene_id
from collections import OrderedDict
from magic import from_file

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

class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self,config,scene_id_func,add_val_scene_LR,eval_mode,scene_norm_coords=None) -> None:
        super(BlenderDataset,self).__init__()
        self.get_scene_id = scene_id_func
        self.downsampling_factors,self.all_scenes,plane_resolutions,val_ids = self.get_scene_configs(getattr(config.dir,'val',{}))
        # if eval_mode:
        #     train_ids = []
        # else:
        train_dirs = self.get_scene_configs(config.dir.train,add_val_scene_LR=add_val_scene_LR,excluded_scene_ids=val_ids if getattr(config,'auto_remove_val',False) else [])
        self.downsampling_factors += train_dirs[0]
        plane_resolutions += train_dirs[2]
        train_ids = train_dirs[3]
        if len(set(train_ids+val_ids))!=len(train_ids+val_ids):
            raise Exception('I suspect an overlap between training and validation scenes. The following appear in both:\n%s'%([s for s in val_ids if s in train_ids]))
        train_dirs = train_dirs[1]
        self.all_scenes.extend(train_dirs)
        self.images, self.poses, render_poses, self.hwfDs, i_split,self.per_im_scene_id = [],torch.zeros([0,4,4]),[],[],[np.array([]).astype(np.int64) for i in range(3)],[]
        scene_id,self.val_only_scene_ids,self.coords_normalization = -1,[],{}
        self.scene_id_plane_resolution = {}
        self.i_train,self.i_val = OrderedDict(),OrderedDict()
        self.scenes_set = set()
        for basedir,ds_factor,plane_res in zip(tqdm(self.all_scenes,desc='Loading scenes'),self.downsampling_factors,plane_resolutions):
            scene_id = self.get_scene_id(basedir,ds_factor,plane_res)
            if scene_id in self.i_train:
                raise Exception("Scene %s already in the set"%(scene_id))
            self.scenes_set.add(scene_id)
            val_only = scene_id not in train_ids
            if val_only:    self.val_only_scene_ids.append(scene_id)
            self.scene_id_plane_resolution[scene_id] = plane_res
            if eval_mode:
                if not val_only:    continue
                splits2use = ['test']
            else:
                splits2use = ['val'] if val_only else ['train','val']
            scene_path = os.path.join(config.root,basedir)
            if search('#(\d)+',basedir) is not None:
                scene_path = scene_path.replace(search('#(\d)+',basedir).group(0),'')
            cur_images, cur_poses, cur_render_poses, cur_hwfDs, cur_i_split = load_blender_data(
                scene_path,
                testskip=config.testskip,
                downsampling_factor=ds_factor,
                val_downsampling_factor=None,
                splits2use=splits2use,
                class_config=True,
            )

            if scene_norm_coords is not None: # No need to calculate the per-scene normalization coefficients as those will be loaded with the saved model.
                self.coords_normalization[scene_id] =\
                    calc_scene_box({'camera_poses':cur_poses.numpy()[:,:3,:4],'near':config.near,'far':config.far,'H':cur_hwfDs[0],'W':cur_hwfDs[1],'f':cur_hwfDs[2]},
                        including_dirs=scene_norm_coords.use_viewdirs,adjust_elevation_range=getattr(scene_norm_coords,'adjust_elevation_range',False))
            if eval_mode:
                self.i_val[scene_id] = [v+len(self.images) for v in cur_i_split[2]]
            else:
                self.i_val[scene_id] = [v+len(self.images) for v in cur_i_split[1]]
            if not val_only:
                self.i_train[scene_id] = [v+len(self.images) for v in cur_i_split[0]]
            self.images += cur_images
            self.poses = torch.cat((self.poses,cur_poses),0)
            # for i in range(len(self.hwfDs)):
            #     self.hwfDs[i] += cur_hwfDs[i]
            # for i in range(len(cur_hwfDs)):
            self.hwfDs += [(cur_hwfDs[0][i],cur_hwfDs[1][i],cur_hwfDs[2][i],cur_hwfDs[3][i]) for i in range(len(cur_hwfDs[0]))]
            self.per_im_scene_id += [scene_id for i in cur_images]
    
    def item(self,index,device):
        img_target = (imageio.imread(self.images[index])/ 255.0).astype(np.float32)
        cur_H,cur_W,cur_focal,cur_ds_factor = self.hwfDs[index]
        if cur_ds_factor>1:
            img_target = im_resize(img_target, scale_factor=cur_ds_factor)
        pose_target = self.poses[index].to(device)
        return torch.from_numpy(img_target).to(device),pose_target,cur_H,cur_W,cur_focal,cur_ds_factor

    def __len__(self):
        return len(self.images)

    def get_scene_configs(self,config_dict,add_val_scene_LR=False,excluded_scene_ids=[]):
        ds_factors,dir,plane_res,scene_ids = [],[],[],[]
        config_dict = dict(config_dict)
        if add_val_scene_LR:
            assert len(config_dict)==2
            assert len(excluded_scene_ids)==0,'Unsupported'
            conf_HR_planes,conf_LR_planes = config_dict.keys()
            if len(config_dict[conf_HR_planes])>len(config_dict[conf_LR_planes]):
                conf_HR_planes,conf_LR_planes = conf_LR_planes,conf_HR_planes
            assert conf_HR_planes.split(',')[2]==conf_LR_planes.split(',')[2]
            config_dict.update({','.join([conf_LR_planes.split(',')[0],conf_HR_planes.split(',')[1],conf_LR_planes.split(',')[2]]):
                [sc for sc in config_dict[conf_LR_planes] if sc not in config_dict[conf_HR_planes]]})
        for conf,scenes in config_dict.items():
            conf = eval(conf)
            if not isinstance(scenes,list): scenes = [scenes]
            # for sc in scenes:
            for s in interpret_scene_list(scenes):
                # if isinstance(sc,list):
                #     assert len(sc)==2
                #     scs = [str(i) for i in range(sc[0],sc[1])]
                # else:
                #     scs = [sc]
                # for s in scs:
                cur_factor,cur_dir,cur_res = conf[0],s,(conf[1],conf[2] if len(conf)>2 else conf[1])
                cur_id = self.get_scene_id(cur_dir,cur_factor,cur_res)
                if cur_id in excluded_scene_ids:
                    continue
                scene_ids.append(cur_id)
                ds_factors.append(cur_factor)
                plane_res.append(cur_res)
                dir.append(cur_dir)
                    # scene_ids.append(self.get_scene_id(dir[-1],ds_factors[-1],plane_res[-1]))
        return ds_factors,dir,plane_res,scene_ids

def load_blender_data(basedir, half_res=False, testskip=1, debug=False,
        downsampling_factor=1,val_downsampling_factor=None,cfg=None,splits2use=['train','val'],class_config=False):
    # train_im_inds = None
    assert not half_res,'Depricated'
    if cfg is not None:
        raise Exception("Depricated")
        if cfg.get("super_resolution",None) is not None:
            train_im_inds = cfg.super_resolution.get("dataset",{}).get("train_im_inds",None)
    # assert downsampling_factor==1 or train_im_inds is None,"Should not use a global downsampling_factor when training an SR model, only when learning the LR representation"
    # if downsampling_factor!=1: assert half_res,"Assuming half_res is True"
    if val_downsampling_factor is None:
        val_downsampling_factor = downsampling_factor
    splits = ["train", "val", "test"]
    assert all([s in splits for s in splits2use])
    metas = {}
    for s in splits:
        if s not in splits2use: continue
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    H,W,focal,ds_factor = [],[],[],[]
    ds_f_nums = []
    counts = [0]
    for s in splits:
        if s in splits2use:
            meta = metas[s]
            camera_angle_x = float(meta["camera_angle_x"])
            focal_over_W = 0.5 / np.tan(0.5 * camera_angle_x)
            total_split_frames = len(meta["frames"])
        else:
            meta = {"frames":[]}
        imgs = []
        poses = []
        if s=='val':
            skip = testskip
        else:
            skip = 1

        for f_num,frame in enumerate(meta["frames"][::skip]):
            # if f_num>=2:
            #     print("!!!!!WARNING!!!!!!!")
            #     break
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            if s=='val':
                per_im_ds_factor = 1*val_downsampling_factor
            else:
                per_im_ds_factor = 1*downsampling_factor
            if class_config:
                im_dims = [int(v) for v in search('(\d+) x (\d+)', from_file(fname)).groups()]
                assert len(im_dims)==2 and im_dims[0]==im_dims[1],"Should verify the order of H,W"
                H.append(im_dims[0])
                W.append(im_dims[1])
            else:
                img = (imageio.imread(fname)/ 255.0).astype(np.float32)
                # if s=="train" and train_im_inds is not None and (f_num not in train_im_inds if isinstance(train_im_inds,list) else f_num>train_im_inds*total_split_frames):
                #     per_im_ds_factor = cfg.super_resolution.ds_factor
                #     ds_f_nums.append(f_num)
                H.append(img.shape[0])
                W.append(img.shape[1])
                # if half_res:
                #     per_im_ds_factor *= 2
                resized_img = torch.from_numpy(im_resize(img, scale_factor=per_im_ds_factor))
            H[-1] //= (per_im_ds_factor)
            W[-1] //= (per_im_ds_factor)

            focal.append(focal_over_W*W[-1])
            ds_factor.append(per_im_ds_factor)
            imgs.append(fname if class_config else resized_img)
            poses.append(np.array(frame["transform_matrix"]))
        # imgs = (np.array(imgs) / 255.0).astype(np.float32)
        poses = np.array(poses).reshape([-1,4,4]).astype(np.float32)
        counts.append(counts[-1] + len(imgs))
        all_imgs.append(imgs)
        all_poses.append(poses)

    imgs = [im for s in all_imgs for im in s]
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    if len(ds_f_nums)>0:
        i_split[0] = ([i for i in i_split[0] if i not in ds_f_nums],ds_f_nums)
        pose_dists = np.sum((np.stack([all_poses[0][i] for i in i_split[0][0]],0)[:,None,...]-all_poses[1][None,...])**2,axis=(2,3))
        closest_val = np.argmin(pose_dists)%pose_dists.shape[1]
        furthest_val = np.argmax(np.min(pose_dists,0))
        i_split[1] = (i_split[1],{"closest_val":closest_val,"furthest_val":furthest_val})

    poses = np.concatenate(all_poses, 0)
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
