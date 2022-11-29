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
from copy import deepcopy
import load_llff

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

FIGURE_IMAGES_MODE = False
class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self,config,scene_id_func,add_val_scene_LR,eval_mode,scene_norm_coords=None) -> None:
        ON_THE_FLY_SCENES_THRESHOLD = 2 if eval_mode else 20
        super(BlenderDataset,self).__init__()
        if FIGURE_IMAGES_MODE:  assert eval_mode
        self.get_scene_id = scene_id_func
        train_dirs = self.get_scene_configs(getattr(config.dir,'train',{}),add_val_scene_LR=add_val_scene_LR)
        val_scenes_dict = getattr(config.dir,'val',{})
        if False and eval_mode:
            inferred_ds_factor = max(train_dirs[0])/min(train_dirs[0])
            assert inferred_ds_factor==int(inferred_ds_factor)
            inferred_ds_factor = int(inferred_ds_factor)
            if inferred_ds_factor>1 and len(getattr(config.dir,'val',{}))>0:
                val_scenes_dict_copy = deepcopy(val_scenes_dict)
                for k,v in val_scenes_dict_copy.items():
                    conf = [int(v) for v in k.split(',')]
                    val_scenes_dict.update({','.join([str(v) for v in [conf[0]*inferred_ds_factor,conf[1]//inferred_ds_factor,conf[2]]]):v})
        self.downsampling_factors,self.all_scenes,plane_resolutions,val_ids,scene_types = self.get_scene_configs(val_scenes_dict)
        self.downsampling_factors += train_dirs[0]
        plane_resolutions += train_dirs[2]
        train_ids = train_dirs[3]
        if getattr(config,'auto_remove_val',False): assert not any([id in train_ids for id in val_ids]),'Removed support to this option. Should re-enable.'
        scene_types += train_dirs[4]
        if len(set(train_ids+val_ids))!=len(train_ids+val_ids) and not eval_mode:
            raise Exception('I suspect an overlap between training and validation scenes. The following appear in both:\n%s'%([s for s in val_ids if s in train_ids]))
        train_dirs = train_dirs[1]
        self.all_scenes.extend(train_dirs)
        self.images, self.poses, render_poses, self.hwfDs, i_split,self.per_im_scene_id = [],torch.zeros([0,4,4]),[],[],[np.array([]).astype(np.int64) for i in range(3)],[]
        scene_id,self.val_only_scene_ids,self.coords_normalization = -1,[],{}
        self.scene_id_plane_resolution = {}
        self.i_train,self.i_val = OrderedDict(),OrderedDict()
        self.scenes_set = set()
        self.ds_kernels = {}
        self.scene_types = {}
        self.on_the_fly_load = len(self.all_scenes)>ON_THE_FLY_SCENES_THRESHOLD
        for basedir,ds_factor,plane_res,scene_type in zip(tqdm(self.all_scenes,desc='Loading scenes'),self.downsampling_factors,plane_resolutions,scene_types):
            scene_id = self.get_scene_id(basedir,ds_factor,plane_res)
            if scene_id in self.i_train:
                raise Exception("Scene %s already in the set"%(scene_id))
            self.scenes_set.add(scene_id)
            val_only = (scene_id in val_ids or len(val_ids)==0) if eval_mode else (scene_id not in train_ids)
            if val_only:    self.val_only_scene_ids.append(scene_id)
            self.scene_id_plane_resolution[scene_id] = plane_res
            if eval_mode:
                if not val_only:    continue
                splits2use = ['test']
            else:
                splits2use = ['val'] if val_only else ['train','val']
            if not hasattr(config,scene_type):
            # if not isinstance(config.root,dict): #Legacy support
                assert scene_type=='synt'
                setattr(config,scene_type,{'root':config.root,'near':config.near,'far':config.far})
                # scene_path = os.path.join(config.root,basedir)
            scene_path = os.path.join(config[scene_type].root,basedir)
            if search('##',basedir) is not None:
                if search('##(\d)+',basedir) is not None:
                    scene_path = scene_path.replace(search('##(\d)+',basedir).group(0),'')
                elif search('##Gauss(\d)+(\.)?(\d)*',basedir) is not None:
                    assert ds_factor>1
                    assert not self.on_the_fly_load,"Blurring each image every time would be very slow (consider upgrading to PyTorch Dataloader if necessary)"
                    scene_path = scene_path.replace(search('##Gauss(\d)+(\.)?(\d)*',basedir).group(0),'')
                    self.ds_kernels[scene_id] = {'base_factor':min(self.downsampling_factors),'STD':float(search('(?<=##Gauss)(\d)+(\.)?(\d)*(?=$)',basedir).group(0))}
            # if getattr(config,'type','synt')=='synt':
            self.scene_types[scene_id] = scene_type
            if scene_type=='synt':
                cur_images, cur_poses, cur_render_poses, cur_hwfDs, cur_i_split = load_blender_data(
                    scene_path,
                    testskip=config.testskip,
                    downsampling_factor=ds_factor,
                    val_downsampling_factor=None,
                    splits2use=splits2use,
                    load_imgs=not self.on_the_fly_load,
                    blur_kernel=self.ds_kernels[scene_id] if scene_id in self.ds_kernels else None,
                )
            # elif config.type=='llff':
            elif scene_type=='llff':
                assert scene_id not in self.ds_kernels,'Unsupported'
                cur_images, cur_poses, _, _, cur_i_split = load_llff.load_llff_data(
                    scene_path,
                    factor=ds_factor,
                    load_imgs=not self.on_the_fly_load,
                )
                cur_images = [im for im in cur_images]
                cur_hwfDs = cur_poses[0, :3, -1]
                cur_hwfDs = [int(cur_hwfDs[0]),int(cur_hwfDs[1]),cur_hwfDs[2].item(),ds_factor]
                cur_hwfDs = [len(cur_images)*[v] for v in cur_hwfDs]
                cur_poses = torch.cat([cur_poses[:, :3, :4],(torch.ones([cur_poses.shape[0],1,1])*torch.tensor([0,0,0,1]).reshape([1,1,-1])).type(cur_poses.type())],1)
                EXCLUDE_VAL_FROM_TRAINING = False
                if eval_mode:
                    cur_i_split = [[],[],[i for i in range(len(cur_images))]]
                else:
                    if getattr(config,'llffhold',0)>0:
                        cur_i_split = [(i+len(cur_images)//(2*config.llffhold))%len(cur_images) for i in np.unique(np.round(np.linspace(0,len(cur_images)-1,config.llffhold+1)).astype(int))][:config.llffhold]
                    else:
                        cur_i_split = [cur_i_split]
                    if EXCLUDE_VAL_FROM_TRAINING:
                        cur_i_split = [
                            np.array([i for i in np.arange(len(cur_images)) if (i not in cur_i_split)]),cur_i_split,cur_i_split
                        ]
                    else:
                        cur_i_split = [
                            np.array([i for i in np.arange(len(cur_images))]),cur_i_split,cur_i_split
                        ]
            if scene_norm_coords is not None: # No need to calculate the per-scene normalization coefficients as those will be loaded with the saved model.
                self.coords_normalization[scene_id] =\
                    calc_scene_box({'camera_poses':cur_poses.numpy()[:,:3,:4],'near':config[scene_type].near,'far':config[scene_type].far,'H':cur_hwfDs[0],'W':cur_hwfDs[1],'f':cur_hwfDs[2]},
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
        cur_H,cur_W,cur_focal,cur_ds_factor = self.hwfDs[index]
        if self.on_the_fly_load:
            img_target = (imageio.imread(self.images[index])/ 255.0).astype(np.float32)
            if cur_ds_factor>1:
                img_target = im_resize(img_target, scale_factor=cur_ds_factor,
                    blur_kernel=self.ds_kernels[self.per_im_scene_id[index]] if self.per_im_scene_id[index] in self.ds_kernels else None)
            img_target = torch.from_numpy(img_target)
        else:
            img_target = self.images[index]
        pose_target = self.poses[index].to(device)
        return img_target.to(device),pose_target,cur_H,cur_W,cur_focal,cur_ds_factor

    def __len__(self):
        return len(self.images)

    def get_scene_configs(self,config_dict,add_val_scene_LR=False,excluded_scene_ids=[]):
        ds_factors,dir,plane_res,scene_ids,types = [],[],[],[],[]
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
            for s in interpret_scene_list(scenes):
                cur_factor,cur_dir,cur_res,cur_type = conf[0],s,(conf[1],conf[2] if len(conf)>2 else conf[1]),conf[3] if len(conf)>3 else 'synt'
                cur_id = self.get_scene_id(cur_dir,cur_factor,cur_res)
                if cur_id in excluded_scene_ids:
                    continue
                scene_ids.append(cur_id)
                ds_factors.append(cur_factor)
                plane_res.append(cur_res)
                dir.append(cur_dir)
                types.append(cur_type)
        return ds_factors,dir,plane_res,scene_ids,types

def load_blender_data(basedir, half_res=False, testskip=1, debug=False,
        downsampling_factor=1,val_downsampling_factor=None,cfg=None,splits2use=['train','val'],load_imgs=True,blur_kernel=None):
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
            # if FIGURE_IMAGES_MODE and f_num>=10:
            #     print("!!!!!WARNING!!!!!!!")
            #     break
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            if s=='val':
                per_im_ds_factor = 1*val_downsampling_factor
            else:
                per_im_ds_factor = 1*downsampling_factor
            if load_imgs:
                img = (imageio.imread(fname)/ 255.0).astype(np.float32)
                H.append(img.shape[0])
                W.append(img.shape[1])
                resized_img = torch.from_numpy(im_resize(img, scale_factor=per_im_ds_factor,blur_kernel=blur_kernel))
            else:
                im_dims = [int(v) for v in search('(\d+) x (\d+)', from_file(fname)).groups()]
                assert len(im_dims)==2 and im_dims[0]==im_dims[1],"Should verify the order of H,W"
                H.append(im_dims[0])
                W.append(im_dims[1])
            H[-1] //= (per_im_ds_factor)
            W[-1] //= (per_im_ds_factor)

            focal.append(focal_over_W*W[-1])
            ds_factor.append(per_im_ds_factor)
            imgs.append(resized_img if load_imgs else fname)
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
