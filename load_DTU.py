# Code taken from https://github.com/sxyu/pixel-nerf

import os
from unittest.mock import NonCallableMagicMock
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from torchvision import transforms
from collections import OrderedDict
from nerf_helpers import subsample_dataset

def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        # [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
        [transforms.ToTensor(),]
    )
    return transforms.Compose(ops)

def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )

def old_DTU_sceneID(id):
    return 'DTU'+str(id)

class DVRDataset(torch.utils.data.Dataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """

    def __init__(
        self,
        path,
        stage="train",
        list_prefix="new_",
        image_size=None,
        sub_format='dtu', #"shapenet",
        scale_focal=False, #True,
        max_imgs=100000,
        z_near=0.1, #1.2,
        z_far=5.0, #4.0,
        single_images:bool=True,
        eval_ratio:float=None,
        max_scenes:int=None,
        downsampling_factor:int=1,
        excluded_scenes:list=None,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        :param single_images If True, the dataset treats each image, rather than each scene, as a datapoint
        :param eval_ratio portion of scenes per image to be used for evaluation only, equaly spaced along the images order.
        """
        super().__init__()
        self.base_path = path
        self.single_images = single_images
        self.downsampling_factor = downsampling_factor
        assert eval_ratio is None or single_images,'eval_ratio can only be used when loader returns images, not full scenes.'
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(path, "*")) if os.path.isdir(x)]

        file_lists,val_lists = [],[]
        if "train" in stage:
            file_lists += sorted([os.path.join(x, list_prefix + "train.lst") for x in cats])
        if "val" in stage:
            val_lists.append(len(file_lists))
            file_lists += sorted([os.path.join(x, list_prefix + "val.lst") for x in cats])
        if "test" in stage:
            file_lists += sorted([os.path.join(x, list_prefix + "test.lst") for x in cats])

        self.all_objs,self.val_scenes = [],[]
        for l_num,file_list in enumerate(file_lists):
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            # if max_scenes is not None and len(objs)+len(self.all_objs)>max_scenes:
            #     # For saving time during debug
            #     print('!!! Keeping only %d scenes!!!'%(max_scenes))
            #     if len(self.all_objs)>=max_scenes:
            #         if len(self.val_scenes)==0: self.val_scenes = [max_scenes-1]
            #         break
            #     objs = objs[:max_scenes-len(self.all_objs)]
            if l_num in val_lists:  self.val_scenes.extend([i+len(self.all_objs) for i in range(len(objs))])
            self.all_objs.extend(objs)

        self.all_objs = sorted(self.all_objs)

        if max_scenes is not None and len(self.all_objs)>max_scenes:    # For saving time during debug
            print('!!! Keeping only %d scenes!!!'%(max_scenes))
            self.all_objs,self.val_scenes = subsample_dataset(scenes_dict=self.all_objs,max_scenes=max_scenes,val_only_scenes=self.val_scenes,scene_id_func=self.DTU_sceneID)
            # scenes_list = [self.sceneObj_2_name(obj) for obj in self.all_objs]
            # self.val_scenes = [i for i,s in enumerate(self.val_scene_IDs()) if s in scenes_list]

        if excluded_scenes is not None:
            scene_names = [self.DTU_sceneID(i) for i in range(len(self.all_objs))]
            scenes_2_remove = []
            for s in excluded_scenes:
                if s in scene_names:
                    scenes_2_remove.append(scene_names.index(s))
            temp = 1*self.val_scenes
            self.val_scenes = []
            for s_num in temp:
                if s_num in scenes_2_remove:    continue
                self.val_scenes.append(s_num-sum([v<s_num for v in scenes_2_remove]))
            self.all_objs = [s for i,s in enumerate(self.all_objs) if i not in scenes_2_remove]

        if self.single_images:
            self.im_IDs = []
            self.eval_inds,self.train_ims_per_scene = [],OrderedDict()
            self.i_val = {}
            counter = 0
            for id,obj in enumerate(self.all_objs):
                n = min(max_imgs,len([x for x in glob.glob(os.path.join(obj[1], "image", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]))
                self.im_IDs += [(i,id) for i in range(n)]
                eval_freq = n//(1 if eval_ratio is None else int(np.floor(eval_ratio*n)))
                if id not in self.val_scenes:
                    self.train_ims_per_scene[self.DTU_sceneID(id)] = [i+counter for i in range(n) if (i+1)%eval_freq!=0]
                eval_inds = [i+counter for i in range(n) if (i+1)%eval_freq==0]
                self.i_val[self.DTU_sceneID(id)] = eval_inds
                self.eval_inds += eval_inds
                counter += n

        self.stage = stage

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading DVR dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
            "type:",
            sub_format,
        )

        self.image_size = image_size
        if sub_format == "dtu":
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        self.sub_format = sub_format
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False

    def DTU_sceneID(self,id):
        return self.sceneObj_2_name(self.all_objs[id])

    def sceneObj_2_name(self,obj):
        return obj[1].split('/')[-1]

    def val_scene_IDs(self):
        return [self.DTU_sceneID(id) for id in self.val_scenes]

    def scene_IDs(self):
        return [self.DTU_sceneID(v[1]) for v in self.im_IDs]

    def __len__(self):
        if self.single_images:
            return len(self.im_IDs)
        else:
            return len(self.all_objs)

    def num_scenes(self):
        assert self.single_images
        return len(self.all_objs)

    def scene_info(self,scene_num):
        assert self.single_images
        self.single_images = False
        scene = self.__getitem__(scene_num)
        self.single_images = True
        return {'camera_poses':scene['poses'].numpy()[:,:3,:4],
            'H':[scene['images'].shape[2] for i in range(len(scene['poses']))],
            'W':[scene['images'].shape[3] for i in range(len(scene['poses']))],
            'f':[scene['focal'].numpy().tolist() for i in range(len(scene['poses']))]}

    def item(self,img_idx,device):
        data_item = self.__getitem__(img_idx)
        cur_focal = data_item['focal'].numpy().tolist()
        img_target = data_item['images'].squeeze(0).permute([1,2,0]).to(device)
        pose_target = data_item['poses'].squeeze(0)[:3].to(device)
        cur_H,cur_W = img_target.shape[:2]
        ds_factor = data_item['ds_factor']
        return img_target,pose_target,cur_H,cur_W,cur_focal,ds_factor

    def __getitem__(self, index):
        if self.single_images:
            im_ind,index = self.im_IDs[index]
        cat, root_dir = self.all_objs[index]

        rgb_paths = sorted([
            x
            for x in glob.glob(os.path.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ])
        if self.single_images:
            sel_indices = [im_ind]
            mask_paths = [None] * len(rgb_paths)
        else:
            mask_paths = sorted(glob.glob(os.path.join(root_dir, "mask", "*.png")))
            if len(mask_paths) == 0:
                mask_paths = [None] * len(rgb_paths)
            if len(rgb_paths) <= self.max_imgs:
                sel_indices = np.arange(len(rgb_paths))
            else:
                if self.single_images:
                    sel_indices = np.arange(self.max_imgs)
                else:
                    sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
        rgb_paths = [rgb_paths[i] for i in sel_indices]
        mask_paths = [mask_paths[i] for i in sel_indices]

        cam_path = os.path.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        focal = None
        if self.sub_format != "shapenet":
            # Prepare to average intrinsics over images
            fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if self.downsampling_factor!=1:
                resized_img = cv2.resize(img, dsize=(img.shape[1]//self.downsampling_factor, img.shape[0]//self.downsampling_factor), interpolation=cv2.INTER_AREA)

                # raise Exception('Unsupported yet')
            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            elif self.downsampling_factor!=1:
                x_scale = resized_img.shape[1]/img.shape[1]
                y_scale = resized_img.shape[0]/img.shape[0]
                xy_delta = 0.0
                img = resized_img
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]
            if self.sub_format == "dtu":
                # Decompose projection matrix
                # DVR uses slightly different format for DTU set
                P = all_cam["world_mat_" + str(i)]
                P = P[:3]

                K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
                K = K / K[2, 2]

                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R.transpose()
                pose[:3, 3] = (t[:3] / t[3])[:, 0]

                scale_mtx = all_cam.get("scale_mat_" + str(i))
                if scale_mtx is not None:
                    norm_trans = scale_mtx[:3, 3:]
                    norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

                    pose[:3, 3:] -= norm_trans
                    pose[:3, 3:] /= norm_scale

                fx += torch.tensor(K[0, 0]) * x_scale
                fy += torch.tensor(K[1, 1]) * y_scale
                cx += (torch.tensor(K[0, 2]) + xy_delta) * x_scale
                cy += (torch.tensor(K[1, 2]) + xy_delta) * y_scale
            else:
                # ShapeNet
                wmat_inv_key = "world_mat_inv_" + str(i)
                wmat_key = "world_mat_" + str(i)
                if wmat_inv_key in all_cam:
                    extr_inv_mtx = all_cam[wmat_inv_key]
                else:
                    extr_inv_mtx = all_cam[wmat_key]
                    if extr_inv_mtx.shape[0] == 3:
                        extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                    extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

                intr_mtx = all_cam["camera_mat_" + str(i)]
                fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
                assert abs(fx - fy) < 1e-9
                fx = fx * x_scale
                if focal is None:
                    focal = fx
                else:
                    assert abs(fx - focal) < 1e-5
                pose = extr_inv_mtx

            pose = (
                self._coord_trans_world
                @ torch.tensor(pose, dtype=torch.float32)
                @ self._coord_trans_cam
            )

            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError(
                        "ERROR: Bad image at", rgb_path, "please investigate!"
                    )
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

            all_imgs.append(img_tensor)
            all_poses.append(pose)

        if self.sub_format != "shapenet":
            fx /= len(rgb_paths)
            fy /= len(rgb_paths)
            cx /= len(rgb_paths)
            cy /= len(rgb_paths)
            focal = torch.tensor((fx, fy), dtype=torch.float32)
            c = torch.tensor((cx, cy), dtype=torch.float32)
            all_bboxes = None
        elif mask_path is not None:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            if self.sub_format != "shapenet":
                c *= scale
            elif mask_path is not None:
                all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            "ds_factor": self.downsampling_factor
        }
        if all_masks is not None:
            result["masks"] = all_masks
        if self.sub_format != "shapenet":
            result["c"] = c
        else:
            result["bbox"] = all_bboxes
        return result