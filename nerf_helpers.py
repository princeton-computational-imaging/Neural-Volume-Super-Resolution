from collections import OrderedDict,deque,defaultdict
from pickletools import uint8
from typing import Optional
import numpy as np
import math
import torch
from cfgnode import CfgNode
import yaml
import cv2
import torchvision
from re import search
import functools
from imresize import imresize
import os
import sys
import pickle
import imageio

def safe_saving(file_name,content,suffix,best=False,run_time_signature=0):
    if run_time_signature:
        run_folder = os.path.dirname(file_name.replace('/planes/','/'))
        run_signature_file = os.path.join(run_folder,'time_sig.txt')
        if os.path.exists(run_signature_file):
            with open(run_signature_file,'r') as f:
                saved_signature = float(f.read())
            if saved_signature<run_time_signature:
                with open(run_signature_file,'w') as f:
                    f.write(str(run_time_signature))
            elif saved_signature>run_time_signature:
                sys.exit('Exiting run %f since a newer run %f has started.'%(run_time_signature,saved_signature))
        else:
            with open(run_signature_file,'w') as f:
                f.write(str(run_time_signature))

    if best:
        file_name = file_name.replace('.%s'%(suffix),'.%s_best'%(suffix))
    if suffix=='pkl':
        with open(file_name.replace('.%s'%(suffix),'.%s_temp'%(suffix)), 'wb') as f:
            pickle.dump(content, f)
    else:
        torch.save(content,file_name.replace('.%s'%(suffix),'.%s_temp'%(suffix)))
    del_bckp = False
    if os.path.isfile(file_name):
        del_bckp = True
        os.rename(file_name,file_name.replace('.%s'%(suffix),'.%s_bckp'%(suffix)))
    os.rename(file_name.replace('.%s'%(suffix),'.%s_temp'%(suffix)),file_name)
    if del_bckp:
        os.remove(file_name.replace('.%s'%(suffix),'.%s_bckp'%(suffix)))

def safe_loading(file_name,suffix,best=False):
    if best:
        file_name = file_name.replace('.%s'%(suffix),'.%s_best'%(suffix))
    for version in ['','_temp','_bckp']:
        try:
            if suffix=='pkl':
                with open(file_name.replace('.%s'%(suffix),'.%s%s'%(suffix,version)), 'rb') as f:
                    content = pickle.load(f)
            else:
                content = torch.load(file_name.replace('.%s'%(suffix),'.%s%s'%(suffix,version)))
            break
        except Exception as e:
            if version=='_bckp':
                raise e
            else:
                print("!!!! WARNING: saved file %s seems to be currpted. Loading their backup instead:\n%s"%\
                    (file_name.replace('.%s'%(suffix),'.%s%s'%(suffix,version)),e))
    return content

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def deep_update(org_dict,new_dict):
    for k,v in new_dict.items():
        if isinstance(v,dict):
            org_dict[k] = deep_update(org_dict[k],v)
        else:
            org_dict[k] = v
    return org_dict
    
class null_with:
    def __enter__(self):
        pass
    def __exit__(self,a,b,c):
        pass

def bicubic_interp(im,sf):
    return cv2.resize(im, dsize=(im.shape[1]*sf,im.shape[0]*sf), interpolation=cv2.INTER_CUBIC)
class Counter:
    def __init__(self) -> None:
        self.counter = 0
        self.flag = False

    def count(self):
        return self.counter

    def step(self,print_str:str=None):
        self.counter += 1
        self.flag = True
        if print_str is not None:
            print(print_str+str(self.count()))

    def check_and_reset(self):
        if self.flag:
            self.flag = False
            return True
        return False

class RunningMeanScores:
    def __init__(self,per_scene_len,val_strings,running_mean_logs) -> None:
        self.running_scores = defaultdict(dict)
        for sc in per_scene_len:
            self.running_scores[val_strings[sc]][sc] = dict(zip(running_mean_logs,[deque(maxlen=per_scene_len[sc]) for i in running_mean_logs]))
        self.running_scores['train'] = dict(zip(running_mean_logs,[deque(maxlen=len([sc for sc in per_scene_len if '_train' in sc])) for i in running_mean_logs]))
        # dict([(score,dict([(cat,deque(maxlen=len(training_scenes) if cat=='train' else val_ims_per_scene)) for cat in list(set(val_strings))+['train']])) for score in running_mean_logs])

    def update(self,val_set,metric,scores_dict):
        if val_set=='train':
            self.running_scores['train'][metric].append(v)
        else:
            for k,v in scores_dict.items():
                self.running_scores[val_set][k][metric].append(v)

    def get(self,val_set,metric):
        if val_set=='train':
            return np.mean(self.running_scores['train'][metric])
        else:
            return np.mean([np.mean(self.running_scores[val_set][k][metric]) for k in self.running_scores[val_set]])

class ImageSampler:
    def __init__(self,scenes_dict:dict,scene_probs:list=[]) -> None:
        self.scenes_dict = scenes_dict
        self.scene_probs = scene_probs
        # assert same_scene_probs,'Unsupported yet.'

    def update_active(self,active_scenes):
        self.active_scenes = active_scenes
        self.im_inds,self.im_probs = [],[]
        for sc_num,sc in enumerate(active_scenes):
            self.im_inds.extend([(sc_num,ind) for ind in self.scenes_dict[sc]])
            if sc in self.scene_probs:
                self.im_probs.extend(len(self.scenes_dict[sc])*[self.scene_probs[sc]/len(self.scenes_dict[sc])])
            else:
                self.im_probs.extend(len(self.scenes_dict[sc])*[1/len(self.scenes_dict[sc])])
            # self.im_probs.extend(len(self.scenes_dict[sc])*[1/len(self.scenes_dict[sc])])
        self.im_probs = np.array(self.im_probs)
        self.im_probs /= np.sum(self.im_probs)

    def sample(self):
        chosen = np.random.choice(len(self.im_inds),size=1,p=self.im_probs)[0]
        return self.active_scenes[self.im_inds[chosen][0]],self.im_inds[chosen][1]
    

def set_config_defaults(source,target):
    for k in source.keys():
        if k not in target: setattr(target,k,getattr(source,k))
        elif isinstance(source[k],dict):
            set_config_defaults(source[k],target[k])

def interpret_scene_list(dict_values):
    scenes = []
    for sc in dict_values:
        if isinstance(sc,list):
            assert len(sc)==2
            scenes.extend([str(i) for i in range(sc[0],sc[1])])
        else:
            scenes.append(sc)
    return scenes

def find_scene_match(existing_scenes,pattern):
    matching_scene = [es for es in existing_scenes if search(pattern,es) is not None]
    assert len(matching_scene)==1
    return matching_scene[0],search(pattern,matching_scene[0]).group(0)

def image_STD_2_distribution(patch_size):
    unfold = torch.nn.Unfold(kernel_size=(patch_size,patch_size))
    def mapper(image):
        assert image.dim()==3 and image.shape[2]==3
        im_STD = unfold(image.permute(2,0,1).unsqueeze(0)).reshape(1,3,patch_size**2,-1).std(2).reshape(3,image.shape[0]-patch_size+1,image.shape[1]-patch_size+1).mean(0)
        return im_STD/im_STD.sum()
    return mapper

def is_background_white(image):
    perimeter = torch.cat([image[0,...].reshape([-1]),image[-1,...].reshape([-1]),image[:,0,...].reshape([-1]),image[:,-1,...].reshape([-1]),])
    black_portion = (perimeter<=10/256).float().mean()
    white_portion = (perimeter>=246/256).float().mean()
    # black_portion = (image<=10/256).float().mean()
    # white_portion = (image>=246/256).float().mean()
    # PORTION_THRESHOLD = 0.4
    # assert torch.bitwise_xor(black_portion>PORTION_THRESHOLD,white_portion>PORTION_THRESHOLD),'Not sure about the background color: %.2f black, %.2f white'%(black_portion,white_portion)
    # if black_portion>PORTION_THRESHOLD:
    #     return 0
    # elif white_portion>PORTION_THRESHOLD:
    #     return 1
    return white_portion>black_portion
    #     return 1
    # else:
    #     return 0
def subsample_dataset(max_scenes,pick_first=False,scene_types=[]):
    scenes_nums2keep = []
    for scene_type in set(scene_types):
        scene_nums = [i for i,sc_t in enumerate(scene_types) if sc_t==scene_type]
        inds2keep = [i for i in (range(min(len(scene_nums),max_scenes)) if pick_first else np.unique(np.round(np.linspace(0,len(scene_nums)-1,max_scenes)).astype(int)))]
        scenes_nums2keep.extend([scene_nums[i] for i in inds2keep])
    return scenes_nums2keep


def subsample_dataset_old(scenes_dict,max_scenes,val_only_scenes=[],scene_id_func=None,max_val_only_scenes=None):
    convert2list, = False,
    val_scene_as_inds = len(val_only_scenes)>0 and isinstance(val_only_scenes[0],int)
    if val_scene_as_inds:
        val_only_scenes = [scene_id_func(i) for i in val_only_scenes]
    if isinstance(scenes_dict,list):
        convert2list = True
        scenes_dict = OrderedDict(zip([scene_id_func(i) for i in range(len(scenes_dict))],scenes_dict))
    else:
        assert scene_id_func is None
        scene_id_func = lambda x:x
    i_val_val_only = [(k,v) for k,v in scenes_dict.items() if k in val_only_scenes]
    i_val_others = [(k,v) for k,v in scenes_dict.items() if k not in val_only_scenes]
    sampled_dict = OrderedDict([i_val_others[i] for i in np.unique(np.round(np.linspace(0,len(i_val_others)-1,max_scenes)).astype(int))])
    sampled_val_only_scenes = []
    if len(i_val_val_only)>0:
        if max_val_only_scenes is None:
            sampled_val_only_scenes = i_val_val_only
        else:
            sampled_val_only_scenes = [i_val_val_only[i] for i in np.unique(np.round(np.linspace(0,len(i_val_val_only)-1,max_val_only_scenes)).astype(int))]
        sampled_dict.update(OrderedDict(sampled_val_only_scenes))
        sampled_val_only_scenes = [s[0] for s in sampled_val_only_scenes]
        if val_scene_as_inds:
            sampled_val_only_scenes = [list(sampled_dict.keys()).index(s) for s in sampled_val_only_scenes]
    if convert2list:
        sampled_dict = list(sampled_dict.values())
    return sampled_dict,sampled_val_only_scenes


def estimated_background_2_distribution(patch_size):
    def mapper(image):
        white_bg = is_background_white(image)
        image = integral_image(torch.any(image<246/256 if white_bg else image>10/256,-1))
        image = image[patch_size:,patch_size:]+image[:-(patch_size),:-(patch_size)]-image[patch_size:,:-(patch_size)]-image[:-(patch_size),patch_size:]
        return image.float()/image.sum()
    return mapper

def integral_image(image):
    return torch.cat([torch.zeros([1+image.shape[0],1]).type(image.type()),torch.cat([torch.zeros([1,image.shape[1]]).type(image.type()),torch.cumsum(torch.cumsum(image,0),1)],0)],1)

def img2mse(img_src, img_tgt):
    return torch.nn.functional.mse_loss(img_src, img_tgt)

def imread(path,with_alpha=False):
    image = imageio.imread(path)
    if not with_alpha and image.shape[2]>3:
        image = image[...,:3]*(image[...,3:]>0)
    return (image/255.0).astype(np.float32)

def assert_list(input):
    return input if isinstance(input,list) else [input]

def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)

def num_parameters(model):
    return sum([p.numel() for p in model.parameters()])

def chunksize_to_2D(chunksize):
    return math.floor(math.sqrt(chunksize))

def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8,spatial_margin=None):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    if spatial_margin is not None:
        chunksize = chunksize_to_2D(chunksize)
        # return [inputs[max(0,i-spatial_margin):i+chunksize+spatial_margin,max(0,j-spatial_margin):j+chunksize+spatial_margin,...].reshape([-1,inputs.shape[-1]]) \
        return [inputs[i-spatial_margin:i+chunksize+spatial_margin,j-spatial_margin:j+chunksize+spatial_margin,...] \
            for i in range(spatial_margin,inputs.shape[0]-2*spatial_margin, chunksize) for j in range(spatial_margin,inputs.shape[1]-2*spatial_margin, chunksize)]
    else:
        return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def get_config(config_path):
    with open(config_path, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        return CfgNode(cfg_dict)

def im_resize(image,scale_factor,degradation=None,fname=None):
    assert all([v%scale_factor==0 for v in image.shape[:2]]),'Currently not supporting downscaling to an ambiguous size.'
    cv2_scale_factor = scale_factor if degradation is None else degradation['base_factor']
    output = cv2.resize(image, dsize=(image.shape[1]//cv2_scale_factor,image.shape[0]//cv2_scale_factor), interpolation=cv2.INTER_AREA)
    if degradation is not None and scale_factor>cv2_scale_factor:
        assert degradation['type'] in ['blur','noise']
        if degradation['type']=='blur':
            output = np.clip(imresize(output,scale_factor=1/(scale_factor/cv2_scale_factor),kernel='blurry_cubic_%f'%(degradation['STD']),),0,1).astype(output.dtype)
        elif degradation['type']=='noise':
            output = imresize(output,scale_factor=1/(scale_factor/cv2_scale_factor)).astype(output.dtype)
            if not os.path.isdir(degradation['path']):  os.mkdir(degradation['path'])
            noise_file = os.path.join(degradation['path'],fname+'_%d.npz'%(degradation['STD']))
            if not os.path.exists(noise_file):
                noise = np.random.normal(scale=degradation['STD']/255,size=output.shape)
                np.savez(noise_file,noise=noise)
            output = np.clip(output+np.load(noise_file)['noise'],0,1).astype(output.dtype)
    return output

def calc_resize_crop_margins(im_shape,ds_factor):
    crop_origs = any([v%(ds_factor) for v in im_shape])
    marg2crop = None
    if crop_origs:
        marg2crop = np.zeros([2]).astype(np.int32)
        for dim in [0,1]:
            while (im_shape[dim]-2*marg2crop[dim])%(ds_factor):
                marg2crop[dim] += 1
                assert marg2crop[dim]<ds_factor,'Cannot find a suitable crop'
    return marg2crop

def arange_ims(ims_tensor_list,text,psnrs=[],): #fontScale=1):
    num_rows = lambda n_cols:   int(np.ceil(len(ims_tensor_list)/n_cols))
    num_cols = 1
    while num_cols*ims_tensor_list[0].shape[1]<num_rows(num_cols)*ims_tensor_list[0].shape[0]:
        if num_cols==len(ims_tensor_list): break
        num_cols += 1
    ims_size = sorted([im.shape[:2] for im in ims_tensor_list],key=lambda x:x[0]*x[1])[-1] #Using the largets image's shape
    rows = []
    psnrs += (len(ims_tensor_list)-len(psnrs))*[None]
    for im_num,im in enumerate(ims_tensor_list):
        if im_num%num_cols==0:
            if im_num>0:    rows.append(np.concatenate(row,1))
            row = []
        row.append(
            cast_to_image(
                cv2.resize(np.array(255*im.cpu()).astype(np.uint8),dsize=(ims_size[1],ims_size[0]),interpolation=cv2.INTER_NEAREST),
                img_text=text if im_num==0 else None,psnr=psnrs[im_num], #fontScale=fontScale
            ).transpose(1,2,0)
        )
    row = np.concatenate(row,1)
    rows.append(np.pad(row,((0,0),(0,num_cols*ims_size[1]-row.shape[1]),(0,0)),))
    return np.concatenate(rows,0).transpose(2,0,1)

def cast_to_image(img,img_text=None,psnr=None,): #fontScale=1):
    if isinstance(img,torch.Tensor):
        # Input tensor is (H, W, 3). Convert to (3, H, W).
        img = img.permute(2, 0, 1)
        # Conver to PIL Image and then np.array (output shape: (H, W, 3))
        img = np.array(torchvision.transforms.ToPILImage()(img.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    fontScale = max(1,cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_PLAIN,int(0.05*img.shape[1])))
    if img_text is not None:
        text_color = ((np.mean(img[:,:int(fontScale*img.shape[1]//24),:int(fontScale*15)],axis=(1,2)).astype(np.uint8)+[128,128,128])%256).tolist()
        img = cv2.putText(
            img.transpose(1,2,0),
            img_text,
            (0,int(fontScale*15)),
            cv2.FONT_HERSHEY_PLAIN,
            fontScale=fontScale,
            color=text_color,#[255,255,255],
            thickness=int(np.ceil(np.sqrt(fontScale))),
        ).transpose(2,0,1)
    if psnr is not None:
        psnr_color = ((np.mean(img[:,-int(fontScale*img.shape[1]//24):,img.shape[2]//2:],axis=(1,2)).astype(np.uint8)+[128,128,128])%256).tolist()
        img = cv2.putText(
            img.transpose(1,2,0),
            '%.2f'%(psnr),
            (max(0,img.shape[2]//2-int(fontScale*15)),img.shape[1]),
            # (img.shape[2]//2,img.shape[1]),
            cv2.FONT_HERSHEY_PLAIN,
            fontScale=fontScale,
            color=psnr_color,
            thickness=int(np.ceil(np.sqrt(fontScale))),
        ).transpose(2,0,1)

    return img


def spatial_batch_merge(batches_list,batch_shapes,im_shape):
    b_num = 0
    rows_list = []
    while b_num<len(batches_list):
        cur_row,cur_width = [batches_list[b_num].reshape(list(batch_shapes[b_num])+[-1])],batch_shapes[b_num][1]
        while cur_width<im_shape[1]:
            b_num += 1
            cur_row.append(batches_list[b_num].reshape(list(batch_shapes[b_num])+[-1]))
            cur_width += batch_shapes[b_num][1]
        rows_list.append(torch.cat(cur_row,1))
        b_num += 1
    full_array = torch.cat(rows_list,0)
    return full_array.reshape([full_array.shape[0]*full_array.shape[1]]+list(full_array.shape[2:]))

def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> tuple((torch.Tensor, torch.Tensor)):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2,indexing='ij')
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod

def get_focal(data,dim:int):
    assert dim in ['H','W']
    if isinstance(data,list):
        return data[1] if dim=='H' else data[0]
    else:
        return data


def calc_scene_box(scene_geometry,including_dirs,no_ndc,adjust_az_range=False,adjust_elevation_range=False):
    # FULL_AZ_RANGE = True # Manually set azimuth range to be [-pi,pi]
    # if full_elev_range: full_elev_range = [-np.pi/2,np.pi/2]
    EXHAUSTIVE_CHECK = 10
    def list2pix(end_pixels):
        if EXHAUSTIVE_CHECK:
            if EXHAUSTIVE_CHECK==-1:
                return [i for i in range(end_pixels[0],end_pixels[1])]
            else:
                return np.unique(np.round(np.linspace(end_pixels[0],end_pixels[1],2+EXHAUSTIVE_CHECK)).astype(int))
        else:
            return end_pixels

    num_frames = len(scene_geometry['camera_poses'])
    box = [[np.finfo(np.float).max,np.finfo(np.float).min] for i in range(3+2*including_dirs)]
    for f_num in range(num_frames):
        origin_ = scene_geometry['camera_poses'][f_num][:3, -1]
        for col in list2pix([0,scene_geometry['W'][f_num]-1]):
            for row in list2pix([0,scene_geometry['H'][f_num]-1]):
                coord = np.array([\
                    (col-scene_geometry['W'][f_num]/2)/get_focal(scene_geometry['f'][f_num],'W'),
                    -(row-scene_geometry['H'][f_num]/2)/get_focal(scene_geometry['f'][f_num],'H'),
                    -1
                ])
        # for corner in [np.array([i,j,1]) for i in [-1,1] for j in [-1,1]]:
                dir = np.sum(coord * scene_geometry['camera_poses'][f_num][:3, :3], axis=-1)
                normed_dir = dir/np.linalg.norm(dir)
                if no_ndc:
                    origin = 1*origin_
                else:
                    assert not isinstance(scene_geometry['f'][f_num],list),'Currently not supporting non-isotropic focal length with ndc'
                    origin,dir = ndc_rays(scene_geometry['H'][f_num], scene_geometry['W'][f_num],\
                        scene_geometry['f'][f_num], near=1., rays_o=torch.tensor(origin_),rays_d=torch.tensor(dir))
                    origin,dir = origin.numpy(),dir.numpy()
                for dist in [scene_geometry['near'],scene_geometry['far']]:
                    point = origin+dist*dir
                    for d in range(3):
                        box[d][0] = min(box[d][0],point[d])
                        box[d][1] = max(box[d][1],point[d])
                if including_dirs and (adjust_az_range or adjust_elevation_range):
                    az_el = cart2az_el(torch.tensor(normed_dir)).numpy()
                    for d in range(int(not adjust_az_range),2):
                        box[3+d][0] = min(box[3+d][0],az_el[d])
                        box[3+d][1] = max(box[3+d][1],az_el[d])
    if including_dirs:
        if not adjust_az_range:   box[3] = [-np.pi,np.pi]
        if not adjust_elevation_range:
            box[4] = [-np.pi/2,np.pi/2]
        else:
            box[4] = list(adjust_elevation_range*(np.array(box[4])-np.mean(box[4]))+np.mean(box[4]))
    return torch.from_numpy(np.array(box).transpose(1,0))

def cart2az_el(dirs):
    el = torch.atan2(dirs[...,2],torch.sqrt(torch.sum(dirs[...,:2]**2,-1)))
    # el = torch.atan2(torch.sqrt(torch.sum(dirs[...,:2]**2,-1)),dirs[...,2])
    az = torch.atan2(dirs[...,1],dirs[...,0])
    return torch.stack([az,el],-1)

def downsample_plane(plane,ds_factor,plane_interp,align_corners,antialias=False):
    return torch.nn.functional.interpolate(plane,scale_factor=1/ds_factor,mode=plane_interp,align_corners=align_corners,antialias=antialias)

def calc_im_consistency_loss(sr,ds_factor,plane_interp,align_corners=True,gt_lr=None,gt_hr=None):
    assert (gt_lr is None) ^ (gt_hr is None)
    loss = torch.nn.functional.l1_loss(gt_lr if gt_hr is None else downsample_plane(gt_hr,ds_factor,plane_interp,align_corners,antialias=True),
                                       downsample_plane(sr,ds_factor,plane_interp,align_corners,antialias=True))
    return loss

def get_ray_bundle(
    height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor,padding_size: int=0,downsampling_offset: float=0
):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    """
    # TESTED
    ii, jj = meshgrid_xy(
        (torch.arange(width+2*padding_size)+downsampling_offset).to(tform_cam2world),
        (torch.arange(height+2*padding_size)+downsampling_offset).to(tform_cam2world),
    )
    if padding_size>0:
        ii = ii-padding_size
        jj = jj-padding_size
    directions = torch.stack(
        [
            (ii - width * 0.5) / get_focal(focal_length,'H'),
            -(jj - height * 0.5) / get_focal(focal_length,'W'),
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    # Now, encode the input using a set of high-frequency functions and append the
    # resulting values to the encoding.
    for i in range(num_encoding_functions):
        for func in [torch.sin, torch.cos]:
            encoding.append(func(2.0 ** i * tensor))
    return torch.cat(encoding, dim=-1)


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # UNTESTED, but fairly sure.

    # Shift rays origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def gather_cdf_util(cdf, inds):
    r"""A very contrived way of mimicking a version of the tf.gather()
    call used in the original impl.
    """
    orig_inds_shape = inds.shape
    inds_flat = [inds[i].view(-1) for i in range(inds.shape[0])]
    valid_mask = [
        torch.where(ind >= cdf.shape[1], torch.zeros_like(ind), torch.ones_like(ind))
        for ind in inds_flat
    ]
    inds_flat = [
        torch.where(ind >= cdf.shape[1], (cdf.shape[1] - 1) * torch.ones_like(ind), ind)
        for ind in inds_flat
    ]
    cdf_flat = [cdf[i][ind] for i, ind in enumerate(inds_flat)]
    cdf_flat = [cdf_flat[i] * valid_mask[i] for i in range(len(cdf_flat))]
    cdf_flat = [
        cdf_chunk.reshape([1] + list(orig_inds_shape[1:])) for cdf_chunk in cdf_flat
    ]
    return torch.cat(cdf_flat, dim=0)


def sample_pdf(bins, weights, num_samples, det=False):
    # TESTED (Carefully, line-to-line).
    # But chances of bugs persist; haven't integration-tested with
    # training routines.

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / weights.sum(-1).unsqueeze(-1)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, num_samples).to(weights)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(weights)

    # Invert CDF
    inds = torch.searchsorted(
        cdf.contiguous(), u.contiguous(), side="right"
    )
    below = torch.max(torch.zeros_like(inds), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), -1)
    orig_inds_shape = inds_g.shape

    cdf_g = gather_cdf_util(cdf, inds_g)
    bins_g = gather_cdf_util(bins, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def sample_pdf_2(bins, weights, num_samples, det=False):
    r"""sample_pdf function from another concurrent pytorch implementation
    by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
    """

    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=num_samples)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples]).to(weights)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(weights)

    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()
    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min(cdf.shape[-1] - 1 * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


if __name__ == "__main__":

    bins = torch.rand(2, 4)
    weights = torch.rand(2, 4)
    weights.requires_grad = True
    samples = sample_pdf(bins, weights, 10)
    print(samples)
