from collections import defaultdict
from pickletools import uint8
from typing import Optional
import numpy as np
import math
import torch
from cfgnode import CfgNode
import yaml
import cv2
import torchvision

# import torchsearchsorted

class null_with:
    def __enter__(self):
        pass
    def __exit__(self,a,b,c):
        pass

def set_config_defaults(source,target):
    for k in source.keys():
        if k not in target: setattr(target,k,getattr(source,k))

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


def compute_patches_distribution(image,patch_size):
    white_bg = is_background_white(image)
    image = integral_image(torch.any(image<246/256 if white_bg else image>10/256,-1))
    image = image[patch_size:,patch_size:]+image[:-(patch_size),:-(patch_size)]-image[patch_size:,:-(patch_size)]-image[:-(patch_size),patch_size:]
    return image.float()/image.sum()

def integral_image(image):
    return torch.cat([torch.zeros([1+image.shape[0],1]).type(image.type()),torch.cat([torch.zeros([1,image.shape[1]]).type(image.type()),torch.cumsum(torch.cumsum(image,0),1)],0)],1)

def img2mse(img_src, img_tgt):
    return torch.nn.functional.mse_loss(img_src, img_tgt)

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

def arange_ims(ims_tensor_list,text,psnrs=[],fontScale=1):
    # num_cols = int(np.ceil(np.sqrt(len(ims_tensor_list))))
    num_rows = lambda n_cols:   int(np.ceil(len(ims_tensor_list)/n_cols))
    num_cols = 1 #len(ims_tensor_list)
    while num_cols*ims_tensor_list[0].shape[1]<num_rows(num_cols)*ims_tensor_list[0].shape[0]:
        if num_cols==len(ims_tensor_list): break
        num_cols += 1
    assert all([t.shape[:2]==ims_tensor_list[0].shape[:2] for t in ims_tensor_list[1:]]),'Currently assuming all images have the same dimensions.'
    # im_sides = [np.array(t.shape[:2]) for t in ims_tensor_list]
    # max_side = max(im_sides)
    ims_size = tuple(ims_tensor_list[0].shape[:2])
    rows = []
    psnrs += (len(ims_tensor_list)-len(psnrs))*[None]
    for im_num,im in enumerate(ims_tensor_list):
        if im_num%num_cols==0:
            if im_num>0:    rows.append(np.concatenate(row,1))
            row = []
        row.append(cv2.resize(
            cast_to_image(im,img_text=text if im_num==0 else None,psnr=psnrs[im_num],fontScale=fontScale).transpose(1,2,0),
            dsize=(ims_size[1],ims_size[0]),#(max_side,max_side),
            interpolation=cv2.INTER_CUBIC))
    row = np.concatenate(row,1)
    rows.append(np.pad(row,((0,0),(0,num_cols*ims_size[1]-row.shape[1]),(0,0)),))
    return np.concatenate(rows,0).transpose(2,0,1)

def cast_to_image(tensor,img_text=None,psnr=None,fontScale=1):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    if img_text is not None:
        text_color = ((np.mean(img[:,:int(fontScale*img.shape[1]//24),:int(fontScale*15)],axis=(1,2)).astype(np.uint8)+[128,128,128])%256).tolist()
        img = cv2.cv2.putText(
            img.transpose(1,2,0),
            img_text,
            (0,int(fontScale*15)),
            # (img.shape[1]-20*len(img_text),50),
            cv2.FONT_HERSHEY_PLAIN,
            fontScale=fontScale,
            color=text_color,#[255,255,255],
            thickness=int(np.ceil(np.sqrt(fontScale))),
        ).transpose(2,0,1)
    if psnr is not None:
        psnr_color = ((np.mean(img[:,-int(fontScale*img.shape[1]//24):,img.shape[2]//2:],axis=(1,2)).astype(np.uint8)+[128,128,128])%256).tolist()
        img = cv2.cv2.putText(
            img.transpose(1,2,0),
            '%.2f'%(psnr),
            (img.shape[2]//2,img.shape[1]),
            # (img.shape[1]-20*len(img_text),50),
            cv2.FONT_HERSHEY_PLAIN,
            fontScale=fontScale,
            # color=[255,255,255],
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

def get_focal(data,dim):
    assert dim in ['H','W']
    if isinstance(data,list):
        # return data[0] if dim=='H' else data[1]
        return data[1] if dim=='H' else data[0]
    else:
        return data


def calc_scene_box(scene_geometry,including_dirs):
    FULL_AZ_RANGE = True # Manually set azimuth range to be [-pi,pi]
    FULL_EL_RANGE = [-np.pi/2,np.pi/2] #None
    num_frames = len(scene_geometry['camera_poses'])
    box = [[np.finfo(np.float).max,np.finfo(np.float).min] for i in range(3+2*including_dirs)]
    for f_num in range(num_frames):
        origin = scene_geometry['camera_poses'][f_num][:3, -1]
        for W in [0,scene_geometry['W'][f_num]-1]:
            for H in [0,scene_geometry['H'][f_num]-1]:
                coord = np.array([\
                    (W-scene_geometry['W'][f_num]/2)/get_focal(scene_geometry['f'][f_num],'W'),
                    -(H-scene_geometry['H'][f_num]/2)/get_focal(scene_geometry['f'][f_num],'H'),
                    -1
                ])
        # for corner in [np.array([i,j,1]) for i in [-1,1] for j in [-1,1]]:
                dir = np.sum(coord * scene_geometry['camera_poses'][f_num][:3, :3], axis=-1)
                for dist in [scene_geometry['near'],scene_geometry['far']]:
                    point = origin+dist*dir
                    for d in range(3):
                        box[d][0] = min(box[d][0],point[d])
                        box[d][1] = max(box[d][1],point[d])
                if including_dirs and not (FULL_AZ_RANGE and FULL_EL_RANGE):
                    az_el = cart2az_el(torch.tensor(dir)).numpy()
                    for d in range(int(FULL_AZ_RANGE),2):
                        box[3+d][0] = min(box[3+d][0],az_el[d])
                        box[3+d][1] = max(box[3+d][1],az_el[d])
    if including_dirs:
        if FULL_AZ_RANGE:   box[3] = [-np.pi,np.pi]
        if FULL_EL_RANGE:   box[4] = [-np.pi/2,np.pi/2]
    return torch.from_numpy(np.array(box).transpose(1,0))

def cart2az_el(dirs):
    el = torch.atan2(dirs[...,2],torch.sqrt(torch.sum(dirs[...,:2]**2,-1)))
    az = torch.atan2(dirs[...,1],dirs[...,0])
    return torch.stack([az,el],-1)

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

    # # meshgrid_xy
    # i, j = np.meshgrid(np.arange(3), np.arange(4, 7), indexing='xy')
    # print(i)
    # print(j)
    # ii, jj = torch.meshgrid(torch.arange(3), torch.arange(4, 7))
    # print(ii.transpose(-1, -2))
    # print(jj.transpose(-1, -2))
    # ii, jj = meshgrid_xy(torch.arange(3), torch.arange(4, 7))
    # print(ii)
    # print(jj)

    # # dirs (get_rays_np)
    # H, W = 3, 3
    # focal = 10
    # i, j = np.meshgrid(np.arange(3), np.arange(4, 7), indexing='xy')
    # dirs = np.stack([(i - W) * .5 / focal, -(j - H) * .5 / focal, -np.ones_like(i)], -1)
    # print(dirs)
    # ii, jj = meshgrid_xy(torch.arange(3).float(), torch.arange(4, 7).float())
    # dirs_torch = torch.stack([(ii - W) * .5 / focal, -(jj - H) * .5 / focal, -torch.ones_like(ii)], -1)
    # print(dirs_torch)
    # print(np.allclose(dirs, dirs_torch.cpu().numpy()))

    # # rays_o, rays_d (get_rays_np)
    # H, W = 3, 3
    # focal = 10
    # c2w = np.eye(4)
    # c2w[:3, :3] = 2 * c2w[:3, :3]
    # i, j = np.meshgrid(np.arange(3), np.arange(4, 7), indexing='xy')
    # dirs = np.stack([(i - W) * .5 / focal, -(j - H) * .5 / focal, -np.ones_like(i)], -1)
    # rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    # print(rays_d)
    # print(rays_o)
    # ii, jj = meshgrid_xy(torch.arange(3).float(), torch.arange(4, 7).float())
    # dirs_torch = torch.stack([(ii - W) * .5 / focal, -(jj - H) * .5 / focal, -torch.ones_like(ii)], -1)
    # c2w_torch = torch.eye(4)
    # c2w_torch[:3, :3] = 2 * c2w_torch[:3, :3]
    # rays_d_torch = torch.sum(dirs_torch[..., None, :] * c2w_torch[:3, :3], -1)
    # rays_o_torch = c2w_torch[:3, -1].expand(rays_d_torch.shape)
    # print(rays_d_torch)
    # print(rays_o_torch)
    # print(np.allclose(rays_d, rays_d_torch.cpu().numpy()))
    # print(np.allclose(rays_o, rays_o_torch.cpu().numpy()))

    # # get_rays(_torch) vs get_rays_np
    # H, W = 3, 3
    # focal = 10
    # c2w = np.eye(4)
    # c2w[:3, :3] = 2 * c2w[:3, :3]
    # # c2w_torch = torch.eye(4)
    # # c2w_torch[:3, :3] = 2 * c2w_torch[:3, :3]
    # rays_o, rays_d = get_rays_np(H, W, focal, c2w)
    # c2w_torch = torch.from_numpy(c2w)
    # rays_o_torch, rays_d_torch = get_rays(H, W, focal, c2w_torch)
    # print(np.allclose(rays_o, rays_o_torch.cpu().numpy()))
    # print(np.allclose(rays_d, rays_d_torch.cpu().numpy()))  # Assert fails, values look different.
    # print("Numpy version:")
    # print(rays_d)
    # print("PyTorch version:")
    # print(rays_d_torch.cpu().numpy())

    # Test backprop for sample_pdf()
    bins = torch.rand(2, 4)
    weights = torch.rand(2, 4)
    weights.requires_grad = True
    samples = sample_pdf(bins, weights, 10)
    print(samples)
