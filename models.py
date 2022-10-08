from collections import defaultdict,OrderedDict
import torch
import torch.nn as nn
from nerf_helpers import cart2az_el
import math
import numpy as np
from scipy.interpolate import griddata
from re import search
import os
from tqdm import tqdm
from shutil import copyfile
class VeryTinyNeRFModel(nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """
    def __init__(self, filter_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(VeryTinyNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 65 -> 128)
        self.layer1 = nn.Linear(
            self.xyz_encoding_dims + self.viewdir_encoding_dims, filter_size
        )
        # Layer 2 (default: 128 -> 128)
        self.layer2 = nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = nn.Linear(filter_size, 4)
        # Short hand for nn.functional.relu
        self.relu = nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiHeadNeRFModel(nn.Module):
    r"""Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 39 -> 128)
        self.layer1 = nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 1): Predicts a feature vector (used for color)
        self.layer3_2 = nn.Linear(hidden_size, hidden_size)

        # Layer 4 (default: 39 + 128 -> 128)
        self.layer4 = nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        # Layer 5 (default: 128 -> 128)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 3): Predicts RGB color
        self.layer6 = nn.Linear(hidden_size, 3)

        # Short hand for nn.functional.relu
        self.relu = nn.functional.relu

    def forward(self, x):
        x, view = x[..., : self.xyz_encoding_dims], x[..., self.xyz_encoding_dims :]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)


class ReplicateNeRFModel(nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
    ):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions

        self.dim_xyz = (3 if include_input_xyz else 0) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0) + 2 * 3 * num_encoding_fn_dir

        self.layer1 = nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = nn.Linear(hidden_size, 1)

        self.layer4 = nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = nn.Linear(hidden_size // 2, 3)
        self.relu = nn.functional.relu

    def forward(self, x):
        xyz, direction = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        return torch.cat((rgb, alpha), dim=-1)

class Conv3D(nn.Module):
    def __init__(
        self,
        num_layers=4,
        num_layers_dir=1,
        dirs_hidden_width_ratio=2,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        input_dim=None,
        xyz_input_2_dir=False,
        kernel_size=3,
    ):
        super(Conv3D, self).__init__()
        if not isinstance(hidden_size,list):
            hidden_size = [hidden_size]
        layer_size = lambda x: hidden_size[min([x,len(hidden_size)-1])]
        self.skip_connect_every = skip_connect_every
        self.receptive_field = 1
        if input_dim is not None:
            self.dim_xyz = input_dim[0]
            if use_viewdirs:
                self.dim_dir = input_dim[1]
            else:
                self.dim_xyz = sum(input_dim)
        else:
            include_input_xyz = 3 if include_input_xyz else 0
            include_input_dir = 3 if include_input_dir else 0
            self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
            self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
            if not use_viewdirs:
                self.dim_dir = 0
        self.layer1 = nn.Conv3d(self.dim_xyz, layer_size(0),kernel_size=kernel_size)
        self.receptive_field += kernel_size-1
        self.layers_xyz = nn.ModuleList()
        for i in range(num_layers - 1):
            self.receptive_field += kernel_size-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    nn.Conv3d(self.dim_xyz + layer_size(i), layer_size(i+1),kernel_size=kernel_size)
                )
            else:
                self.layers_xyz.append(nn.Conv3d(layer_size(i), layer_size(i+1),kernel_size=kernel_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.xyz_input_2_dir = xyz_input_2_dir
            self.layers_dir = nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                nn.Conv3d(self.dim_dir + hidden_size[-1]+(self.dim_xyz if xyz_input_2_dir else 0),\
                    hidden_size[-1] // dirs_hidden_width_ratio,kernel_size=kernel_size)
            )
            self.receptive_field += kernel_size-1
            for i in range(num_layers_dir-1):
                self.receptive_field += kernel_size-1
                self.layers_dir.append(
                    nn.Conv3d(hidden_size[-1]//dirs_hidden_width_ratio, hidden_size[-1] // dirs_hidden_width_ratio,kernel_size=kernel_size)
                )

            self.fc_alpha = nn.Conv3d(hidden_size[-1], 1,kernel_size=kernel_size)
            self.receptive_field += kernel_size-1
            self.fc_rgb = nn.Conv3d(hidden_size[-1] // dirs_hidden_width_ratio, 3,kernel_size=kernel_size)
            self.receptive_field += kernel_size-1
            self.fc_feat = nn.Conv3d(hidden_size[-1], hidden_size[-1],kernel_size=kernel_size)
        else:
            self.receptive_field += kernel_size-1
            self.fc_out = nn.Conv3d(hidden_size[-1], 4,kernel_size=kernel_size)

        self.relu = nn.functional.relu

    def forward(self, x):
        if self.use_viewdirs:
            xyz, view = x[:, : self.dim_xyz,...], x[:, self.dim_xyz :,...]
        else:
            xyz = x[:, : self.dim_xyz,...]
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz)
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        def different_shape_concat(x,y):
            shape_crop = (y.shape[2]-x.shape[2])//2
            return torch.cat((x,y[...,shape_crop:-shape_crop,shape_crop:-shape_crop,shape_crop:-shape_crop]),1)

        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            # x = torch.cat((feat, view[...,shape_crop:-shape_crop,shape_crop:-shape_crop,shape_crop:-shape_crop]), dim=1)
            x = different_shape_concat(feat,view)
            if self.xyz_input_2_dir:
                x = different_shape_concat(x,xyz)
                # x = torch.cat((xyz,x),dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            # return torch.cat((rgb, alpha), dim=-1)
            return different_shape_concat(rgb,alpha)
        else:
            return self.fc_out(x)


class FlexibleNeRFModel(nn.Module):
    def __init__(
        self,
        num_layers=4,
        num_layers_dir=1,
        dirs_hidden_width_ratio=2,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        input_dim=None,
        xyz_input_2_dir=False,
    ):
        super(FlexibleNeRFModel, self).__init__()
        if isinstance(hidden_size,list):
            assert not use_viewdirs,"Unsupported yet"
        else:
            hidden_size = [hidden_size]
        layer_size = lambda x: hidden_size[min([x,len(hidden_size)-1])]
        self.skip_connect_every = skip_connect_every
        self.receptive_field = 0
        if input_dim is not None:
            self.dim_xyz = input_dim[0]
            if use_viewdirs:
                self.dim_dir = input_dim[1]
            else:
                self.dim_xyz = sum(input_dim)
        else:
            include_input_xyz = 3 if include_input_xyz else 0
            include_input_dir = 3 if include_input_dir else 0
            self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
            self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
            if not use_viewdirs:
                self.dim_dir = 0
        self.layer1 = nn.Linear(self.dim_xyz, layer_size(0))
        self.layers_xyz = nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    nn.Linear(self.dim_xyz + layer_size(i), layer_size(i+1))
                    # nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                # self.layers_xyz.append(nn.Linear(hidden_size, hidden_size))
                self.layers_xyz.append(nn.Linear(layer_size(i), layer_size(i+1)))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.xyz_input_2_dir = xyz_input_2_dir
            self.layers_dir = nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                nn.Linear(self.dim_dir + hidden_size[-1]+(self.dim_xyz if xyz_input_2_dir else 0),\
                    hidden_size[-1] // dirs_hidden_width_ratio)
            )
            for i in range(num_layers_dir-1):
                self.layers_dir.append(
                    nn.Linear(hidden_size[-1]//dirs_hidden_width_ratio, hidden_size[-1] // dirs_hidden_width_ratio)
                )

            self.fc_alpha = nn.Linear(hidden_size[-1], 1)
            self.fc_rgb = nn.Linear(hidden_size[-1] // dirs_hidden_width_ratio, 3)
            self.fc_feat = nn.Linear(hidden_size[-1], hidden_size[-1])
        else:
            self.fc_out = nn.Linear(hidden_size[-1], 4)

        self.relu = nn.functional.relu

    def forward(self, x):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz)
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            if self.xyz_input_2_dir:
                x = torch.cat((xyz,x),dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)

def get_plane_name(scene_id,dimension):
    if scene_id is None:
        return "_D%d"%(dimension)    
    return "sc%s_D%d"%(scene_id,dimension)

# def decode_plane_name(name):
#     scene_id = search('(?<=sc)(\w)+(?=_D)',name).group(0)
#     dim = int(search('(?<=_D)(\d)+(?=$)',name).group(0))
#     return scene_id,dim

class TwoDimPlanesModel(nn.Module):
    def __init__(
        self,
        use_viewdirs,
        scene_id_plane_resolution,
        coords_normalization,
        dec_density_layers=4,
        dec_rgb_layers=4,
        dec_channels=128,
        skip_connect_every=None,
        num_plane_channels=48,
        rgb_dec_input='projections',
        proj_combination='sum',
        planes=None,
        plane_interp='bilinear',
        align_corners=True,
        interp_viewdirs=None,
        viewdir_downsampling=True,
        viewdir_proj_combination=None,
    ):
        self.N_PLANES_DENSITY = 3
        self.PLANES_2_INFER = [self.N_PLANES_DENSITY]

        super(TwoDimPlanesModel, self).__init__()
        self.box_coords = coords_normalization
        self.use_viewdirs = use_viewdirs
        self.num_plane_channels = num_plane_channels
        assert interp_viewdirs in ['bilinear','bicubic',None]
        self.interp_from_learned = interp_viewdirs
        self.viewdir_downsampling = viewdir_downsampling
        self.align_corners = align_corners
        assert rgb_dec_input in ['projections','features','projections_features']
        self.rgb_dec_input = rgb_dec_input
        assert proj_combination in ['sum','concat']
        assert use_viewdirs or viewdir_proj_combination is None
        if viewdir_proj_combination is None:    viewdir_proj_combination = proj_combination
        assert viewdir_proj_combination in ['sum','concat','mult']
        self.proj_combination = proj_combination
        self.viewdir_proj_combination = viewdir_proj_combination
        self.plane_interp = plane_interp
        self.planes_ds_factor = 1
        self.skip_connect_every = skip_connect_every # if skip_connect_every is not None else max(dec_rgb_layers,dec_density_layers)

        # Density (alpha) decoder:
        self.density_dec = nn.ModuleList()
        self.debug = {'max_norm':defaultdict(lambda: torch.finfo(torch.float32).min),'min_norm':defaultdict(lambda: torch.finfo(torch.float32).max)}
        in_channels = num_plane_channels*(self.N_PLANES_DENSITY if proj_combination=='concat' else 1)
        self.density_dec.append(nn.Linear(in_channels,dec_channels))
        for layer_num in range(dec_density_layers-1):
            if self.is_skip_layer(layer_num=layer_num):
            # if layer_num % self.skip_connect_every == 0 and layer_num > 0: # and layer_num != dec_density_layers-1:
                self.density_dec.append(nn.Linear(in_channels + dec_channels, dec_channels))
            else:
                self.density_dec.append(nn.Linear(dec_channels,dec_channels))
        self.fc_alpha = nn.Linear(dec_channels,1)
        if 'features' in self.rgb_dec_input:
            self.fc_feat = nn.Linear(dec_channels,num_plane_channels)

        # RGB decoder:
        self.rgb_dec = nn.ModuleList()
        plane_C_mult = 1
        if proj_combination=='concat':  plane_C_mult += self.N_PLANES_DENSITY-1
        if viewdir_proj_combination=='concat':  plane_C_mult +=1

        self.rgb_dec.append(nn.Linear(num_plane_channels*plane_C_mult,dec_channels))
        for layer_num in range(dec_rgb_layers-1):
            if self.is_skip_layer(layer_num=layer_num):
            # if layer_num % self.skip_connect_every == 0 and layer_num > 0: # and layer_num != dec_rgb_layers-1:
                self.rgb_dec.append(nn.Linear(num_plane_channels*plane_C_mult + dec_channels, dec_channels))
            else:
                self.rgb_dec.append(nn.Linear(dec_channels,dec_channels))
        self.fc_rgb = nn.Linear(dec_channels,3)

        self.relu = nn.functional.relu

        if scene_id_plane_resolution is not None: #If not using the store_panes configuration
            if planes is None:
                self.planes_ = nn.ParameterDict([
                    (get_plane_name(id,d),
                        create_plane(res[0] if d<self.N_PLANES_DENSITY else res[1],num_plane_channels=num_plane_channels,init_STD=0.1*self.fc_alpha.weight.data.std())
                    )
                    for id,res in scene_id_plane_resolution.items() for d in range(self.N_PLANES_DENSITY+use_viewdirs)])
                if self.interp_from_learned:
                    self.copy_planes()
                    assert not align_corners,'The following corresponding grid assumes -1 and 1 correspond to array corners (rather than the center of its corner pixels)'
                    self.corresponding_grid = OrderedDict()
                    for k,v in self.planes_copy.items():
                        res = list(v.shape[2:])
                        # Dimensions of self.corresponding_grid[k] are resolution X resolution X 2, where the last dimension corresponds to indecis [x,y] (column,row):
                        self.corresponding_grid[k] = np.stack(np.meshgrid(np.linspace(-1+1/res[1],1-1/res[1],res[1]),np.linspace(-1+1/res[0],1-1/res[0],res[0])),-1)
            else:
                self.planes_ = planes

    def planes2cpu(self):
        for p in self.planes_.values():
            p.data = p.data.to("cpu")

    def is_skip_layer(self,layer_num):
        if self.skip_connect_every is None:
            return False
        else:
            return layer_num % self.skip_connect_every == 0 and layer_num > 0 # and layer_num != dec_rgb_layers-1:

    def copy_planes(self):
        self.planes_copy = OrderedDict([(k,1*v.detach().cpu().numpy()) for k,v in self.planes_.items() if any([get_plane_name(None,d) in k for d in self.PLANES_2_INFER])])

    def eval(self):
        super(TwoDimPlanesModel, self).eval()
        self.use_downsampled_planes(1)
        if self.interp_from_learned and hasattr(self,'planes_copy'):
            self.learned = OrderedDict([(k,np.all((self.planes_[k].detach().cpu().numpy()-self.planes_copy[k]!=0),1).squeeze()) for k in self.planes_copy.keys()])
            for k in self.learned.keys():
                if np.any(self.learned[k]): self.interpolate_plane(k)

    def interpolate_plane(self,k):
        interpolated = griddata(\
            points=self.corresponding_grid[k][self.learned[k]],
            values=self.planes_[k].squeeze(0).detach().cpu().numpy().transpose(1,2,0)[self.learned[k]],
            xi=self.corresponding_grid[k],
            method=self.interp_from_learned[2:],
            # fill_value=0.,
        )
        valid_indecis = np.logical_not(np.any(np.isnan(interpolated),-1))
        # For debug:
        if False:
            import matplotlib.pyplot as plt
            learned_plane0 = self.planes_[k].squeeze(0).detach().cpu().numpy().transpose(1,2,0)[...,0]*self.learned[k]
            plt.imsave('learned.png',learned_plane0)
            interpolated0 = np.zeros_like(learned_plane0)
            interpolated0[valid_indecis] = interpolated[valid_indecis][:,0]
            plt.imsave('interpolated.png',interpolated0)
            interpolated_lin = griddata(\
                points=self.corresponding_grid[k][self.learned[k]],
                values=self.planes_[k].squeeze(0).detach().cpu().numpy().transpose(1,2,0)[self.learned[k]],
                xi=self.corresponding_grid[k],
                method='linear',
                )
            interpolated0_lin = np.zeros_like(learned_plane0)
            valid_lin = np.logical_not(np.any(np.isnan(interpolated),-1))
            interpolated0_lin[valid_lin] = interpolated_lin[valid_lin][:,0]
            plt.imsave('interpolated_lin.png',interpolated0_lin)
            interpolated_nearest = griddata(\
                points=self.corresponding_grid[k][self.learned[k]],
                values=self.planes_[k].squeeze(0).detach().cpu().numpy().transpose(1,2,0)[self.learned[k]],
                xi=self.corresponding_grid[k],
                method='nearest',
                )
            interpolated0_nearest = np.zeros_like(learned_plane0)
            valid_nearest = np.logical_not(np.any(np.isnan(interpolated),-1))
            interpolated0_nearest[valid_nearest] = interpolated_nearest[valid_nearest][:,0]
            plt.imsave('interpolated_nearest.png',interpolated0_nearest)
            plane0 = self.planes_[k].squeeze(0).detach().cpu().numpy().transpose(1,2,0)[...,0]
            plt.imsave('plane0.png',plane0)
        print('%s: Interpolating %.2f of values based on %.2f of them'%(k,np.logical_and(np.logical_not(self.learned[k]),valid_indecis).mean(),self.learned[k].mean()))
        self.planes_[k].data[:,:,valid_indecis] = torch.from_numpy(interpolated[valid_indecis].transpose()[None,...]).type(self.planes_[k].data.type())
        if False:
            plane0 = self.planes_[k].squeeze(0).detach().cpu().numpy().transpose(1,2,0)[...,0]
            plt.imsave('plane0_after.png',plane0)

        self.planes_copy[k][...,np.logical_not(self.learned[k])] = self.planes_[k].detach().cpu().numpy()[...,np.logical_not(self.learned[k])]
        
    def assign_SR_model(self,SR_model,SR_viewdir,save_interpolated=True,set_planes=True):
        self.SR_model = SR_model
        self.SR_model.align_corners = self.align_corners
        self.SR_model.SR_viewdir = SR_viewdir
        self.skip_SR_ = False
        if set_planes:
            for k,v in self.planes_.items():
                if not SR_viewdir and get_plane_name(None,self.N_PLANES_DENSITY) in k:  continue
                self.SR_model.set_LR_planes(v.detach(),id=k,save_interpolated=save_interpolated)
        # else:
        #     self.SR_model.LR_planes = None

    def set_cur_scene_id(self,scene_id):
        self.cur_id = scene_id

    def normalize_coords(self,coords):
        EPSILON = 1e-5
        normalized_coords = 2*(coords-self.box_coords[self.cur_id].type(coords.type())[:1])/\
            (self.box_coords[self.cur_id][1:]-self.box_coords[self.cur_id][:1]).type(coords.type())-1
        # assert normalized_coords.min()>=-1-EPSILON and normalized_coords.max()<=1+EPSILON,"Sanity check"
        self.debug['max_norm'][self.cur_id] = np.maximum(self.debug['max_norm'][self.cur_id],normalized_coords.max(0)[0].cpu().numpy())
        self.debug['min_norm'][self.cur_id] = np.minimum(self.debug['min_norm'][self.cur_id],normalized_coords.min(0)[0].cpu().numpy())
        return normalized_coords

    def use_downsampled_planes(self,ds_factor:int): # USed for debug
        self.planes_ds_factor = ds_factor

    def planes(self,dim_num:int,grid:torch.tensor=None)->torch.tensor:
        plane_name = get_plane_name(self.cur_id,dim_num)
        if hasattr(self,'SR_model') and not self.skip_SR_ and plane_name in self.SR_model.LR_planes:
            if grid is not None and self.SR_model.training:
                roi = torch.stack([grid.min(1)[0].squeeze(),grid.max(1)[0].squeeze()],0)
                roi = torch.stack([roi[:,1],roi[:,0]],1) # Converting from (x,y) to (y,x) on the columns dimension
                plane_name = (plane_name,roi)
            plane = self.SR_model(plane_name)
        else:
            if self.planes_ds_factor>1 and (self.viewdir_downsampling or dim_num<self.N_PLANES_DENSITY): # Used for debug or enforcing loss
                plane = nn.functional.interpolate(self.planes_[plane_name],scale_factor=1/self.planes_ds_factor,
                    align_corners=self.align_corners,mode=self.plane_interp,antialias=True)
            else:
                plane = self.planes_[plane_name]
        return plane.cuda()

    def skip_SR(self,skip):
        self.skip_SR_ = skip

    def project_xyz(self,coords):
        projections = []
        for d in range(self.N_PLANES_DENSITY): # (Currently not supporting viewdir input)
            grid = coords[:,[c for c in range(3) if c!=d]].reshape([1,coords.shape[0],1,2])
            projections.append(nn.functional.grid_sample(
                    input=self.planes(d,grid),
                    grid=grid,
                    mode=self.plane_interp,
                    align_corners=self.align_corners,
                    padding_mode='border',
                ))
        projections = self.combine_pos_planes(projections)
        return projections.squeeze(0).squeeze(-1).permute(1,0)

    def project_viewdir(self,dirs):
        grid = dirs.reshape([1,dirs.shape[0],1,2])
        # self.update_planes_coverage(self.N_PLANES_DENSITY,grid)
        return nn.functional.grid_sample(
                input=self.planes(self.N_PLANES_DENSITY,grid),
                grid=grid,
                mode=self.plane_interp,
                align_corners=self.align_corners,
                padding_mode='border',
            ).squeeze(0).squeeze(-1).permute(1,0)

    def combine_pos_planes(self,tensors):
        return torch.stack(tensors,0).sum(0) if self.proj_combination=='sum' else torch.cat(tensors,1)

    def combine_all_planes(self,pos_planes,viewdir_planes):
        pos_planes_shape = pos_planes.shape
        if self.viewdir_proj_combination!='concat' and pos_planes_shape[1]>viewdir_planes.shape[1]:
            pos_planes = pos_planes.reshape([pos_planes_shape[0],viewdir_planes.shape[1],-1])
            viewdir_planes = viewdir_planes.unsqueeze(-1)
        if self.viewdir_proj_combination=='sum':
            return torch.reshape(pos_planes+viewdir_planes,pos_planes_shape)
        elif self.viewdir_proj_combination=='mult':
            return torch.reshape(pos_planes*(1+viewdir_planes),pos_planes_shape)
        elif self.viewdir_proj_combination=='concat':
            return torch.cat([pos_planes,viewdir_planes],1)

    def forward(self, x):
        if False:
            change_maps = [torch.all((self.planes_[k]-self.planes_copy[k].cuda()!=0),1).squeeze(0) for k in self.planes_.keys()]
            import matplotlib.pyplot as plt
            for i,m in enumerate(change_maps):
                plt.imsave('Changed%d.png'%(i),m.cpu(),cmap='gray')
        if self.use_viewdirs:
            x = torch.cat([x[...,:3],cart2az_el(x[...,3:])],-1)
        else:
            x = x[..., : 3]
        x = self.normalize_coords(x)
        projected_xyz = self.project_xyz(x[..., : 3])
        if self.use_viewdirs:
            projected_views = self.project_viewdir(x[...,3:])

        # Projecting and summing
        x = 1*projected_xyz
        for layer_num,l in enumerate(self.density_dec):
            if self.is_skip_layer(layer_num=layer_num-1):
                x = torch.cat((x, projected_xyz), dim=-1)
            x = self.relu(l(x))
        alpha = self.fc_alpha(x)

        if 'features' in self.rgb_dec_input:
            x_rgb = self.fc_feat(x)

        if self.rgb_dec_input=='projections_features':
            x_rgb = self.combine_pos_planes([x_rgb,projected_xyz])
        elif self.rgb_dec_input=='projections':
            x_rgb = 1*projected_xyz

        if self.use_viewdirs:
            x_rgb = self.combine_all_planes(pos_planes=x_rgb,viewdir_planes=projected_views)

        if False and self.skip_connect_every is None:
            for layer_num,l in enumerate(self.rgb_dec):
                x_rgb = self.relu(l(x_rgb))
            rgb = self.fc_rgb(x_rgb)
        else:
            x = x_rgb
            for layer_num,l in enumerate(self.rgb_dec):
                if self.is_skip_layer(layer_num=layer_num-1):
                    x = torch.cat((x, x_rgb), dim=-1)
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)

        return torch.cat((rgb, alpha), dim=-1)

def create_plane(resolution,num_plane_channels,init_STD):
    # if init_STD is None:
    #     init_STD = 0.1*self.fc_alpha.weight.data.std()
    return nn.Parameter(init_STD*torch.randn(size=[1,num_plane_channels,resolution,resolution]))

class SceneSampler:
    def __init__(self,scenes:list,do_when_reshuffling=lambda:None) -> None:
        self.scenes = scenes
        self.do_when_reshuffling = lambda:None
        self.shuffle()
        self.do_when_reshuffling = do_when_reshuffling

    def shuffle(self):
        self.sample_from = [self.scenes[i] for i in np.random.permutation(len(self.scenes))]
        self.do_when_reshuffling()

    def sample(self,n):
        assert n<=len(self.scenes)
        sampled = []
        cursor = 0
        while len(sampled)<n:
            if len(self.sample_from)==0:
                self.shuffle()
                cursor = 0
            if self.sample_from[cursor] not in sampled:
                sampled.append(self.sample_from.pop(cursor))
            else:
                cursor += 1
        return sampled

class PlanesOptimizer(nn.Module):
    def __init__(self,optimizer_type:str,scene_id_plane_resolution:dict,options,save_location:str,
            lr:float,model_coarse:TwoDimPlanesModel,model_fine:TwoDimPlanesModel,use_coarse_planes:bool,
            init_params:bool,optimize:bool,training_scenes:list=None,coords_normalization:dict=None,
            do_when_reshuffling=lambda:None,STD_factor:float=0.1) -> None:
        super(PlanesOptimizer,self).__init__()
        self.scenes = list(scene_id_plane_resolution.keys())
        if training_scenes is None:
            training_scenes = 1*self.scenes
        self.training_scenes = training_scenes
        self.buffer_size = getattr(options,'buffer_size',len(self.training_scenes))
        self.steps_per_buffer,self.steps_since_drawing = options.steps_per_buffer,0
        if self.buffer_size>=len(self.training_scenes):  self.steps_per_buffer = -1
        assert all([s in self.scenes for s in self.training_scenes])
        assert optimizer_type=='Adam','Optimizer %s not supported yet.'%(optimizer_type)
        assert use_coarse_planes,'Unsupported yet, probably requires adding a param_group to the optimizer'
        assert not init_params or optimize,'This would means using (frozen) random planes...'
        assert self.steps_per_buffer==-1 or self.steps_per_buffer>=self.buffer_size,\
            'Trying to use %d steps for a buffer of size %d: Some scenes would be loaded in vain.'%(options.steps_per_buffer,self.buffer_size)
        self.scene_sampler = SceneSampler(self.training_scenes,do_when_reshuffling=do_when_reshuffling)
        self.models = {}
        self.use_coarse_planes = use_coarse_planes
        self.save_location = save_location
        self.lr = lr
        for model_name,model in zip(['coarse','fine'],[model_coarse,model_fine]):
            self.models[model_name] = model
            if model_name=='fine' and use_coarse_planes:    continue
            self.planes_per_scene = model.N_PLANES_DENSITY+model.use_viewdirs
            if init_params:
                for scene in tqdm(self.scenes,desc='Initializing scene planes',):
                    res = scene_id_plane_resolution[scene]
                    params = nn.ParameterDict([
                        (get_plane_name(scene,d),
                            create_plane(res[0] if d<model.N_PLANES_DENSITY else res[1],num_plane_channels=model.num_plane_channels,
                            init_STD=STD_factor*model.fc_alpha.weight.data.std().cpu())
                        )
                        for d in range(self.planes_per_scene)])
                    torch.save({'params':params,'coords_normalization':coords_normalization[scene]},self.param_path(model_name=model_name,scene=scene))
        self.optimize = optimize
        self.optimizer = None
        self.saving_needed = False
        # self.draw_scenes()

    def load_scene(self,scene):
        if self.saving_needed:
            self.save_params()
        for model_name in ['coarse','fine']:
            if model_name=='coarse' or not self.use_coarse_planes:
                loaded_params = torch.load(self.param_path(model_name=model_name,scene=scene))
            self.models[model_name].planes_ = loaded_params['params']
            self.models[model_name].box_coords = {scene:loaded_params['coords_normalization']}
            if hasattr(self.models[model_name],'SR_model'):
                self.models[model_name].SR_model.clear_SR_planes(all_planes=True)
                for k,v in loaded_params['params'].items():
                    if not self.models[model_name].SR_model.SR_viewdir and get_plane_name(None,self.models[model_name].N_PLANES_DENSITY) in k:  continue
                    self.models[model_name].SR_model.set_LR_planes(v.detach(),id=k,save_interpolated=False)
        self.cur_scenes = [scene]

    def load_from_checkpoint(self,checkpoint):
        model_name = 'coarse'
        param2num = dict([(k,i) for i,k in enumerate(checkpoint['plane_parameters'])])
        for scene in self.scenes:
            params = nn.ParameterDict([(get_plane_name(scene,d),checkpoint['plane_parameters'][get_plane_name(scene,d)]) for d in range(self.planes_per_scene)])
            opt_states = [checkpoint['plane_optimizer_states'][param2num[get_plane_name(scene,d)]] for d in range(self.planes_per_scene)]
            torch.save({'params':params,'opt_states':opt_states},self.param_path(model_name=model_name,scene=scene))

    def param_path(self,model_name,scene):
        return os.path.join(self.save_location,"%s_%s.par"%(model_name,scene))

    def get_plane_stats(self,viewdir=False):
        model_name='coarse'
        plane_means,plane_STDs = [],[]
        for scene in tqdm(self.training_scenes,desc='Collecting plane statistics'):
            loaded_params = torch.load(self.param_path(model_name=model_name,scene=scene))
            for k,p in loaded_params['params'].items():
                if not viewdir and get_plane_name(None,self.models[model_name].N_PLANES_DENSITY) in k:  continue
                plane_means.append(torch.mean(p,(2,3)).squeeze(0))
                plane_STDs.append(torch.std(p.reshape(p.shape[1],-1),1))
        return {'mean':torch.stack(plane_means,0).mean(0),'std':torch.stack(plane_STDs,0).mean(0)}

    def save_params(self,to_checkpoint=False):
        assert self.optimize,'Why would you want to save if not optimizing?'
        model_name = 'coarse'
        model = self.models[model_name]
        scenes_list = self.scenes if to_checkpoint else self.cur_scenes
        if to_checkpoint:
            all_params,all_states = nn.ParameterDict(),[]
        scene_num = 0
        for scene in scenes_list:
            if scene in self.cur_scenes:
                params = nn.ParameterDict([(get_plane_name(scene,d),model.planes_[get_plane_name(scene,d)]) for d in range(self.planes_per_scene)])
                opt_states = [self.optimizer.state_dict()['state'][i+scene_num*self.planes_per_scene] if (i+scene_num*self.planes_per_scene) in self.optimizer.state_dict()['state'] else None for i in range(self.planes_per_scene)]
                scene_num += 1
            else:
                loaded_params = torch.load(self.param_path(model_name=model_name,scene=scene))
                params = loaded_params['params']
                opt_states = loaded_params['opt_states'] if 'opt_states' in loaded_params else [None for p in params]
            if to_checkpoint:
                all_params.update(params)
                all_states.extend(opt_states)
            else:
                param_file_name = self.param_path(model_name=model_name,scene=scene)
                del_temp = False
                if os.path.isfile(param_file_name):
                    del_temp = True
                    copyfile(param_file_name,param_file_name.replace('.par','.par_temp'))
                torch.save({'params':params,'opt_states':opt_states,'coords_normalization':model.box_coords[scene]},param_file_name)
                if del_temp:
                    os.remove(param_file_name.replace('.par','.par_temp'))

        if to_checkpoint:
            return all_params,all_states
        else:
            self.saving_needed = False

    def draw_scenes(self):
        if self.saving_needed:
            self.save_params()
        self.steps_since_drawing = 0
        self.cur_scenes = self.scene_sampler.sample(self.buffer_size)
        for model_name in ['coarse','fine']:
            model = self.models[model_name]
            if model_name=='coarse' or not self.use_coarse_planes:
                params_dict,optimizer_states,box_coords = nn.ParameterDict(),[],{}
                for scene in self.cur_scenes:
                    loaded_params = torch.load(self.param_path(model_name=model_name,scene=scene))
                    params_dict.update(loaded_params['params'])
                    box_coords.update({scene:loaded_params['coords_normalization']})
                    if self.optimize:
                        if 'opt_states' in loaded_params:
                            optimizer_states.extend(loaded_params['opt_states'])
                        else:
                            optimizer_states.extend([None for p in loaded_params['params']])
            model.planes_ = params_dict.cuda()
            self.models[model_name].box_coords = box_coords
            if hasattr(model,'SR_model'):
                model.SR_model.clear_SR_planes(all_planes=True)
                for k,v in params_dict.items():
                    # model.SR_model.set_LR_planes(v.detach(),id=k,save_interpolated=True)
                    if not model.SR_model.SR_viewdir and get_plane_name(None,model.N_PLANES_DENSITY) in k:  continue
                    model.SR_model.set_LR_planes(v.detach(),id=k,save_interpolated=False)
            if not self.optimize:   continue
            if model_name=='coarse' or not self.use_coarse_planes:
                params = list(model.planes_.values())
                if self.optimizer is None: # First call to this function:
                    self.optimizer = torch.optim.Adam(params, lr=self.lr)
                else:
                    self.optimizer.param_groups[0]['params'] = params
                self.optimizer.state = defaultdict(dict,[(params[i],v) for i,v in enumerate(optimizer_states) if v is not None])
        self.saving_needed = False

    def step(self,opt_step=True):
        if self.optimize and opt_step:
            self.optimizer.step()
            self.saving_needed = True
        self.steps_since_drawing += 1
        if self.steps_since_drawing==self.steps_per_buffer:
            self.draw_scenes()
            return self.cur_scenes
        else:
            return None

    def zero_grad(self):
        if self.optimize:   self.optimizer.zero_grad()


# EDSR code taken and modified from https://github.com/twtygqyy/pytorch-edsr/blob/master/edsr.py
class _Residual_Block(nn.Module): 
    def __init__(self,hidden_size,padding,kernel_size):
        super(_Residual_Block, self).__init__()
        self.margins = None if padding else 2*(kernel_size//2)
        self.conv1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

    def forward(self, x): 
        if self.margins is None:
            identity_data = x
        else:
            identity_data = x[...,self.margins:-self.margins,self.margins:-self.margins]
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output 

class EDSR(nn.Module):
    def __init__(self,scale_factor,in_channels,out_channels,hidden_size,plane_interp,n_blocks=32,
        input_normalization=False,consistentcy_loss_w:float=None):
        super(EDSR, self).__init__()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        self.scale_factor = scale_factor
        self.plane_interp = plane_interp
        self.n_blocks = n_blocks
        PADDING,KERNEL_SIZE = 0,3
        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False)
        self.required_padding,rf_factor = KERNEL_SIZE//2,1

        self.residual = self.make_layer(_Residual_Block,{'hidden_size':hidden_size,'padding':PADDING,'kernel_size':KERNEL_SIZE}, n_blocks)
        self.required_padding += rf_factor*2*n_blocks*((KERNEL_SIZE-1)//2)

        self.conv_mid = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False)
        self.required_padding += rf_factor*((KERNEL_SIZE-1)//2)
        assert math.log2(scale_factor)==int(math.log2(scale_factor)),"Supperting only scale factors that are an integer power of 2."
        upscaling_layers = []
        for _ in range(int(math.log2(scale_factor))):
            upscaling_layers += [
                nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*4, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False),
                nn.PixelShuffle(2),
            ]
            self.required_padding += rf_factor*((KERNEL_SIZE-1)//2)
            rf_factor /= 2

        self.upscale = nn.Sequential(*upscaling_layers)

        self.conv_output = nn.Conv2d(in_channels=hidden_size, out_channels=out_channels, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False)
        self.required_padding += rf_factor*((KERNEL_SIZE-1)//2)
        self.HR_overpadding = int(self.required_padding*self.scale_factor)
        self.required_padding = int(np.ceil(self.required_padding))
        self.HR_overpadding = self.required_padding*self.scale_factor-self.HR_overpadding
        # self.add_mean = MeanShift(rgb_mean, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)/10)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.clear_SR_planes(all_planes=True)
        if input_normalization:
            self.normalization_params({'mean':float('nan')*torch.ones([in_channels]),'std':float('nan')*torch.ones([in_channels])})
        self.consistentcy_loss_w = consistentcy_loss_w
        if consistentcy_loss_w is not None:
            self.planes_diff = nn.L1Loss()
            self.consistentcy_loss = []

    def make_layer(self, block,args, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(**args))
        return nn.Sequential(*layers)

    def interpolate_LR(self,id):
        return torch.nn.functional.interpolate(self.LR_planes[id].cuda(),scale_factor=self.scale_factor,mode=self.plane_interp,align_corners=self.align_corners)

    def normalization_params(self,norm_dict):
        self.planes_mean_NON_LEARNED = nn.Parameter(norm_dict['mean'].reshape([1,-1,1,1]))
        self.planes_std_NON_LEARNED = nn.Parameter(norm_dict['std'].reshape([1,-1,1,1]))

    def residual_plane(self,id):
        if id in self.residual_planes:
            return self.residual_planes[id].cuda()
        else:
            return self.interpolate_LR(id)

    def set_LR_planes(self,plane,id:str,save_interpolated:bool):
        assert id not in self.LR_planes,"Plane ID already exists."
        self.LR_planes[id] = plane
        if save_interpolated:
            # self.residual_planes[id] = torch.nn.functional.interpolate(plane.cuda(),scale_factor=self.scale_factor,mode=self.plane_interp,align_corners=align_corners).cpu()
            self.residual_planes[id] = self.interpolate_LR(id).cpu()
        # self.SR_planes[id] = 1*self.residual_planes[id]

    def clear_SR_planes(self,all_planes=False):
        planes_2_clear = ['SR_planes']
        if all_planes:
            planes_2_clear += ['LR_planes','residual_planes']
        for attr in planes_2_clear:
            setattr(self,attr,{})

    def return_consistency_loss(self):
        loss = None
        if self.consistentcy_loss_w is not None and len(self.consistentcy_loss)>0:
            loss = torch.mean(torch.stack(self.consistentcy_loss))
            self.consistentcy_loss = []
        return loss

    def forward(self, plane_name):
        if isinstance(plane_name,tuple):
            full_plane = False
            plane_roi = plane_name[1]
            plane_name = plane_name[0]
        else:
            full_plane = True
            plane_roi = torch.tensor([[-1,-1],[1,1]]).type(self.LR_planes[plane_name].type()).cuda()
        # if full_plane and plane_name in self.SR_planes:
        if plane_name in self.SR_planes:
            # assert plane_roi is None,'Unsupported yet'
            out = self.SR_planes[plane_name].cuda()
        else:
            LR_plane = self.LR_planes[plane_name].detach().cuda()
            if hasattr(self,'planes_mean_NON_LEARNED'):
                LR_plane = LR_plane-self.planes_mean_NON_LEARNED
                LR_plane = LR_plane/self.planes_std_NON_LEARNED
            plane_roi = (torch.tensor(LR_plane.shape[2:])).to(plane_roi.device)*(1+plane_roi)/2
            plane_roi = torch.stack([torch.floor(plane_roi[0]),torch.ceil(plane_roi[1])],0).cpu().numpy().astype(np.int32)
            plane_roi[0] = np.maximum(0,plane_roi[0]-1)
            plane_roi[1] = np.minimum(np.array(LR_plane.shape[2:]),plane_roi[1]+1)
            pre_padding = np.minimum(plane_roi[0],self.required_padding)
            post_padding = np.minimum(np.array(LR_plane.shape[2:])-plane_roi[1],self.required_padding)
            take_last = lambda ind: -ind if ind>0 else None
            DEBUG = False
            if DEBUG:   print('!! WARNING !!!!')
            x = LR_plane[...,plane_roi[0,0]-pre_padding[0]:plane_roi[1,0]+post_padding[0],plane_roi[0,1]-pre_padding[1]:plane_roi[1,1]+post_padding[1]]
            x = torch.nn.functional.pad(x,
                pad=(self.required_padding-pre_padding[1],self.required_padding-post_padding[1],self.required_padding-pre_padding[0],self.required_padding-post_padding[0]),
                mode='replicate') 
            difference = self.inner_forward(x)[...,self.HR_overpadding:take_last(self.HR_overpadding),self.HR_overpadding:take_last(self.HR_overpadding)]
            # if '_D0' in plane_name:
            #     print('!!!!!!WARNING!!!!!!!!')
            #     import matplotlib.pyplot as plt
            #     plt.plot(torch.std(difference.reshape([difference.shape[1],-1]),1).cpu().numpy(),'+')
            min_index = plane_roi[0]*self.scale_factor
            max_index = plane_roi[1]*self.scale_factor
            residual_plane = self.residual_plane(plane_name)
            super_resolved = torch.add(difference,residual_plane[...,min_index[0]:max_index[0],min_index[1]:max_index[1]])
            out = (torch.ones_like(residual_plane)*float('nan'))
            out[...,min_index[0]:max_index[0],min_index[1]:max_index[1]] = super_resolved
            if full_plane:   self.SR_planes[plane_name] = out.cpu()
            if self.consistentcy_loss_w is not None and self.training:
                self.consistentcy_loss.append(
                    self.planes_diff(
                        LR_plane[...,plane_roi[0,0]:plane_roi[1,0],plane_roi[0,1]:plane_roi[1,1]],
                        torch.nn.functional.interpolate(
                            super_resolved,scale_factor=1/self.scale_factor,mode=self.plane_interp,
                            align_corners=self.align_corners,antialias=True
                            )
                    )
                )
        return out

    def inner_forward(self,x):
        out = self.conv_input(x)
        out = self.conv_mid(self.residual(out))
        out = self.upscale(out)
        return self.conv_output(out)
        # return torch.add(out,residual)
