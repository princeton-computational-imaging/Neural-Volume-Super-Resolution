from collections import defaultdict,OrderedDict
from distutils.log import debug
import torch
import torch.nn as nn
from nerf_helpers import cart2az_el,rgetattr,rsetattr
import math
import numpy as np
from scipy.interpolate import griddata
from re import search
import os
from tqdm import tqdm
from shutil import copyfile
# from time import time
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
        num_planes_or_rot_mats=3,
        viewdir_mapping=False,
        plane_loss_w=None,
        scene_coupler=None,
        point_coords_noise=0,
        zero_mean_planes_w=None,
        ensemble_size=1,
    ):
        self.num_density_planes = num_planes_or_rot_mats if isinstance(num_planes_or_rot_mats,int) else len(num_planes_or_rot_mats)
        # self.PLANES_2_INFER = [self.num_density_planes]

        super(TwoDimPlanesModel, self).__init__()
        self.box_coords = coords_normalization
        self.use_viewdirs = use_viewdirs
        self.point_coords_noise = point_coords_noise
        self.num_plane_channels = num_plane_channels
        assert interp_viewdirs in ['bilinear','bicubic',None]
        assert interp_viewdirs is None,'Depricated'
        self.interp_from_learned = interp_viewdirs
        self.viewdir_downsampling = viewdir_downsampling
        self.align_corners = align_corners
        assert rgb_dec_input in ['projections','features','projections_features']
        self.rgb_dec_input = rgb_dec_input
        assert proj_combination in ['sum','concat','avg']
        assert use_viewdirs or viewdir_proj_combination is None
        if viewdir_proj_combination is None:    viewdir_proj_combination = proj_combination
        assert viewdir_proj_combination in ['sum','concat','avg','mult']
        self.proj_combination = proj_combination
        self.viewdir_proj_combination = viewdir_proj_combination
        self.plane_interp = plane_interp
        self.planes_ds_factor = 1
        self.skip_connect_every = skip_connect_every # if skip_connect_every is not None else max(dec_rgb_layers,dec_density_layers)
        self.coord_projector = CoordProjector(self.num_density_planes,rot_mats=None if isinstance(num_planes_or_rot_mats,int) else num_planes_or_rot_mats)
        self.viewdir_mapping = viewdir_mapping
        self.plane_loss_w = plane_loss_w
        self.scene_coupler = scene_coupler
        if self.plane_loss_w is not None:
            if len(scene_coupler.HR_planes)>0:
                self.planes_diff = nn.L1Loss()
                self.plane_loss = []
            else:
                assert self.plane_loss_w==0,'Cannot penalize for plane differences when using HR planes'
                self.plane_loss_w = None
        self.zero_mean_planes_w = zero_mean_planes_w
        if self.zero_mean_planes_w is not None:
            self.zero_mean_planes_loss = {}

        # Density (alpha) decoder:
        self.density_dec = nn.ModuleDict([(str(i),nn.ModuleList()) for i in range(ensemble_size)])
        # self.density_dec = nn.ModuleList()
        self.debug = {'max_norm':defaultdict(lambda: torch.finfo(torch.float32).min),'min_norm':defaultdict(lambda: torch.finfo(torch.float32).max)}
        in_channels = num_plane_channels*(self.num_density_planes if proj_combination=='concat' else 1)
        for i in range(ensemble_size):
            self.density_dec[str(i)].append(nn.Linear(in_channels,dec_channels))
            for layer_num in range(dec_density_layers-1):
                if self.is_skip_layer(layer_num=layer_num):
                    self.density_dec[str(i)].append(nn.Linear(in_channels + dec_channels, dec_channels))
                else:
                    self.density_dec[str(i)].append(nn.Linear(dec_channels,dec_channels))
        self.fc_alpha = nn.ModuleDict([(str(i),nn.Linear(dec_channels,1)) for i in range(ensemble_size)])
        if 'features' in self.rgb_dec_input:
            self.fc_feat = nn.ModuleDict([(str(i),nn.Linear(dec_channels,num_plane_channels)) for i in range(ensemble_size)])
        # self.fc_alpha = nn.Linear(dec_channels,1)
        # if 'features' in self.rgb_dec_input:
        #     self.fc_feat = nn.Linear(dec_channels,num_plane_channels)

        # RGB decoder:
        # self.rgb_dec = nn.ModuleList()
        self.rgb_dec = nn.ModuleDict([(str(i),nn.ModuleList()) for i in range(ensemble_size)])
        plane_C_mult = 1
        if proj_combination=='concat':  plane_C_mult += self.num_density_planes-1
        if use_viewdirs and viewdir_proj_combination=='concat':  plane_C_mult +=1

        for i in range(ensemble_size):
            self.rgb_dec[str(i)].append(nn.Linear(num_plane_channels*plane_C_mult,dec_channels))
            for layer_num in range(dec_rgb_layers-1):
                if self.is_skip_layer(layer_num=layer_num):
                    self.rgb_dec[str(i)].append(nn.Linear(num_plane_channels*plane_C_mult + dec_channels, dec_channels))
                else:
                    self.rgb_dec[str(i)].append(nn.Linear(dec_channels,dec_channels))
        # self.fc_rgb = nn.Linear(dec_channels,3)
        self.fc_rgb = nn.ModuleDict([(str(i),nn.Linear(dec_channels,3)) for i in range(ensemble_size)])

        self.relu = nn.functional.relu

        if scene_id_plane_resolution is not None: #If not using the store_panes configuration
            raise Exception('Depricated')
            if planes is None:
                self.planes_ = nn.ParameterDict([
                    (get_plane_name(id,d),
                        create_plane(res[0] if d<self.num_density_planes else res[1],num_plane_channels=num_plane_channels,init_STD=0.1*self.fc_alpha.weight.data.std())
                    )
                    for id,res in scene_id_plane_resolution.items() for d in range(self.num_density_planes+use_viewdirs)])
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

    def rot_mats(self):
        return self.coord_projector.rot_mats_NON_LEARNED

    def remove_low_freqs(self,plane,sf):
        return plane-torch.nn.functional.interpolate(\
                    torch.nn.functional.interpolate(
                        plane,
                        scale_factor=1/sf,mode=self.plane_interp,
                        align_corners=self.align_corners,
                    ),
                scale_factor=sf,mode=self.plane_interp,
                align_corners=self.align_corners,
            )

    def gen_plane(self,plane_name):
        if self.plane_rank is None or plane_name not in self.plane_rank:
            return self.planes_[plane_name]
        else:
            if plane_name not in self.generated_planes:
                self.generated_planes[plane_name] = torch.matmul(self.planes_[plane_name][...,:self.plane_rank[plane_name]],
                    self.planes_[plane_name][...,self.plane_rank[plane_name]:].permute([0,1,3,2]))
            return self.generated_planes[plane_name]

    def raw_plane(self,plane_name,downsample=False):
        plane = self.gen_plane(plane_name)
        if self.zero_mean_planes_w is not None and self.scene_coupler.plane2saved[plane_name] not in self.zero_mean_planes_loss:
            self.zero_mean_planes_loss[self.scene_coupler.plane2saved[plane_name]] = torch.mean(plane,(2,3)).abs().mean()
        if downsample:
            if plane_name not in self.downsampled_planes:
                self.downsampled_planes[plane_name] = self.downsample_plane(plane)
            plane = self.downsampled_planes[plane_name]
        else:
            if plane_name not in self.planes_ and not hasattr(self,'SR_model'): # Should only happen with the coarse model:
                raise Exception('No longer expected')
                plane_name = self.scene_coupler.downsample_planes[plane_name]
            # return self.planes_[plane_name]
        return plane

    def rot_mat_backward_support(self,loaded_dict):
        if not any(['rot_mats' in k for k in loaded_dict]):
            loaded_dict.update(dict([(k,v) for k,v in self.state_dict().items() if 'rot_mats' in k]))
        return loaded_dict

    def eval(self):
        super(TwoDimPlanesModel, self).eval()
        self.use_downsampled_planes(1)
        self.SR_planes2drop = []
        self.planes2drop = []
        
    def assign_SR_model(self,SR_model,SR_viewdir,save_interpolated=True,set_planes=True,plane_dropout=0,single_plane=True):
        self.SR_model = SR_model
        self.SR_model.align_corners = self.align_corners
        self.SR_model.SR_viewdir = SR_viewdir
        self.SR_plane_dropout = plane_dropout
        self.single_plane_SR = single_plane
        self.skip_SR_ = False
        assert not SR_viewdir,'Ceased supporting this option'
        if set_planes: 
            raise Exception('Depricated')
            for k,v in self.planes_.items():
                if not SR_viewdir and get_plane_name(None,self.num_density_planes) in k:  continue
                self.SR_model.set_LR_plane(v.detach(),id=k,save_interpolated=save_interpolated)

    def set_cur_scene_id(self,scene_id):
        self.cur_id = scene_id

    def normalize_coords(self,coords):
        EPSILON = 1e-5
        scene_name = self.cur_id+''
        normalized_coords = 2*(coords-self.box_coords[scene_name].type(coords.type())[:1])/\
            (self.box_coords[scene_name][1:]-self.box_coords[scene_name][:1]).type(coords.type())-1
        # assert normalized_coords.min()>=-1-EPSILON and normalized_coords.max()<=1+EPSILON,"Sanity check"
        self.debug['max_norm'][scene_name] = np.maximum(self.debug['max_norm'][scene_name],normalized_coords.max(0)[0].cpu().numpy())
        self.debug['min_norm'][scene_name] = np.minimum(self.debug['min_norm'][scene_name],normalized_coords.min(0)[0].cpu().numpy())
        return normalized_coords

    def use_downsampled_planes(self,ds_factor:int): # USed for debug
        self.planes_ds_factor = ds_factor

    def planes(self,dim_num:int,super_resolve:bool,grid:torch.tensor=None)->torch.tensor:
        plane_name = get_plane_name(self.cur_id,dim_num)
        downsample_plane = self.scene_coupler.should_downsample(plane_name)
        plane_name = self.scene_coupler.scene_with_saved_plane(plane_name,plane_not_scene=True)
        if super_resolve:
            SR_input = plane_name
            if grid is not None and self.SR_model.training and self.single_plane_SR:
                roi = torch.stack([grid.min(1)[0].squeeze(),grid.max(1)[0].squeeze()],0)
                roi = torch.stack([roi[:,1],roi[:,0]],1) # Converting from (x,y) to (y,x) on the columns dimension
                SR_input = (SR_input,roi)
            compute_plane_loss = self.plane_loss_w is not None and plane_name not in self.SR_model.SR_planes and plane_name in self.scene_coupler.upsample_planes.values()
            plane = self.SR_model(SR_input)
            if compute_plane_loss:
                plane_mask = torch.logical_not(torch.isnan(plane))
                self.plane_loss.append(self.planes_diff(plane[plane_mask],self.raw_plane(plane_name).to(plane.device)[plane_mask]))
        else:
            if self.planes_ds_factor>1 and (self.viewdir_downsampling or dim_num<self.num_density_planes): # Used for debug or enforcing loss
                plane = nn.functional.interpolate(self.raw_plane(plane_name,downsample_plane),scale_factor=1/self.planes_ds_factor,
                    align_corners=self.align_corners,mode=self.plane_interp,antialias=True)
            else:
                plane = self.raw_plane(plane_name,downsample_plane)
        return plane.cuda()

    def skip_SR(self,skip):
        self.skip_SR_ = skip

    def project_xyz(self,coords):
        projections = []
        joint_planes = None
        if self.point_coords_noise and self.training:
            projected_to_res = int(search('(?<=PlRes)(\d)+(?=_)',self.cur_id).group(0))
            coords = coords+torch.normal(mean=0,std=self.point_coords_noise*2/(1+projected_to_res),size=coords.shape).type(coords.type())
        for d in range(self.num_density_planes):
            grid = self.coord_projector((coords,d)).reshape([1,coords.shape[0],1,2])
            plane_name = get_plane_name(self.cur_id,d)
            # Check whether the planes SHOULD be super-resolved:
            super_resolve = hasattr(self,'SR_model') and (not hasattr(self,'scene_coupler') or self.scene_coupler.should_SR(plane_name,plane_not_scene=True))
            # Check whether the planes CAN be super-resolved:
            # super_resolve = super_resolve and (plane_name in self.SR_model.LR_planes or self.scene_coupler.plane2saved[plane_name] in self.SR_model.LR_planes)
            super_resolve = super_resolve and not self.skip_SR_
            if super_resolve and d in self.SR_planes2drop and self.single_plane_SR: super_resolve = False
            if joint_planes is None:
                input_plane = self.planes(d,super_resolve=super_resolve,grid=grid)
                if super_resolve and not self.single_plane_SR:
                    joint_planes = 1*input_plane

            if joint_planes is not None:
                input_plane = self.planes(d,super_resolve=False,grid=grid) if d in self.SR_planes2drop else joint_planes[:,d*self.num_plane_channels:(d+1)*self.num_plane_channels,...]
            projections.append(nn.functional.grid_sample(
                    input=input_plane,
                    grid=grid,
                    mode=self.plane_interp,
                    align_corners=self.align_corners,
                    padding_mode='border',
                ))
        projections = self.combine_pos_planes(projections)
        return projections.squeeze(0).squeeze(-1).permute(1,0)

    def project_viewdir(self,dirs):
        grid = dirs.reshape([1,dirs.shape[0],1,2])
        plane_name = get_plane_name(self.cur_id,self.num_density_planes)
        # Stopped supporting viewdir-planes SR:
        super_resolve = False and hasattr(self,'SR_model') and not self.skip_SR_ and plane_name in self.SR_model.LR_planes and self.num_density_planes not in self.SR_planes2drop
        assert not super_resolve,'Unexpected'
        if hasattr(self,'viewdir_plane_coverage') and self.training:
            plane_res = self.raw_plane(get_plane_name(self.cur_id,self.num_density_planes)).shape[3]
            logging_res = self.viewdir_plane_coverage[get_plane_name(self.cur_id,self.num_density_planes)].shape[0]
            covered_points = (grid/2*plane_res).squeeze()+logging_res/2
            floor_int = lambda x:torch.floor(x).type(torch.LongTensor)
            ceil_int = lambda x:torch.ceil(x).type(torch.LongTensor)
            # floor_int = lambda x:torch.clamp(torch.floor(x).type(torch.LongTensor),0,plane_res-1)
            # ceil_int = lambda x:torch.clamp(torch.ceil(x).type(torch.LongTensor),0,plane_res-1)
            for row_f in [floor_int,ceil_int]:
                for col_f in [floor_int,ceil_int]:
                    for p in covered_points[::64]:
                        self.viewdir_plane_coverage[get_plane_name(self.cur_id,self.num_density_planes)][row_f(p[0]),col_f(p[1])] += 1
            import matplotlib.pyplot as plt
            plt.imsave('viewdir_coverage_%s.png'%(self.cur_id),np.log(self.viewdir_plane_coverage[get_plane_name(self.cur_id,self.num_density_planes)].cpu().numpy()+1))
            plt.clf()
            plt.plot(self.viewdir_plane_coverage[get_plane_name(self.cur_id,self.num_density_planes)].mean(0))
            plt.plot(self.viewdir_plane_coverage[get_plane_name(self.cur_id,self.num_density_planes)].mean(1))
            plt.savefig('%s_coverage.png'%(get_plane_name(self.cur_id,self.num_density_planes)))
        return nn.functional.grid_sample(
                input=self.planes(self.num_density_planes,super_resolve=super_resolve,grid=grid),
                grid=grid,
                mode=self.plane_interp,
                align_corners=self.align_corners,
                padding_mode='border',
            ).squeeze(0).squeeze(-1).permute(1,0)

    def combine_pos_planes(self,tensors):
        if self.proj_combination=='sum':
            return torch.stack(tensors,0).sum(0)  
        elif self.proj_combination=='avg':
            return torch.stack([t for i,t in enumerate(tensors) if i not in self.planes2drop],0).mean(0)
        elif self.proj_combination=='concat':
            return torch.cat(tensors,1)

    def combine_all_planes(self,pos_planes,viewdir_planes):
        pos_planes_shape = pos_planes.shape
        if self.viewdir_proj_combination!='concat' and pos_planes_shape[1]>viewdir_planes.shape[1]:
            pos_planes = pos_planes.reshape([pos_planes_shape[0],viewdir_planes.shape[1],-1])
            viewdir_planes = viewdir_planes.unsqueeze(-1)
        if self.viewdir_proj_combination=='sum':
            return torch.reshape(pos_planes+viewdir_planes,pos_planes_shape)
        elif self.viewdir_proj_combination=='avg':
            return torch.reshape((pos_planes+viewdir_planes)/2,pos_planes_shape)
        elif self.viewdir_proj_combination=='mult':
            return torch.reshape(pos_planes*(1+viewdir_planes),pos_planes_shape)
        elif self.viewdir_proj_combination=='concat':
            return torch.cat([pos_planes,viewdir_planes],1)

    def forward(self, x):
        if self.use_viewdirs:
            x = torch.cat([x[...,:3],cart2az_el(x[...,3:])],-1)
        else:
            x = x[..., : 3]
        x = self.normalize_coords(x)
        projected_xyz = self.project_xyz(x[..., : 3])
        if self.use_viewdirs:
            projected_views = self.project_viewdir(x[...,3:])

        model_num = str(np.random.randint(len(self.density_dec))) if self.training else '0'
        # Projecting and summing
        x = 1*projected_xyz
        for layer_num,l in enumerate(self.density_dec[model_num]):
            if self.is_skip_layer(layer_num=layer_num-1):
                x = torch.cat((x, projected_xyz), dim=-1)
            x = self.relu(l(x))
        alpha = self.fc_alpha[model_num](x)

        if 'features' in self.rgb_dec_input:
            x_rgb = self.fc_feat[model_num](x)

        if self.rgb_dec_input=='projections_features':
            x_rgb = self.combine_pos_planes([x_rgb,projected_xyz])
        elif self.rgb_dec_input=='projections':
            x_rgb = 1*projected_xyz

        if self.use_viewdirs:
            x_rgb = self.combine_all_planes(pos_planes=x_rgb,viewdir_planes=projected_views)

        x = x_rgb
        for layer_num,l in enumerate(self.rgb_dec[model_num]):
            if self.is_skip_layer(layer_num=layer_num-1):
                x = torch.cat((x, x_rgb), dim=-1)
            x = self.relu(l(x))
        rgb = self.fc_rgb[model_num](x)

        return torch.cat((rgb, alpha), dim=-1)

    def return_planes_SR_loss(self):
        loss = None
        if self.plane_loss_w is not None and len(self.plane_loss)>0:
            loss = torch.mean(torch.stack(self.plane_loss))
            self.plane_loss = []
        return loss

    def downsample_plane(self,plane):
        return torch.nn.functional.interpolate(plane,scale_factor=1/self.scene_coupler.ds_factor,mode=self.plane_interp,align_corners=self.align_corners)

    def assign_LR_planes(self,scene=None):
        for k in self.planes_:
        # for k,v in self.planes_.items():
            if scene is not None and self.scene_coupler.scene2saved[scene] not in k:    continue
            if not self.SR_model.SR_viewdir and get_plane_name(None,self.num_density_planes) in k:  continue
            if self.scene_coupler.should_downsample(k,for_LR_loading=True):
                # LR_plane = self.downsample_plane(v)
                LR_plane = self.downsample_plane(self.raw_plane(k))
            else:
                # LR_plane = v
                LR_plane = self.raw_plane(k)
            if self.SR_model.detach_LR_planes: LR_plane = LR_plane.detach()
            self.SR_model.set_LR_plane(LR_plane,id=k,save_interpolated=False)

    def return_zero_mean_planes_loss(self):
        loss = None
        if self.zero_mean_planes_w is not None and len(self.zero_mean_planes_loss)>0:
            loss = torch.mean(torch.stack(list(self.zero_mean_planes_loss.values())))
            self.zero_mean_planes_loss = {}
        return loss


def create_plane(resolution,num_plane_channels,init_STD):
    if not isinstance(resolution,list):
        resolution = [resolution,resolution]
    return nn.Parameter(init_STD*torch.randn(size=[1,num_plane_channels,resolution[0],resolution[1]]))

class SceneSampler:
    def __init__(self,scenes:list,do_when_reshuffling=lambda:None) -> None:
        self.scenes = scenes
        self.do_when_reshuffling = lambda:None
        self.shuffle()
        self.do_when_reshuffling = do_when_reshuffling

    def shuffle(self,inhibit_func=False):
        self.sample_from = [self.scenes[i] for i in np.random.permutation(len(self.scenes))]
        if not inhibit_func:    self.do_when_reshuffling()

    def sample(self,n,just_shuffle=False):
        assert n<=len(self.scenes)
        sampled = []
        cursor = 0
        if just_shuffle: # Used when the buffer-size equals the total number of scenes
            self.shuffle(inhibit_func=True)
            sampled = [self.sample_from.pop() for i in range(len(self.sample_from))]
        else:
            while len(sampled)<n:
                if len(self.sample_from)==0:
                    self.shuffle()
                    cursor = 0
                if self.sample_from[cursor] not in sampled:
                    sampled.append(self.sample_from.pop(cursor))
                else:
                    cursor += 1
        return sampled

class CoordProjector(nn.Module):
    def __init__(self,N:int=None,rot_mats:nn.Parameter=None) -> None:
        super(CoordProjector,self).__init__()
        N_RANDOM_TRIALS = 10000
        if rot_mats is None:
            if N==3: #  For the basic case, conforming with the previous convention of the standard basis:
                base_mat = torch.eye(3)
                self.rot_mats_NON_LEARNED = nn.ParameterList([base_mat,base_mat[:,[1,0,2]],base_mat[:,[2,0,1]]])
            else:
                plane_axes = np.random.uniform(low=-1,high=1,size=[N_RANDOM_TRIALS,N,3])
                plane_axes /= np.sqrt(np.sum(plane_axes**2,2,keepdims=True))
                plane_axes = np.concatenate((plane_axes,-1*plane_axes),1)
                chosen = plane_axes[np.argmax(np.sum(np.sort(np.sum((plane_axes[...,None,:]-np.expand_dims(plane_axes,1))**2,-1),1)[:,1,...],-1))][:N]
                # chosen = np.array([[1,0,0],[0,1,0],[0,0,1]])
                self.rot_mats_NON_LEARNED = nn.ParameterList()
                for norm in chosen:
                    independent = False
                    while not independent:
                        mat = np.concatenate([norm[:,None],np.random.uniform(size=[3,2])],1)
                        independent = np.linalg.matrix_rank(mat)==3
                    self.rot_mats_NON_LEARNED.append(torch.from_numpy(np.linalg.qr(mat)[0]))
        else:
            assert len(rot_mats)==N
            self.rot_mats_NON_LEARNED = rot_mats

    def forward(self,points_dim):
        with torch.no_grad():
            return torch.matmul(points_dim[0],self.rot_mats_NON_LEARNED[points_dim[1]][:,1:].type(points_dim[0].type()))

class PlanesOptimizer(nn.Module):
    def __init__(self,optimizer_type:str,scene_id_plane_resolution:dict,options,save_location:str,
            lr:float,model_coarse:TwoDimPlanesModel,model_fine:TwoDimPlanesModel,use_coarse_planes:bool,
            init_params:bool,optimize:bool,training_scenes:list=None,coords_normalization:dict=None,
            do_when_reshuffling=lambda:None,STD_factor:float=0.1,
            available_scenes:list=[],planes_rank_ratio:float=None,copy_params_path=None,
            ) -> None:
        super(PlanesOptimizer,self).__init__()
        self.scenes = available_scenes
        if training_scenes is None:
            training_scenes = 1*self.scenes
        self.training_scenes = training_scenes
        self.scenes_with_planes = scene_id_plane_resolution.keys()
        assert len(available_scenes)>0
        assert copy_params_path is None or not init_params,"Those two don't work together"
        self.buffer_size = getattr(options,'buffer_size',len(self.training_scenes))
        self.steps_per_buffer,self.steps_since_drawing = options.steps_per_buffer,0
        if self.buffer_size>=len(self.training_scenes):  self.steps_per_buffer = -1
        assert all([s in self.scenes or model_fine.scene_coupler.downsample_couples[s] in self.scenes for s in self.training_scenes])
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
        plane_rank_dict = None if planes_rank_ratio is None else dict([(get_plane_name(sc,d),int(np.ceil(planes_rank_ratio*res[0]))) for sc,res in scene_id_plane_resolution.items() for d in range(model_coarse.num_density_planes)])
        self.generated_planes = {}
        self.downsampled_planes = {}
        for model_name,model in zip(['coarse','fine'],[model_coarse,model_fine]):
            self.models[model_name] = model
            model.plane_rank = plane_rank_dict
            model.generated_planes = self.generated_planes
            model.downsampled_planes = self.downsampled_planes
            if model_name=='fine' and use_coarse_planes:    continue
            self.planes_per_scene = model.num_density_planes+model.use_viewdirs
            if init_params or copy_params_path:
                if model.viewdir_mapping:
                    model.viewdir_plane_coverage = {}
                for scene,res in tqdm(scene_id_plane_resolution.items(),desc='Initializing scene planes',):
                    if init_params:
                        params = nn.ParameterDict([
                            (get_plane_name(scene,d),
                                create_plane(res[0] if d<model.num_density_planes else res[1],num_plane_channels=model.num_plane_channels,
                                init_STD=STD_factor*model.fc_alpha['0'].weight.data.std().cpu())
                                if plane_rank_dict is None or d>=model.num_density_planes else
                                create_plane([res[0],2*plane_rank_dict[get_plane_name(scene,d)]] ,num_plane_channels=model.num_plane_channels,
                                init_STD=np.sqrt(STD_factor*model.fc_alpha.weight.data.std().cpu()))
                            )
                            for d in range(self.planes_per_scene)])
                        cn = coords_normalization[scene]
                    else:
                        params = torch.load(self.param_path(model_name=model_name,scene=scene,save_location=copy_params_path))
                        cn = params['coords_normalization']
                        params = params['params']
                    torch.save({'params':params,'coords_normalization':cn},self.param_path(model_name=model_name,scene=scene))
                    if model.viewdir_mapping:
                        model.viewdir_plane_coverage[get_plane_name(scene,model.num_density_planes)] = torch.zeros([res[1]+15,res[1]+15])

        self.optimize = optimize
        self.optimizer = None
        self.saving_needed = False

    def load_scene(self,scene):
        if self.saving_needed:
            self.save_params()
        for model_name in ['coarse','fine']:
            model = self.models[model_name]
            # scenes_planes_name = scene if scene in self.scenes_with_planes else model.scene_coupler.scene_with_saved_plane(scene)
            scenes_planes_name = model.scene_coupler.scene2saved[scene]
            if model_name=='coarse' or not self.use_coarse_planes:
                loaded_params = torch.load(self.param_path(model_name=model_name,scene=scenes_planes_name))
                # model.scene_coupler.downsampled_planes = {}
            model.planes_ = loaded_params['params'].cuda()
            self.generated_planes.clear()
            self.downsampled_planes.clear()
            model.box_coords = {scenes_planes_name:loaded_params['coords_normalization'],scene:loaded_params['coords_normalization']}
            if hasattr(model,'SR_model'):
                model.SR_model.clear_SR_planes(all_planes=True)
                if model.scene_coupler.should_SR(scene):
                    model.assign_LR_planes()

        self.cur_scenes = [scene]

    def load_from_checkpoint(self,checkpoint):
        model_name = 'coarse'
        param2num = dict([(k,i) for i,k in enumerate(checkpoint['plane_parameters'])])
        for scene in self.scenes:
            params = nn.ParameterDict([(get_plane_name(scene,d),checkpoint['plane_parameters'][get_plane_name(scene,d)]) for d in range(self.planes_per_scene)])
            opt_states = [checkpoint['plane_optimizer_states'][param2num[get_plane_name(scene,d)]] for d in range(self.planes_per_scene)]
            torch.save({'params':params,'opt_states':opt_states},self.param_path(model_name=model_name,scene=scene))

    def param_path(self,model_name,scene,save_location=None):
        if save_location is None:
            save_location = self.save_location
        return os.path.join(save_location,"%s_%s.par"%(model_name,scene))

    def get_plane_stats(self,viewdir=False,single_plane=True):
        model_name='coarse'
        plane_means,plane_STDs = [],[]
        # for scene in tqdm(self.training_scenes,desc='Collecting plane statistics'):
        # for scene in tqdm(self.models[model_name].scene_coupler.downsample_couples.values(),desc='Collecting plane statistics'):
        for scene in tqdm(self.training_scenes,desc='Collecting plane statistics'):
            loaded_params = torch.load(self.param_path(model_name=model_name,scene=self.models[model_name].scene_coupler.scene2saved[scene]))
            for k,p in loaded_params['params'].items():
                if not viewdir and get_plane_name(None,self.models[model_name].num_density_planes) in k:  continue
                plane_means.append(torch.mean(p,(2,3)).squeeze(0))
                plane_STDs.append(torch.std(p.reshape(p.shape[1],-1),1))
        if single_plane:
            return {'mean':torch.stack(plane_means,0).mean(0),'std':torch.stack(plane_STDs,0).mean(0)}
        else:
            return {'mean':torch.cat(plane_means,0).reshape([len(self.training_scenes),-1]).mean(0),'std':torch.cat(plane_STDs,0).reshape([len(self.training_scenes),-1]).mean(0)}

    def save_params(self,to_checkpoint=False):
        assert self.optimize,'Why would you want to save if not optimizing?'
        model_name = 'coarse'
        model = self.models[model_name]
        scenes_list = self.scenes if to_checkpoint else self.cur_scenes
        if to_checkpoint:
            all_params,all_states = nn.ParameterDict(),[]
        scene_num = 0
        already_saved = []
        for scene in scenes_list:
            scene = model.scene_coupler.scene_with_saved_plane(scene)
            if scene in already_saved: continue
            already_saved.append(scene)
            if scene in self.cur_scenes:
                # params = nn.ParameterDict([(get_plane_name(scene,d),model.raw_plane(get_plane_name(scene,d),downsample=False)) for d in range(self.planes_per_scene)])
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

    def draw_scenes(self,assign_LR_planes=True):
        if self.saving_needed:
            self.save_params()
        self.steps_since_drawing = 0
        self.cur_scenes = self.scene_sampler.sample(self.buffer_size,just_shuffle=self.steps_per_buffer==-1)
        for model_name in ['coarse','fine']:
            model = self.models[model_name]
            if model_name=='coarse' or not self.use_coarse_planes:
                # model.scene_coupler.downsampled_planes = {}
                params_dict,optimizer_states,box_coords = nn.ParameterDict(),[],{}
                already_loaded = []
                scenes2load = self.cur_scenes if len(self.cur_scenes)<20 else tqdm(self.cur_scenes,desc='Loading scene planes')
                for scene in scenes2load:
                    scene = model.scene_coupler.scene_with_saved_plane(scene)
                    if scene in already_loaded: continue
                    already_loaded.append(scene)
                    try:
                        loaded_params = torch.load(self.param_path(model_name=model_name,scene=scene))
                    except Exception as e:
                        print("!!!! WARNING: planes model seems to be currpted for scene %s:\n%s"%(scene,e))
                        raise e
                        # res = [int(search('(?<=PlRes)(\d)+(?=_)',scene).group(0)),int(search('(?<=_)(\d)+(?=$)',scene).group(0))]
                        # params = nn.ParameterDict([
                        #     (get_plane_name(scene,d),
                        #         create_plane(res[0] if d<model.num_density_planes else res[1],num_plane_channels=model.num_plane_channels,
                        #         init_STD=STD_factor*model.fc_alpha['0'].weight.data.std().cpu())
                        #     )
                        #     for d in range(self.planes_per_scene)])
                        # cn = coords_normalization[scene]
                        # torch.save({'params':params,'coords_normalization':cn},self.param_path(model_name=model_name,scene=scene))
                    params_dict.update(loaded_params['params'])
                    box_coords.update(dict([(sc,loaded_params['coords_normalization']) for sc in [scene]+model.scene_coupler.coupled_scene(scene)]))
                    if self.optimize:
                        if 'opt_states' in loaded_params:
                            optimizer_states.extend(loaded_params['opt_states'])
                        else:
                            optimizer_states.extend([None for p in loaded_params['params']])
            model.planes_ = params_dict.cuda()
            self.generated_planes.clear()
            self.downsampled_planes.clear()
            self.models[model_name].box_coords = box_coords
            if hasattr(model,'SR_model'):
                model.SR_model.clear_SR_planes(all_planes=True)
                if assign_LR_planes:
                    model.assign_LR_planes()

            if not self.optimize:   continue
            if model_name=='coarse' or not self.use_coarse_planes:
                params = list(params_dict.values())
                if self.optimizer is None: # First call to this function:
                    self.optimizer = torch.optim.Adam(params, lr=self.lr)
                else:
                    self.optimizer.param_groups[0]['params'] = params
                self.optimizer.state = defaultdict(dict,[(params[i],v) for i,v in enumerate(optimizer_states) if v is not None])
        self.saving_needed = False

    def step(self):
        if self.optimize:
            self.optimizer.step()
            self.saving_needed = True
            self.generated_planes.clear()
            self.downsampled_planes.clear()
        self.steps_since_drawing += 1
        if hasattr(self.models['fine'],'SR_model'):
            self.models['fine'].SR_model.clear_SR_planes(all_planes=self.optimize)
            # self.models['fine'].SR_model.clear_SR_planes(all_planes=reload_planes_4_SR)

        if self.steps_since_drawing==self.steps_per_buffer:
            # self.draw_scenes(assign_LR_planes=not reload_planes_4_SR)
            self.draw_scenes(assign_LR_planes=not self.optimize)
            return self.cur_scenes
        else:
            # if reload_planes_4_SR and hasattr(self.models['fine'],'SR_model'):
            #     self.models['fine'].assign_LR_planes()
            return None

    def zero_grad(self):
        if self.optimize:   self.optimizer.zero_grad()

    def jump_start(self,config=None,on=True):
        items2memorize = ['steps_per_buffer'] #,'scene_sampler.scenes''buffer_size',
        if on:
            num_scenes = config[0]
            if isinstance(num_scenes,float):
                num_scenes = int(np.ceil(num_scenes*len(self.scene_sampler.scenes)))
            self.memory_dict = dict([(k,rgetattr(self,k)) for k in items2memorize])
            self.scene_sampler.sample_from = []
            self.steps_per_buffer = -1
            print('\nTraining using only %d scenes until average loss drops below %.2e'%(num_scenes,config[1]))
            return num_scenes
        else:
            for k in items2memorize:
                rsetattr(self,k,self.memory_dict[k])
            self.scene_sampler.sample_from = []
            self.draw_scenes()
            print('\nJump-start phase over!!!')
            return self.cur_scenes


# EDSR code taken and modified from https://github.com/twtygqyy/pytorch-edsr/blob/master/edsr.py
class _Residual_Block(nn.Module): 
    def __init__(self,hidden_size,padding,kernel_size):
        super(_Residual_Block, self).__init__()
        self.margins = None if (padding or kernel_size==1) else 2*(kernel_size//2)
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
    def __init__(self,scale_factor,in_channels,out_channels,sr_config,plane_interp,detach_LR_planes):
        super(EDSR, self).__init__()

        hidden_size = sr_config.model.hidden_size
        n_blocks = sr_config.model.n_blocks
        input_normalization = sr_config.get("input_normalization",False)
        self.detach_LR_planes = detach_LR_planes
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        self.scale_factor = scale_factor
        self.plane_interp = plane_interp
        self.n_blocks = n_blocks
        self.input_noise = getattr(sr_config,'sr_input_noise',0)
        self.output_noise = getattr(sr_config,'sr_output_noise',0)
        self.single_plane = getattr(sr_config.model,'single_plane',True)
        self.per_channel_sr = getattr(sr_config.model,'per_channel_sr',False)
        if self.per_channel_sr:
            in_channels = out_channels = 1
        PADDING,KERNEL_SIZE = 0,3
        # kernel_size = 1*KERNEL_SIZE
        self.required_padding,rf_factor = 0,1

        def kernel_size(num_layers=1):
            if (1+2*(self.required_padding+rf_factor*num_layers*((KERNEL_SIZE-1)//2)))<=getattr(sr_config.model,'receptive_field_bound',np.iinfo(np.int32).max):
                self.required_padding += rf_factor*num_layers*(KERNEL_SIZE//2)
                return KERNEL_SIZE
            else:
                return 1

        # self.required_padding,rf_factor = KERNEL_SIZE//2,1
        # kernel_size = lambda: KERNEL_SIZE if (1+2*self.required_padding)<=getattr(sr_config.model,'receptive_field_bound',np.iinfo(np.int32).max) else 1
        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=kernel_size(), stride=1, padding=PADDING, bias=False)
        # self.required_padding,rf_factor = kernel_size()//2,1

        self.residual = nn.Sequential()
        for _ in range(n_blocks):
            self.residual.append(_Residual_Block(hidden_size=hidden_size,padding=PADDING,kernel_size=kernel_size(2)))
        # self.residual = self.make_layer(_Residual_Block,{'hidden_size':hidden_size,'padding':PADDING,'kernel_size':kernel_size()}, n_blocks)
        # self.required_padding += rf_factor*2*n_blocks*((kernel_size()-1)//2)

        self.conv_mid = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size(), stride=1, padding=PADDING, bias=False)
        # self.required_padding += rf_factor*((kernel_size()-1)//2)
        assert math.log2(scale_factor)==int(math.log2(scale_factor)),"Supperting only scale factors that are an integer power of 2."
        upscaling_layers = []
        for _ in range(int(math.log2(scale_factor))):
            upscaling_layers += [
                nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*4, kernel_size=kernel_size(), stride=1, padding=PADDING, bias=False),
                nn.PixelShuffle(2),
            ]
            # self.required_padding += rf_factor*((kernel_size()-1)//2)
            rf_factor /= 2

        self.upscale = nn.Sequential(*upscaling_layers)

        self.conv_output = nn.Conv2d(in_channels=hidden_size, out_channels=out_channels, kernel_size=kernel_size(), stride=1, padding=PADDING, bias=False)
        # self.required_padding += rf_factor*((kernel_size()-1)//2)
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
        self.consistency_loss_w = sr_config.get("consistency_loss_w",None)
        if self.consistency_loss_w is not None:
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

    def set_LR_plane(self,plane,id:str,save_interpolated:bool):
        assert id not in self.LR_planes,"Plane ID already exists."
        if self.single_plane or get_plane_name(None,0) in id:
            self.LR_planes[id] = plane
        else:
            assert not save_interpolated,'Unsupported'
            self.LR_planes[id] = None
            dim_num = int(search('(?<=_D)(\d)+(?=$)',id.split('PlRes')[-1]).group(0))
            id = id.replace(get_plane_name(None,dim_num),get_plane_name(None,0))
            self.LR_planes[id] = torch.cat((self.LR_planes[id],plane),1)
        if save_interpolated:
            self.residual_planes[id] = self.interpolate_LR(id).cpu()

    def clear_SR_planes(self,all_planes=False):
        planes_2_clear = ['SR_planes']
        if all_planes:
            planes_2_clear += ['LR_planes','residual_planes']
        for attr in planes_2_clear:
            setattr(self,attr,{})

    def return_consistency_loss(self):
        loss = None
        if self.consistency_loss_w is not None and len(self.consistentcy_loss)>0:
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
        if plane_name in self.SR_planes:
            out = self.SR_planes[plane_name].cuda()
        else:
            LR_plane = self.LR_planes[plane_name].cuda()
            if self.detach_LR_planes:
                LR_plane = LR_plane.detach()
            if self.training and self.input_noise>0:
                LR_plane = torch.add(LR_plane,torch.normal(mean=0,std=self.input_noise*LR_plane.std(),size=LR_plane.shape).type(LR_plane.type()))
            x = 1*LR_plane
            if hasattr(self,'planes_mean_NON_LEARNED'):
                x = x-self.planes_mean_NON_LEARNED
                x = x/self.planes_std_NON_LEARNED
            plane_roi = (torch.tensor(x.shape[2:])).to(plane_roi.device)*(1+plane_roi)/2
            plane_roi = torch.stack([torch.floor(plane_roi[0]),torch.ceil(plane_roi[1])],0).cpu().numpy().astype(np.int32)
            plane_roi[0] = np.maximum(0,plane_roi[0]-1)
            plane_roi[1] = np.minimum(np.array(x.shape[2:]),plane_roi[1]+1)
            pre_padding = np.minimum(plane_roi[0],self.required_padding)
            post_padding = np.minimum(np.array(x.shape[2:])-plane_roi[1],self.required_padding)
            take_last = lambda ind: -ind if ind>0 else None
            DEBUG = False
            if DEBUG:   print('!! WARNING !!!!')
            x = x[...,plane_roi[0,0]-pre_padding[0]:plane_roi[1,0]+post_padding[0],plane_roi[0,1]-pre_padding[1]:plane_roi[1,1]+post_padding[1]]
            x = torch.nn.functional.pad(x,
                pad=(self.required_padding-pre_padding[1],self.required_padding-post_padding[1],self.required_padding-pre_padding[0],self.required_padding-post_padding[0]),
                mode='replicate')
            if self.per_channel_sr:
                difference = torch.cat([self.inner_forward(x[:,ch:ch+1,...]) for ch in range(x.shape[1])],1)[...,self.HR_overpadding:take_last(self.HR_overpadding),self.HR_overpadding:take_last(self.HR_overpadding)]
            else:
                difference = self.inner_forward(x)[...,self.HR_overpadding:take_last(self.HR_overpadding),self.HR_overpadding:take_last(self.HR_overpadding)]
            # if '_D0' in plane_name:
            #     print('!!!!!!WARNING!!!!!!!!')
            #     import matplotlib.pyplot as plt
            #     plt.plot(torch.std(difference.reshape([difference.shape[1],-1]),1).cpu().numpy(),'+')
            min_index = plane_roi[0]*self.scale_factor
            max_index = plane_roi[1]*self.scale_factor
            residual_plane = self.residual_plane(plane_name)
            super_resolved = torch.add(difference,residual_plane[...,min_index[0]:max_index[0],min_index[1]:max_index[1]])
            if self.training and self.output_noise>0:
                super_resolved = torch.add(super_resolved,torch.normal(mean=0,std=self.output_noise*difference.detach().std(),size=super_resolved.shape).type(super_resolved.type()))
            out = (torch.ones_like(residual_plane)*float('nan'))
            out[...,min_index[0]:max_index[0],min_index[1]:max_index[1]] = super_resolved
            if full_plane:   
                self.SR_planes[plane_name] = out.cpu()
                # if not self.training:
                #     if not hasattr(self,'STD_hist'):    self.STD_hist = []
                #     self.STD_hist.append(torch.nn.Unfold(kernel_size=4)(difference).reshape([48,16,-1]).std(1).mean().item())
                #     print('LR/SR %s, mean abs diff %.3f/%.3f (%.1f), mean diff STD %.3f/%.3f (%.1f). Spatial STD %.3f.'%
                #         (plane_name,LR_plane.abs().mean(),difference.abs().mean(),LR_plane.abs().mean()/difference.abs().mean(),
                #         LR_plane.std(),difference.std(),LR_plane.std()/difference.std(),self.STD_hist[-1]))
            if self.consistency_loss_w is not None: # and self.training:
                self.consistentcy_loss.append(
                    torch.nn.functional.interpolate(
                        difference,
                        scale_factor=1/self.scale_factor,mode=self.plane_interp,
                        align_corners=self.align_corners,
                        antialias=True
                    ).abs().mean()
                )
        return out

    def inner_forward(self,x):
        out = self.conv_input(x)
        out = self.conv_mid(self.residual(out))
        out = self.upscale(out)
        return self.conv_output(out)
        # return torch.add(out,residual)

def get_scene_id(basedir,ds_factor,plane_res):
    return '%s_DS%d_PlRes%s'%(basedir,ds_factor,'' if plane_res[0] is None else '%d_%d'%(plane_res))

def extract_ds_and_res(scene_name):
    ds = int(search('(?<=_DS)(\d)+(?=_PlRes)',scene_name).group(0))
    res = int(search('(?<=_PlRes)(\d)+(?=_)',scene_name).group(0))
    return ds,res

class SceneCoupler:
    def __init__(self,scenes_list,planes_res_level,num_pos_planes,viewdir_plane,training_scenes) -> None:
        num_planes = num_pos_planes+viewdir_plane
        name_pattern = lambda name: '^'+name.split('_DS')[0]+'_DS'+'(\d)+_PlRes(\d)+_'+name.split('_')[-1]
        ds_ratios,res_ratios = [],[]
        self.upsample_couples,self.downsample_couples,self.HR_planes_LR_ims_scenes = {},{},[]
        scenes_list = list(set(scenes_list+training_scenes))
        for sc_num in range(len(scenes_list)):
            matching_scenes = [sc for sc in [s for i,s in enumerate(scenes_list) if i!=sc_num] if search(name_pattern(scenes_list[sc_num]),sc)]
            if len(matching_scenes)>0:
                assert len(matching_scenes)==1 or planes_res_level!='HR_planes','Not supporting HR planes with multiple scene matches'
                org_vals = extract_ds_and_res(scenes_list[sc_num])
                for match in matching_scenes:
                    found_vals = extract_ds_and_res(match)
                    res_ratio = found_vals[1]/org_vals[1]
                    if res_ratio==1:
                        continue
                    res_ratios.append(res_ratio)
                    ds_ratios.append(found_vals[0]/org_vals[0])
                    if ds_ratios[-1]==1 and res_ratio>1:
                        self.HR_planes_LR_ims_scenes.append(match)
                    # if ds_ratios[-1]>1:
                    if res_ratios[-1]<1:
                        if scenes_list[sc_num] in training_scenes:
                            self.upsample_couples[match] = scenes_list[sc_num]
                        self.downsample_couples[scenes_list[sc_num]] = match
                    elif res_ratios[-1]>1:
                        self.downsample_couples[match] = scenes_list[sc_num]
                        if match in training_scenes:
                            self.upsample_couples[scenes_list[sc_num]] = match
        
        self.ds_factor = int(max(1/res_ratios[0],res_ratios[0])) if len(self.downsample_couples)>0 else 1
        for match_num in range(len(ds_ratios)):
            if res_ratios[match_num]!=1/ds_ratios[match_num]:
                assert ds_ratios[match_num]==1,"I expect to have the downsampling factor match the plane resolution ratio."
            assert res_ratios[match_num] in [self.ds_factor,1/self.ds_factor],"Not all plane resolution ratios/downsampling factors are the same"
        self.use_HR_planes = False
        if planes_res_level=='HR_planes':
            # self.downsampled_planes = {}
            self.HR_planes = [get_plane_name(k,d) for k in self.downsample_couples.keys() for d in range(num_pos_planes)]
            self.scene2saved = dict([(sc,self.upsample_couples[sc] if sc in self.upsample_couples else sc if sc in training_scenes else self.downsample_couples[sc])
                for sc in scenes_list])
            self.upsample_planes = dict([(get_plane_name(lr_scene,d),get_plane_name(hr_scene,d))\
                for lr_scene,hr_scene in self.upsample_couples.items() if hr_scene in training_scenes for d in range(num_pos_planes)])
        else:
            self.HR_planes = []
            self.scene2saved = dict([(sc,self.downsample_couples[sc] if sc in self.downsample_couples else sc) for sc in scenes_list])
            self.downsample_planes = dict([(get_plane_name(hr_scene,d),get_plane_name(lr_scene,d))\
                for hr_scene,lr_scene in self.downsample_couples.items() for d in range(num_pos_planes)])
        self.plane2saved = dict([(get_plane_name(k,d),get_plane_name(v,d)) for k,v in self.scene2saved.items() for d in range(num_planes)])

    def toggle_used_planes_res(self,HR):
        if HR==True:    assert len(self.HR_planes)>0,'No HR planes to use.'
        self.use_HR_planes = HR

    def coupled_scene(self,scene):
        couples = []
        if scene in self.downsample_couples:
            couples.append(self.downsample_couples[scene])
        if scene in self.upsample_couples:
            couples.append(self.upsample_couples[scene])
        assert len(couples)<=1,"Expecting to have up to 1 couple, since this function is called during training phase, where each actual scene should not have more than two 'virtual' scenes associated with it."
        return couples
        # return self.downsample_couples[scene] if scene in self.downsample_couples else self.upsample_couples[scene] if scene in self.upsample_couples else scene

    def scene_with_saved_plane(self,scene,plane_not_scene=False):
        return getattr(self,('plane' if plane_not_scene else 'scene')+'2saved')[scene]
        # if (not self.HR_planes) and scene in getattr(self,'downsample_'+('planes' if plane_not_scene else 'couples')):
        #     return getattr(self,'downsample_'+('planes' if plane_not_scene else 'couples'))[scene]
        # elif self.HR_planes and scene in getattr(self,'upsample_'+('planes' if plane_not_scene else 'couples')):
        #     return getattr(self,'upsample_'+('planes' if plane_not_scene else 'couples'))[scene]
        # else:
        #     return scene

    def should_SR(self,scene,plane_not_scene=False):
        if len(self.HR_planes)>0:
            if self.use_HR_planes:
                return False
            else:
                # return scene in getattr(self,'upsample_'+('planes' if plane_not_scene else 'couples')).values()
                return scene in (self.HR_planes if plane_not_scene else self.downsample_couples.keys())
        else:
            return scene in getattr(self,'downsample_'+('planes' if plane_not_scene else 'couples'))

    def should_downsample(self,plane_name,for_LR_loading=False):
        return len(self.HR_planes)>0 and (for_LR_loading or not self.use_HR_planes) and self.plane2saved[plane_name] in self.HR_planes
