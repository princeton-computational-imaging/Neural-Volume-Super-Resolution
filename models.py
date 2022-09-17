from collections import defaultdict,OrderedDict
import torch
import torch.nn as nn
from nerf_helpers import cart2az_el
import math
import numpy as np
from scipy.interpolate import griddata
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

class TwoDimPlanesModel(nn.Module):
    def __init__(
        self,
        use_viewdirs,
        scene_id_plane_resolution,
        # scene_geometry,
        coords_normalization,
        dec_density_layers=4,
        dec_rgb_layers=4,
        dec_channels=128,
        # skip_connect_every=4,
        num_plane_channels=48,
        rgb_dec_input='projections',
        proj_combination='sum',
        planes=None,
        plane_interp='bilinear',
        align_corners=True,
        interp_viewdirs=None,
        viewdir_downsampling=True,
        # track_planes_coverage=False,
    ):
        self.N_PLANES_DENSITY = 3
        self.PLANES_2_INFER = [self.N_PLANES_DENSITY]
        # self.PLANES_2_INFER = [i for i in range(self.N_PLANES_DENSITY+1)]

        super(TwoDimPlanesModel, self).__init__()
        # self.ds_2_res = dict([(v,k) for k,v in plane_resolutions.items()])
        # self.box_coords = calc_scene_box(scene_geometry=scene_geometry,including_dirs=use_viewdirs)
        self.box_coords = coords_normalization
        self.use_viewdirs = use_viewdirs
        assert interp_viewdirs in ['bilinear','bicubic',None]
        self.interp_from_learned = interp_viewdirs
        self.viewdir_downsampling = viewdir_downsampling
        self.align_corners = align_corners
        assert rgb_dec_input in ['projections','features','projections_features']
        self.rgb_dec_input = rgb_dec_input
        assert proj_combination in ['sum','concat']
        self.proj_combination = proj_combination
        self.plane_interp = plane_interp
        self.planes_ds_factor = 1
        # self.track_planes_coverage = track_planes_coverage
        # Density (alpha) decoder:
        self.density_dec = nn.ModuleList()
        self.debug = {'max_norm':defaultdict(lambda: torch.finfo(torch.float32).min),'min_norm':defaultdict(lambda: torch.finfo(torch.float32).max)}
        self.density_dec.append(
            nn.Linear(num_plane_channels*(self.N_PLANES_DENSITY if proj_combination=='concat' else 1),dec_channels)
        )
        for layer_num in range(dec_density_layers-1):
            self.density_dec.append(
                nn.Linear(dec_channels,dec_channels)
            )
        self.fc_alpha = nn.Linear(dec_channels,1)
        if 'features' in self.rgb_dec_input:
            self.fc_feat = nn.Linear(dec_channels,num_plane_channels)

        # RGB decoder:
        self.rgb_dec = nn.ModuleList()
        self.rgb_dec.append(
            nn.Linear(num_plane_channels*(self.N_PLANES_DENSITY+1 if proj_combination=='concat' else 1),dec_channels)
        )
        for layer_num in range(dec_rgb_layers-1):
            self.rgb_dec.append(
                nn.Linear(dec_channels,dec_channels)
            )
        self.fc_rgb = nn.Linear(dec_channels,3)

        self.relu = nn.functional.relu
        def create_plane(resolution):
            init_STD = 0.1*self.fc_alpha.weight.data.std()
            # init_STD = 0.1*self.fc_alpha.weight.data.std()/np.sqrt(resolution/400)
            # init_STD = 0.14
            return nn.Parameter(init_STD*torch.randn(size=[1,num_plane_channels,resolution,resolution]))

        if planes is None:
            self.planes_ = nn.ParameterDict([
                (self.plane_name(id,d),
                    create_plane(res[0] if d<self.N_PLANES_DENSITY else res[1])
                    # nn.Parameter(0.1*self.fc_alpha.weight.data.std()*torch.randn(size=[1,num_plane_channels,res,res]))
                )
                 for id,res in scene_id_plane_resolution.items() for d in range(self.N_PLANES_DENSITY+use_viewdirs)])
            # if self.track_planes_coverage:
            #     raise Exception("Should be adapted to support multiple plane resolutions.")
            #     self.planes_coverage = torch.zeros([len(self.planes_),plane_resolutions,plane_resolutions,]).type(torch.bool)
            if self.interp_from_learned:
                self.copy_planes()
                # self.planes_copy = OrderedDict([(k,1*v.detach().cpu().numpy()) for k,v in self.planes_.items()])
                assert not align_corners,'The following corresponding grid assumes -1 and 1 correspond to array corners (rather than the center of its corner pixels)'
                self.corresponding_grid = OrderedDict()
                for k,v in self.planes_copy.items():
                    res = list(v.shape[2:])
                    # Dimensions of self.corresponding_grid[k] are resolution X resolution X 2, where the last dimension corresponds to indecis [x,y] (column,row):
                    self.corresponding_grid[k] = np.stack(np.meshgrid(np.linspace(-1+1/res[1],1-1/res[1],res[1]),np.linspace(-1+1/res[0],1-1/res[0],res[0])),-1)
        else:
            self.planes_ = planes

    def copy_planes(self):
        self.planes_copy = OrderedDict([(k,1*v.detach().cpu().numpy()) for k,v in self.planes_.items() if any([self.plane_name(None,d) in k for d in self.PLANES_2_INFER])])

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
        # self.planes_[k]

    def plane_name(self,scene_id,dimension):
        if scene_id is None:
            return "_D%d"%(dimension)    
        return "sc%s_D%d"%(scene_id,dimension)
        
    def assign_SR_model(self,SR_model,SR_viewdir):
        self.SR_model = SR_model
        for k,v in self.planes_.items():
            if not SR_viewdir and self.plane_name(None,self.N_PLANES_DENSITY) in k:  continue
            self.SR_model.set_LR_planes(v.detach(),id=k,align_corners=self.align_corners)
        self.skip_SR_ = False

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

    def planes(self,dim_num:int)->torch.tensor:
        plane_name = self.plane_name(self.cur_id,dim_num)
        if hasattr(self,'SR_model') and plane_name in self.SR_model.LR_planes and not self.skip_SR_:
            plane = self.SR_model(plane_name)
        else:
            if self.planes_ds_factor>1 and (self.viewdir_downsampling or dim_num<self.N_PLANES_DENSITY): # USed for debug or enforcing loss
                plane = nn.functional.interpolate(self.planes_[plane_name],scale_factor=1/self.planes_ds_factor,
                    align_corners=self.align_corners,mode=self.plane_interp,antialias=True)
            else:
                plane = self.planes_[plane_name]
        return plane

    def skip_SR(self,skip):
        self.skip_SR_ = skip

    def project_xyz(self,coords):
        projections = []
        for d in range(self.N_PLANES_DENSITY): # (Currently not supporting viewdir input)
            grid = coords[:,[c for c in range(3) if c!=d]].reshape([1,coords.shape[0],1,2])
            projections.append(nn.functional.grid_sample(
                    input=self.planes(d),
                    grid=grid,
                    mode=self.plane_interp,
                    align_corners=self.align_corners,
                    padding_mode='border',
                ))
        projections = self.sum_or_cat(projections)
        return projections.squeeze(0).squeeze(-1).permute(1,0)

    def project_viewdir(self,dirs):
        grid = dirs.reshape([1,dirs.shape[0],1,2])
        # self.update_planes_coverage(self.N_PLANES_DENSITY,grid)
        return nn.functional.grid_sample(
                input=self.planes(self.N_PLANES_DENSITY),
                grid=grid,
                mode=self.plane_interp,
                align_corners=self.align_corners,
                padding_mode='border',
            ).squeeze(0).squeeze(-1).permute(1,0)

    def sum_or_cat(self,tensors):
        return torch.stack(tensors,0).sum(0) if self.proj_combination=='sum' else torch.cat(tensors,1)

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
        for l in self.density_dec:
            x = self.relu(l(x))
        alpha = self.fc_alpha(x)

        # x_rgb = torch.zeros([len(projected_xyz),projected_xyz.shape[1] if self.proj_combination=='sum' else 0]).type(projected_xyz.type())
        if 'features' in self.rgb_dec_input:
            x_rgb = self.fc_feat(x)

        if self.rgb_dec_input=='projections_features':
            x_rgb = self.sum_or_cat([x_rgb,projected_xyz])
        elif self.rgb_dec_input=='projections':
            x_rgb = 1*projected_xyz

        if self.use_viewdirs:
            x_rgb = self.sum_or_cat([x_rgb,projected_views])

        for l in self.rgb_dec:
            x_rgb = self.relu(l(x_rgb))
        rgb = self.fc_rgb(x_rgb)

        return torch.cat((rgb, alpha), dim=-1)


# EDSR code taken and modified from https://github.com/twtygqyy/pytorch-edsr/blob/master/edsr.py
class _Residual_Block(nn.Module): 
    def __init__(self,hidden_size,padding,kernel_size):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

    def forward(self, x): 
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output 

class EDSR(nn.Module):
    def __init__(self,scale_factor,in_channels,out_channels,hidden_size,plane_interp,n_blocks=32):
        super(EDSR, self).__init__()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        self.scale_factor = scale_factor
        self.plane_interp = plane_interp
        PADDING,KERNEL_SIZE = 1,3
        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False)
        self.required_padding,rf_factor = KERNEL_SIZE/2,1

        self.residual = self.make_layer(_Residual_Block,{'hidden_size':hidden_size,'padding':PADDING,'kernel_size':KERNEL_SIZE}, n_blocks)
        self.required_padding += rf_factor*2*n_blocks*(KERNEL_SIZE-1)/2

        self.conv_mid = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False)
        self.required_padding += rf_factor*(KERNEL_SIZE-1)/2
        assert math.log2(scale_factor)==int(math.log2(scale_factor)),"Supperting only scale factors that are an integer power of 2."
        upscaling_layers = []
        for _ in range(int(math.log2(scale_factor))):
            upscaling_layers += [
                nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*4, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False),
                nn.PixelShuffle(2),
            ]
            self.required_padding += rf_factor*(KERNEL_SIZE-1)/2
            rf_factor /= 2

        self.upscale = nn.Sequential(*upscaling_layers)

        self.conv_output = nn.Conv2d(in_channels=hidden_size, out_channels=out_channels, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False)
        self.required_padding += rf_factor*(KERNEL_SIZE-1)/2
        self.required_padding = int(np.ceil(self.required_padding))
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

        self.LR_planes,self.residual_planes,self.SR_planes = {},{},{}
        # self.SR_updated = {},

    def make_layer(self, block,args, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(**args))
        return nn.Sequential(*layers)

    def set_LR_planes(self,plane,id,align_corners):
        assert id not in self.LR_planes,"Plane ID already exists."
        self.LR_planes[id] = plane
        self.residual_planes[id] = torch.nn.functional.interpolate(plane,scale_factor=self.scale_factor,mode=self.plane_interp,align_corners=align_corners)
        self.SR_planes[id] = 1*self.residual_planes[id]
        # print('WARNING!!!!!!!!!!!!!!!!!!!!!!!! uncomment above')

    def weights_updated(self):
        self.SR_planes = {}

    def forward(self, plane_name):
        if plane_name in self.SR_planes:
            out = self.SR_planes[plane_name]
        else:
            x = self.LR_planes[plane_name].detach()
            success,num_batches = False,1
            take_last = lambda ind: -ind if ind>0 else None
            while not success:
                spatial_dims = np.array([int(np.ceil(v/num_batches)) for v in  x.shape[2:]])
                try:
                    out = torch.zeros_like(self.residual_planes[plane_name])
                    for b_num in range(num_batches):
                        min_index = b_num*spatial_dims
                        max_index = np.minimum((b_num+1)*spatial_dims,np.array(x.shape[2:]))
                        min_index[1],max_index[1] = 0,x.shape[3]
                        pre_padding = np.minimum(min_index,self.required_padding)
                        post_padding = np.minimum(np.array(x.shape[2:])-max_index,self.required_padding)
                        difference = self.inner_forward(
                            x[...,min_index[0]-pre_padding[0]:max_index[0]+post_padding[0],
                            min_index[1]-pre_padding[1]:max_index[1]+post_padding[1]])
                        min_index *= self.scale_factor
                        max_index *= self.scale_factor
                        out[...,min_index[0]:max_index[0],min_index[1]:max_index[1]] = torch.add(
                            difference[...,self.scale_factor*pre_padding[0]:take_last(self.scale_factor*post_padding[0]),
                                self.scale_factor*pre_padding[1]:take_last(self.scale_factor*post_padding[1])],
                            self.residual_planes[plane_name][...,min_index[0]:max_index[0],min_index[1]:max_index[1]])
                    success = True
                except Exception as e:
                    if 'CUDA out of memory.' in e.args[0]:
                        num_batches += 1
                    else:
                        raise e
            # out = self.conv_input(x)
            # out = self.conv_mid(self.residual(out))

            # out = self.upscale(out)
            # out = self.conv_output(out)
            # out = torch.add(out,self.residual_planes[plane_name])
            self.SR_planes[plane_name] = out
        return out

    def inner_forward(self,x):
        out = self.conv_input(x)
        out = self.conv_mid(self.residual(out))
        out = self.upscale(out)
        return self.conv_output(out)
        # return torch.add(out,residual)
