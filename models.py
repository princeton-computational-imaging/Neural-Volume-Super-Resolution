import torch
import torch.nn as nn
from nerf_helpers import calc_scene_box,cart2az_el
import math

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
        scene_geometry,
        dec_density_layers=4,
        dec_rgb_layers=4,
        dec_channels=128,
        # skip_connect_every=4,
        num_plane_channels=48,
        rgb_dec_input='projections',
        planes=None,
        plane_interp='bilinear',
        align_corners=True,
        track_planes_coverage=False,
    ):
        super(TwoDimPlanesModel, self).__init__()
        # self.ds_2_res = dict([(v,k) for k,v in plane_resolutions.items()])
        self.box_coords = calc_scene_box(scene_geometry=scene_geometry,including_dirs=use_viewdirs)
        self.use_viewdirs = use_viewdirs
        self.align_corners = align_corners
        assert rgb_dec_input in ['sum','projections','features']
        self.rgb_dec_input = rgb_dec_input
        self.plane_interp = plane_interp
        self.track_planes_coverage = track_planes_coverage
        # Density (alpha) decoder:
        self.density_dec = nn.ModuleList()
        self.density_dec.append(
            nn.Linear(num_plane_channels,dec_channels)
        )
        for layer_num in range(dec_density_layers-1):
            self.density_dec.append(
                nn.Linear(dec_channels,dec_channels)
            )
        self.fc_alpha = nn.Linear(dec_channels,1)
        if self.rgb_dec_input in ['sum','features']:
            self.fc_feat = nn.Linear(dec_channels,num_plane_channels)

        # RGB decoder:
        self.rgb_dec = nn.ModuleList()
        self.rgb_dec.append(
            nn.Linear(num_plane_channels,dec_channels)
        )
        for layer_num in range(dec_rgb_layers-1):
            self.rgb_dec.append(
                nn.Linear(dec_channels,dec_channels)
            )
        self.fc_rgb = nn.Linear(dec_channels,3)

        self.relu = nn.functional.relu
        if planes is None:
            self.planes_ = nn.ParameterDict([
                (self.plane_name(id,d),
                nn.Parameter(self.fc_alpha.weight.data.std()*torch.randn(size=[1,num_plane_channels,res,res])))
                 for id,res in scene_id_plane_resolution.items() for d in range(3+use_viewdirs)])
            # self.planes_ = [nn.ParameterList(
            #     [nn.Parameter(self.fc_alpha.weight.data.std()*torch.randn(size=[1,num_plane_channels,res,res]
            #     )) for i in range(3+use_viewdirs)]) for res in plane_resolutions.keys()]
            if self.track_planes_coverage:
                raise Exception("Should be adapted to support multiple plane resolutions.")
                self.planes_coverage = torch.zeros([len(self.planes_),plane_resolutions,plane_resolutions,]).type(torch.bool)
        else:
            self.planes_ = planes

    def plane_name(self,scene_id,dimension):
        return "sc%s_D%d"%(scene_id,dimension)
        
    def assign_SR_model(self,SR_model):
        self.SR_model = SR_model
        for k,v in self.planes_.items():
            self.SR_model.set_LR_planes(v.detach(),id=k,align_corners=self.align_corners)

        # for id in self.model_ids:
        #     self.SR_model.set_LR_planes([p.detach() for p in self.planes_[id]],id=id)
        self.skip_SR_ = False

    def set_cur_scene_id(self,scene_id):
        self.cur_id = scene_id

    def normalize_coords(self,coords):
        EPSILON = 1e-5
        normalized_coords = 2*(coords-self.box_coords.type(coords.type())[:1])/\
            (self.box_coords[1:]-self.box_coords[:1]).type(coords.type())-1
        assert normalized_coords.min()>=-1-EPSILON and normalized_coords.max()<=1+EPSILON,"Sanity check"
        return normalized_coords

    def grid2plane_inds(self,grid):
        return (grid+1)/2*(self.planes_[0].shape[-1]-1)

    def update_planes_coverage(self,plane_ind:int,grid:torch.tensor):
        if self.track_planes_coverage and self.training:
            for r_func in [torch.ceil,torch.floor]:
                for c_func in [torch.ceil,torch.floor]:
                    self.planes_coverage[plane_ind,r_func(self.grid2plane_inds(grid[...,0])).type(torch.long),c_func(self.grid2plane_inds(grid[...,1])).type(torch.long)] = True

    def planes(self,dim_num:int)->torch.tensor:
        plane_name = self.plane_name(self.cur_id,dim_num)
        if hasattr(self,'SR_model') and not self.skip_SR_:
            plane = self.SR_model(plane_name)
        else:
            plane = self.planes_[plane_name]#[self.cur_res][dim_num]
        # plane = self.planes_[dim_num]
        # if hasattr(self,'SR_model') and not self.skip_SR_:
        #     plane = self.SR_model(plane)
        return plane

    def skip_SR(self,skip):
        self.skip_SR_ = skip

    def project_xyz(self,coords):
        projections = []
        for d in range(3): # (Currently not supporting viewdir input)
            grid = coords[:,[c for c in range(3) if c!=d]].reshape([1,coords.shape[0],1,2])
            self.update_planes_coverage(d,grid)
            projections.append(nn.functional.grid_sample(
                    input=self.planes(d),
                    grid=grid,
                    mode=self.plane_interp,
                    align_corners=self.align_corners,
                    padding_mode='border',
                ))
        projections = torch.sum(torch.stack(projections,0),0)
        return projections.squeeze(0).squeeze(-1).permute(1,0)

    def project_viewdir(self,dirs):
        grid = dirs.reshape([1,dirs.shape[0],1,2])
        self.update_planes_coverage(3,grid)
        return nn.functional.grid_sample(
                input=self.planes(3),
                grid=grid,
                mode=self.plane_interp,
                align_corners=self.align_corners,
                padding_mode='border',
            ).squeeze(0).squeeze(-1).permute(1,0)


    def forward(self, x):
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
        x_rgb = 0
        if self.rgb_dec_input in ['sum','features']:
            x_rgb = self.fc_feat(x)

        if self.rgb_dec_input in ['sum','projections']:
            x_rgb = x_rgb+projected_xyz

        if self.use_viewdirs:
            x_rgb = x_rgb+projected_views

        for l in self.rgb_dec:
            x_rgb = self.relu(l(x_rgb))
        rgb = self.fc_rgb(x_rgb)

        return torch.cat((rgb, alpha), dim=-1)


# EDSR code taken and modified from https://github.com/twtygqyy/pytorch-edsr/blob/master/edsr.py
class _Residual_Block(nn.Module): 
    def __init__(self,hidden_size):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1, bias=False)

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
        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block,{'hidden_size':hidden_size}, n_blocks)

        self.conv_mid = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1, bias=False)
        assert math.log2(scale_factor)==int(math.log2(scale_factor)),"Supperting only scale factors that are an integer power of 2."
        upscaling_layers = []
        for _ in range(int(math.log2(scale_factor))):
            upscaling_layers += [
                nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
            ]
        self.upscale = nn.Sequential(*upscaling_layers)

        self.conv_output = nn.Conv2d(in_channels=hidden_size, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

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

        self.LR_planes,self.residual_planes,self.SR_planes,self.SR_updated = {},{},{},{}

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
        self.SR_updated[id] = True

    def forward(self, plane_name):
        if self.training or not self.SR_updated[plane_name]:
            if self.training:   self.SR_updated = dict([(id,False) for id in self.SR_updated.keys()])
            x = self.LR_planes[plane_name].detach()
        # def forward(self, x):
            # residual = x
            out = self.conv_input(x)
            # residual = out
            out = self.conv_mid(self.residual(out))
            # out = torch.add(out,residual)
            out = self.upscale(out)
            out = self.conv_output(out)
            # out = torch.add(out,torch.nn.functional.interpolate(residual,size=tuple(out.shape[2:]),mode=self.plane_interp,align_corners=True))
            out = torch.add(out,self.residual_planes[plane_name])
            if not self.training:
                self.SR_planes[plane_name] = out
                self.SR_updated[plane_name] = True
        else:
            out = self.SR_planes[plane_name]
        return out
 