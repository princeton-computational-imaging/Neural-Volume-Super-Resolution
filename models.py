from collections import defaultdict,OrderedDict
from distutils.log import debug
import torch
import torch.nn as nn
from nerf_helpers import cart2az_el,rgetattr,rsetattr,safe_saving,safe_loading,downsample_plane
import math
import numpy as np
from scipy.interpolate import griddata
from re import search
import os
from tqdm import tqdm
from shutil import copyfile

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
                )
            else:
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

def plane_name2scene(plane_name):
    return search('(?<=sc).*(?=_D)',plane_name).group(0)

class TwoDimPlanesModel(nn.Module):
    def __init__(
        self,
        use_viewdirs,
        dec_density_layers=4,
        dec_rgb_layers=4,
        dec_channels=128,
        skip_connect_every=None,
        num_plane_channels=48,
        num_viewdir_plane_channels=None,
        rgb_dec_input='projections',
        proj_combination='sum',
        plane_interp='bilinear',
        align_corners=True,
        interp_viewdirs=None,
        viewdir_proj_combination=None,
        num_planes_or_rot_mats=3,
        plane_stats=False,
        detach_LR_planes=False,
        scene_coupler=None,
        point_coords_noise=0,
        ensemble_size=1,
    ):
        self.num_density_planes = num_planes_or_rot_mats if isinstance(num_planes_or_rot_mats,int) else len(num_planes_or_rot_mats)

        super(TwoDimPlanesModel, self).__init__()
        self.box_coords = None
        self.use_viewdirs = use_viewdirs
        assert use_viewdirs or (viewdir_proj_combination is None and num_viewdir_plane_channels is None)
        self.point_coords_noise = point_coords_noise
        self.num_plane_channels = num_plane_channels
        if num_viewdir_plane_channels is None:
            num_viewdir_plane_channels = num_plane_channels if use_viewdirs else 0
        self.num_viewdir_plane_channels = num_viewdir_plane_channels
        self.plane_stats = plane_stats
        self.detach_LR_planes = detach_LR_planes
        assert interp_viewdirs in ['bilinear','bicubic',None]
        assert interp_viewdirs is None,'Depricated'
        self.interp_from_learned = interp_viewdirs
        self.align_corners = align_corners
        assert rgb_dec_input in ['projections','features','projections_features']
        self.rgb_dec_input = rgb_dec_input
        assert proj_combination in ['sum','concat','avg']
        if viewdir_proj_combination is None:    viewdir_proj_combination = proj_combination
        assert viewdir_proj_combination in ['sum','concat','avg','mult','concat_pos']
        if num_viewdir_plane_channels!=num_plane_channels:
            assert 'concat' in viewdir_proj_combination
        self.proj_combination = proj_combination
        self.viewdir_proj_combination = viewdir_proj_combination
        self.plane_interp = plane_interp
        self.skip_connect_every = skip_connect_every # if skip_connect_every is not None else max(dec_rgb_layers,dec_density_layers)
        self.coord_projector = CoordProjector(self.num_density_planes,rot_mats=None if isinstance(num_planes_or_rot_mats,int) else num_planes_or_rot_mats)
        self.scene_coupler = scene_coupler

        # Density (alpha) decoder:
        self.density_dec = nn.ModuleDict([(str(i),nn.ModuleList()) for i in range(ensemble_size)])
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

        # RGB decoder:
        self.rgb_dec = nn.ModuleDict([(str(i),nn.ModuleList()) for i in range(ensemble_size)])
        plane_C_mult = 0
        if proj_combination=='concat' or viewdir_proj_combination=='concat_pos':  plane_C_mult += self.num_density_planes

        for i in range(ensemble_size):
            self.rgb_dec[str(i)].append(nn.Linear(num_viewdir_plane_channels+num_plane_channels*plane_C_mult,dec_channels))
            for layer_num in range(dec_rgb_layers-1):
                if self.is_skip_layer(layer_num=layer_num):
                    self.rgb_dec[str(i)].append(nn.Linear(num_viewdir_plane_channels+num_plane_channels*plane_C_mult + dec_channels, dec_channels))
                else:
                    self.rgb_dec[str(i)].append(nn.Linear(dec_channels,dec_channels))
        self.fc_rgb = nn.ModuleDict([(str(i),nn.Linear(dec_channels,3)) for i in range(ensemble_size)])

        self.relu = nn.functional.relu

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

    def gen_plane(self,plane_name,detach=False):
        if self.plane_rank is None or plane_name not in self.plane_rank:
            plane = self.planes_[plane_name]
        else:
            if plane_name not in self.generated_planes:
                self.generated_planes[plane_name] = torch.matmul(self.planes_[plane_name][...,:self.plane_rank[plane_name]],
                    self.planes_[plane_name][...,self.plane_rank[plane_name]:].permute([0,1,3,2]))
            plane = self.generated_planes[plane_name]
        return plane.detach() if detach else plane

    def raw_plane(self,plane_name,downsample=False,detach=False):
        plane = self.gen_plane(plane_name,detach=detach)
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

    def assign_SR_model(self,SR_model,SR_viewdir):
        self.SR_model = SR_model
        self.SR_model.align_corners = self.align_corners
        self.SR_model.SR_viewdir = SR_viewdir
        self.skip_SR_ = False
        assert not SR_viewdir,'Ceased supporting this option'

    def set_cur_scene_id(self,scene_id):
        self.cur_id = scene_id

    def normalize_coords(self,coords):
        EPSILON = 1e-5
        scene_name = self.cur_id+''
        normalized_coords = 2*(coords-self.box_coords[scene_name].type(coords.type())[:1])/\
            (self.box_coords[scene_name][1:]-self.box_coords[scene_name][:1]).type(coords.type())-1
        self.debug['max_norm'][scene_name] = np.maximum(self.debug['max_norm'][scene_name],normalized_coords.max(0)[0].cpu().numpy())
        self.debug['min_norm'][scene_name] = np.minimum(self.debug['min_norm'][scene_name],normalized_coords.min(0)[0].cpu().numpy())
        return normalized_coords

    def planes(self,dim_num:int,super_resolve:bool,grid:torch.tensor=None)->torch.tensor:
        plane_name = get_plane_name(self.cur_id,dim_num)
        detach = self.detach_LR_planes and plane_name2scene(plane_name) in self.scene_coupler.downsample_couples
        downsample_plane = self.scene_coupler.should_downsample(plane_name)
        plane_name = self.scene_coupler.scene_with_saved_plane(plane_name,plane_not_scene=True)
        if super_resolve:
            SR_input = plane_name
            if grid is not None and self.SR_model.training:
                roi = torch.stack([grid.min(1)[0].squeeze(),grid.max(1)[0].squeeze()],0)
                roi = torch.stack([roi[:,1],roi[:,0]],1) # Converting from (x,y) to (y,x) on the columns dimension
                SR_input = (SR_input,roi)
            plane = self.SR_model(SR_input)
        else:
            plane = self.raw_plane(plane_name,downsample_plane,detach=detach)
        return plane.cuda()

    def skip_SR(self,skip):
        self.skip_SR_ = skip

    def project_xyz(self,coords):
        projections = []
        if self.point_coords_noise and self.training:
            projected_to_res = int(search('(?<=PlRes)(\d)+(?=_)',self.cur_id).group(0))
            coords = coords+torch.normal(mean=0,std=self.point_coords_noise*2/(1+projected_to_res),size=coords.shape).type(coords.type())
        for d in range(self.num_density_planes):
            grid = self.coord_projector((coords,d)).reshape([1,coords.shape[0],1,2])
            if self.plane_stats and self.training:
                self.plane_coverage(grid,d)
            plane_name = get_plane_name(self.cur_id,d)
            # Check whether the planes SHOULD be super-resolved:
            super_resolve = hasattr(self,'SR_model') and (not hasattr(self,'scene_coupler') or self.scene_coupler.should_SR(plane_name,plane_not_scene=True))
            # Check whether the planes CAN be super-resolved:
            super_resolve = super_resolve and not self.skip_SR_
            projections.append(nn.functional.grid_sample(
                    input=self.planes(d,super_resolve=super_resolve,grid=grid),
                    grid=grid,
                    mode=self.plane_interp,
                    align_corners=self.align_corners,
                    padding_mode='border',
                ))
        return [p.squeeze(0).squeeze(-1).permute(1,0) for p in projections]

    def project_viewdir(self,dirs):
        grid = dirs.reshape([1,dirs.shape[0],1,2])
        plane_name = get_plane_name(self.cur_id,self.num_density_planes)
        # Stopped supporting viewdir-planes SR:
        super_resolve = False
        assert not super_resolve,'Unexpected'
        if self.plane_stats and self.training:
            self.plane_coverage(grid,self.num_density_planes)
        return nn.functional.grid_sample(
                input=self.planes(self.num_density_planes,super_resolve=super_resolve,grid=grid),
                grid=grid,
                mode=self.plane_interp,
                align_corners=self.align_corners,
                padding_mode='border',
            ).squeeze(0).squeeze(-1).permute(1,0)

    def plane_coverage(self,grid,d):
        APPROX_COV = True
        plane_name = get_plane_name(self.cur_id,d)
        if plane_name not in self.coverages:    return
        plane_res = self.raw_plane(plane_name).shape[3]
        logging_res = self.coverages[plane_name].shape[0]
        covered_points = (grid/2*plane_res).squeeze()+logging_res/2
        if APPROX_COV:
            covered_points = torch.round(covered_points).type(torch.LongTensor)
            covered_points = torch.unique(covered_points,dim=0)
            covered_points = torch.clamp(covered_points,min=0,max=self.coverages[plane_name].shape[0]-1)
            self.coverages[plane_name][covered_points[:,0],covered_points[:,1]] += 1
        else:
            floor_int = lambda x:torch.floor(x).type(torch.LongTensor)
            ceil_int = lambda x:torch.ceil(x).type(torch.LongTensor)
            for row_f in [floor_int,ceil_int]:
                for col_f in [floor_int,ceil_int]:
                    for p in (covered_points[::64] if d==self.num_density_planes else covered_points):
                        self.coverages[plane_name][row_f(p[0]),col_f(p[1])] += 1
        import matplotlib.pyplot as plt
        plt.imsave('coverage/plane_coverage_%s.png'%(plane_name),np.log(self.coverages[plane_name].cpu().numpy()+1))
        plt.clf()
        plt.plot(self.coverages[plane_name].mean(0))
        plt.plot(self.coverages[plane_name].mean(1))
        plt.savefig('coverage/%s_coverage.png'%(plane_name))


    def combine_pos_planes(self,tensors):
        if self.proj_combination=='sum':
            return torch.stack(tensors,0).sum(0)  
        elif self.proj_combination=='avg':
            return torch.stack([t for i,t in enumerate(tensors) if i not in self.planes2drop],0).mean(0)
        elif self.proj_combination=='concat':
            return torch.cat(tensors,1)

    def combine_all_planes(self,pos_planes,viewdir_planes):
        if self.viewdir_proj_combination!='concat_pos':
            pos_planes = self.combine_pos_planes(pos_planes)
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
        elif self.viewdir_proj_combination=='concat_pos':
            return torch.cat(pos_planes+[viewdir_planes],1)

    def forward(self, x):
        if self.use_viewdirs:
            x = torch.cat([x[...,:3],cart2az_el(x[...,3:])],-1)
        else:
            x = x[..., : 3]
        x = self.normalize_coords(x)
        pos_projections = self.project_xyz(x[..., : 3])
        projected_xyz = self.combine_pos_planes(pos_projections)

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
            raise Exception('Depricated')
            x_rgb = self.combine_pos_planes([x_rgb,projected_xyz])
        elif self.rgb_dec_input=='projections':
            x_rgb = 1*pos_projections

        if self.use_viewdirs:
            x_rgb = self.combine_all_planes(pos_planes=x_rgb,viewdir_planes=projected_views)

        x = x_rgb
        for layer_num,l in enumerate(self.rgb_dec[model_num]):
            if self.is_skip_layer(layer_num=layer_num-1):
                x = torch.cat((x, x_rgb), dim=-1)
            x = self.relu(l(x))
        rgb = self.fc_rgb[model_num](x)

        return torch.cat((rgb, alpha), dim=-1)

    def downsample_plane(self,plane,antialias=False):
        return downsample_plane(plane,ds_factor=self.scene_coupler.ds_factor,plane_interp=self.plane_interp,alilgn_corners=self.align_corners,antialias=antialias)

    def assign_LR_planes(self,scene=None):
        for k in self.planes_:
            if scene is not None and self.scene_coupler.scene2saved[scene] not in k:    continue
            if not self.SR_model.SR_viewdir and get_plane_name(None,self.num_density_planes) in k:  continue
            if self.scene_coupler.should_downsample(k,for_LR_loading=True):
                LR_plane = self.downsample_plane(self.raw_plane(k,detach=self.detach_LR_planes))
            else:
                LR_plane = self.raw_plane(k,detach=self.detach_LR_planes)
            self.SR_model.set_LR_plane(LR_plane,id=k,save_interpolated=False)

def create_plane(resolution,num_plane_channels,init_STD):
    if not isinstance(resolution,list):
        resolution = [resolution,resolution]
    return nn.Parameter(init_STD*torch.randn(size=[1,num_plane_channels,resolution[0],resolution[1]]))

class SceneSampler:
    def __init__(self,scenes:list,do_when_reshuffling=lambda:None,frozen_scenes:list=[]) -> None:
        self.scenes = scenes
        self.frozen_scenes = frozen_scenes
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
                if cursor>=len(self.sample_from):
                    self.shuffle()
                    cursor = 0
                if self.sample_from[cursor] in sampled or (len(sampled)==n-1 and self.sample_from[cursor] in self.frozen_scenes and all([sc in self.frozen_scenes for sc in sampled])):
                    cursor += 1
                else:
                    sampled.append(self.sample_from.pop(cursor))
        return sampled

class CoordProjector(nn.Module):
    def __init__(self,N:int=None,rot_mats:nn.Parameter=None) -> None:
        super(CoordProjector,self).__init__()
        N_RANDOM_TRIALS = 10000
        if rot_mats is None:
            if N<=3: #  For the basic case, conforming with the previous convention of the standard basis:
                base_mat = torch.eye(3)
                self.rot_mats_NON_LEARNED = nn.ParameterList([torch.nn.Parameter(p) for p in  [base_mat,base_mat[:,[1,0,2]],base_mat[:,[2,0,1]]][:N]])
            else:
                plane_axes = np.random.uniform(low=-1,high=1,size=[N_RANDOM_TRIALS,N,3])
                plane_axes /= np.sqrt(np.sum(plane_axes**2,2,keepdims=True))
                plane_axes = np.concatenate((plane_axes,-1*plane_axes),1)
                chosen = plane_axes[np.argmax(np.sum(np.sort(np.sum((plane_axes[...,None,:]-np.expand_dims(plane_axes,1))**2,-1),1)[:,1,...],-1))][:N]
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
            available_scenes:list=[],planes_rank_ratio:float=None,copy_params_path=None,run_time_signature:float=np.nan,
            lr_scheduler=None,use_frozen_planes='',
            ) -> None:
        super(PlanesOptimizer,self).__init__()
        self.scenes = available_scenes
        self.run_time_signature = run_time_signature
        if training_scenes is None:
            training_scenes = 1*self.scenes
        self.training_scenes = training_scenes
        self.frozen_scene_paths = dict()
        if use_frozen_planes!='':
            for sc in training_scenes:
                lr_scene = sc.replace(str(max(model_fine.scene_coupler.plane_res_vals)),str(min(model_fine.scene_coupler.plane_res_vals))).replace('DS%d'%(min(model_fine.scene_coupler.ds_vals)),'DS%d'%(max(model_fine.scene_coupler.ds_vals)))
                frozen_planes_path = self.param_path(model_name='coarse',scene=lr_scene,save_location=use_frozen_planes)
                if os.path.isfile(frozen_planes_path):
                    self.frozen_scene_paths.update({sc:frozen_planes_path,lr_scene:frozen_planes_path})
                    model_fine.scene_coupler.scene2saved[sc] = lr_scene
                    model_fine.scene_coupler.downsample_couples[sc] = lr_scene
        self.scenes_with_planes = scene_id_plane_resolution.keys()
        assert len(available_scenes)>0
        assert copy_params_path is None or not init_params,"Those two don't work together"
        self.buffer_size = getattr(options,'buffer_size',len(self.training_scenes))
        self.steps_per_buffer,self.steps_since_drawing = options.steps_per_buffer,0
        if self.buffer_size>=len(self.training_scenes):
            self.buffer_size = len(self.training_scenes)
            self.steps_per_buffer = -1
        assert all([s in self.scenes or model_fine.scene_coupler.downsample_couples[s] in self.scenes for s in self.training_scenes])
        assert optimizer_type=='Adam','Optimizer %s not supported yet.'%(optimizer_type)
        assert use_coarse_planes,'Unsupported yet, probably requires adding a param_group to the optimizer'
        assert not init_params or optimize,'This would means using (frozen) random planes...'
        assert self.steps_per_buffer==-1 or self.steps_per_buffer>=self.buffer_size,\
            'Trying to use %d steps for a buffer of size %d: Some scenes would be loaded in vain.'%(options.steps_per_buffer,self.buffer_size)
        self.scene_sampler = SceneSampler(self.training_scenes,do_when_reshuffling=do_when_reshuffling,frozen_scenes=list(self.frozen_scene_paths.keys()))
        self.models = {}
        self.use_coarse_planes = use_coarse_planes
        self.save_location = save_location
        self.lr = lr
        plane_rank_dict = None if planes_rank_ratio is None else dict([(get_plane_name(sc,d),int(np.ceil(planes_rank_ratio*res[0]))) for sc,res in scene_id_plane_resolution.items() for d in range(model_coarse.num_density_planes)])
        self.generated_planes = {}
        self.downsampled_planes = {}
        coverages = {}
        for model_name,model in zip(['coarse','fine'],[model_coarse,model_fine]):
            self.models[model_name] = model
            model.plane_rank = plane_rank_dict
            model.generated_planes = self.generated_planes
            model.downsampled_planes = self.downsampled_planes
            model.coverages = coverages
            if model_name=='fine' and use_coarse_planes:    continue
            self.planes_per_scene = model.num_density_planes+model.use_viewdirs
            if init_params or copy_params_path:
                for scene,res in tqdm([(k,v) for k,v in scene_id_plane_resolution.items() if k not in self.frozen_scene_paths],desc='Initializing scene planes',):
                    if init_params:
                        params = nn.ParameterDict([
                            (get_plane_name(scene,d),
                                create_plane(res[0] if d<model.num_density_planes else res[1],
                                num_plane_channels=model.num_viewdir_plane_channels if d==model.num_density_planes else model.num_plane_channels,
                                init_STD=STD_factor*model.fc_alpha['0'].weight.data.std().cpu())
                                if plane_rank_dict is None or d>=model.num_density_planes else
                                create_plane([res[0],2*plane_rank_dict[get_plane_name(scene,d)]],
                                num_plane_channels=model.num_viewdir_plane_channels if d==model.num_density_planes else model.num_plane_channels,
                                init_STD=np.sqrt(STD_factor*model.fc_alpha.weight.data.std().cpu()))
                            )
                            for d in range(self.planes_per_scene)])
                        cn = coords_normalization[scene]
                    else: # copy_params_path:
                        params = self.load_scene_planes(model_name=model_name,scene=scene,save_location=copy_params_path,prefer_best=True)
                        cn = params['coords_normalization']
                        params = params['params']
                    if not os.path.isdir(self.save_location[-1].replace('/planes/','/')) or any(['.ckpt' in f for f in os.listdir(self.save_location[-1].replace('/planes/','/'))]):
                        assert not os.path.exists(self.param_path(model_name=model_name,scene=scene)),"Planes scene file %s already exists"%(self.param_path(model_name=model_name,scene=scene))
                    torch.save({'params':params,'coords_normalization':cn},self.param_path(model_name=model_name,scene=scene))
                    if model.plane_stats:
                        coverages.update(
                            dict([(get_plane_name(scene,d),torch.zeros([res[1 if d==model.num_density_planes else 0]+15,res[1 if d==model.num_density_planes else 0]+15])) for d in range(model.num_density_planes+model.use_viewdirs)])
                        )

        self.optimize = optimize
        self.optimizer = None
        self.lr_scheduler = lr_scheduler if lr_scheduler else None
        self.saving_needed = False

    def lr_scheduler_step(self,loss):
        if self.optimizer is not None and self.lr_scheduler is not None:
            self.lr_scheduler.step(loss)

    def load_scene(self,scene,load_best=False):
        if self.saving_needed:
            self.save_params()
        for model_name in ['coarse','fine']:
            model = self.models[model_name]
            scenes_planes_name = model.scene_coupler.scene2saved[scene]
            if model_name=='coarse' or not self.use_coarse_planes:
                loaded_params = self.load_scene_planes(model_name=model_name,scene=scenes_planes_name,prefer_best=load_best)
                if scene not in self.frozen_scene_paths and not all([model.scene_coupler.scene2saved[scene] in k for k in loaded_params['params']]):
                    print('!!! Warning: Applying patch designed for the preliminary-SR baseline experiment !!!')
                    assert(all([model.scene_coupler.scene2saved[scene].replace('_DS2_','_DS1_') in k for k in loaded_params['params']]))
                    loaded_params['params'] = nn.ParameterDict([(k.replace('_DS1_','_DS2_'),v) for k,v in loaded_params['params'].items()])
            model.planes_ = loaded_params['params'].cuda()
            self.generated_planes.clear()
            self.downsampled_planes.clear()
            model.box_coords = {scenes_planes_name:loaded_params['coords_normalization'],scene:loaded_params['coords_normalization']}
            if hasattr(model,'SR_model'):
                model.SR_model.clear_SR_planes(all_planes=True)
                if model.scene_coupler.should_SR(scene):
                    model.assign_LR_planes()

        self.cur_scenes = [scene]

    def param_path(self,model_name,scene,save_location=None,prefer_best=False):
        path = lambda loc:  os.path.join(loc,"%s_%s.par"%(model_name,scene))
        if save_location is None:
            for loc in self.save_location:
                if os.path.isfile(path(loc).replace('.par','.par_best') if prefer_best else path(loc)):
                    break
            save_location = loc
        return path(save_location)

    def get_plane_stats(self,viewdir=False):
        model_name='coarse'
        plane_means,plane_STDs = [],[]
        for scene in tqdm(self.training_scenes,desc='Collecting plane statistics'):
            loaded_params = self.load_scene_planes(model_name=model_name,scene=self.models[model_name].scene_coupler.scene2saved[scene],prefer_best=True)
            for k,p in loaded_params['params'].items():
                if not viewdir and get_plane_name(None,self.models[model_name].num_density_planes) in k:  continue
                plane_means.append(torch.mean(p,(2,3)).squeeze(0))
                plane_STDs.append(torch.std(p.reshape(p.shape[1],-1),1))
        return {'mean':torch.stack(plane_means,0).mean(0),'std':torch.stack(plane_STDs,0).mean(0)}

    def save_params(self,as_best=False):
        assert self.optimize,'Why would you want to save if not optimizing?'
        model_name = 'coarse'
        model = self.models[model_name]
        scenes_list = self.training_scenes if as_best else self.cur_scenes
        scene_num = 0
        already_saved = []
        scenes2save_ = [sc for sc in scenes_list if sc not in self.frozen_scene_paths]
        scenes2save = []
        for sc in scenes2save_:
            if model.scene_coupler.scene_with_saved_plane(sc) not in scenes2save:
                scenes2save.append(model.scene_coupler.scene_with_saved_plane(sc))
        scenes2save = scenes2save if len(scenes2save)<20 else tqdm(scenes2save,desc='Saving scene planes')
        for scene in scenes2save:
            scene = model.scene_coupler.scene_with_saved_plane(scene)
            if scene in already_saved: continue
            already_saved.append(scene)
            if scene in self.cur_scenes:
                params = nn.ParameterDict([(get_plane_name(scene,d),model.planes_[get_plane_name(scene,d)]) for d in range(self.planes_per_scene)])
                opt_states = [self.optimizer.state_dict()['state'][i+scene_num*self.planes_per_scene] if (i+scene_num*self.planes_per_scene) in self.optimizer.state_dict()['state'] else None for i in range(self.planes_per_scene)]
                coords_normalization = model.box_coords[scene]
                scene_num += 1
            else:
                loaded_params = self.load_scene_planes(model_name=model_name,scene=scene,prefer_best=False)
                params = loaded_params['params']
                opt_states = loaded_params['opt_states'] if 'opt_states' in loaded_params else [None for p in params]
                coords_normalization = loaded_params['coords_normalization']
            param_file_name = self.param_path(model_name=model_name,scene=scene)
            safe_saving(param_file_name,content={'params':params,'opt_states':opt_states,'coords_normalization':coords_normalization},
                suffix='par',best=as_best,run_time_signature=self.run_time_signature)
        if not as_best: self.saving_needed = False

    def load_scene_planes(self,model_name,scene,prefer_best,save_location=None):
        if scene in self.frozen_scene_paths:
            file2load = self.frozen_scene_paths[scene]
            prefer_best = True
        else:
            file2load = self.param_path(model_name=model_name,scene=scene,save_location=save_location,prefer_best=prefer_best)
        loaded_params = safe_loading(file2load,suffix='par',best=prefer_best)
        return loaded_params


    def draw_scenes(self,assign_LR_planes=True):
        if self.saving_needed:
            self.save_params()
        self.steps_since_drawing = 0
        self.cur_scenes = self.scene_sampler.sample(self.buffer_size,just_shuffle=self.steps_per_buffer==-1)
        for model_name in ['coarse','fine']:
            model = self.models[model_name]
            if model_name=='coarse' or not self.use_coarse_planes:
                params_dict,optimizer_states,box_coords = nn.ParameterDict(),[],{}
                already_loaded = []
                scenes2load = self.cur_scenes if len(self.cur_scenes)<20 else tqdm(self.cur_scenes,desc='Loading scene planes')
                for scene in scenes2load:
                    if scene not in self.frozen_scene_paths:
                        scene = model.scene_coupler.scene_with_saved_plane(scene)
                    if scene in already_loaded: continue
                    already_loaded.append(scene)
                    loaded_params = self.load_scene_planes(model_name=model_name,scene=scene,prefer_best=not self.optimize)
                    params_dict.update(loaded_params['params'])
                    box_coords.update(dict([(sc,loaded_params['coords_normalization']) for sc in [scene]+model.scene_coupler.coupled_scene(scene)]))
                    if self.optimize and scene not in self.frozen_scene_paths:
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
                params = list([v for i,v in enumerate(params_dict.values()) if self.cur_scenes[i//(model.num_density_planes+1)] not in self.frozen_scene_paths])
                if self.optimizer is None: # First call to this function:
                    self.optimizer = torch.optim.Adam(params, lr=self.lr)
                    if self.lr_scheduler is not None:
                        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=self.lr_scheduler['patience'],factor=self.lr_scheduler['factor'],verbose=True)
                else:
                    self.optimizer.param_groups[0]['params'] = params
                self.optimizer.state = defaultdict(dict,[(params[i],v) for i,v in enumerate(optimizer_states) if v is not None])
        self.saving_needed = False

    def step(self):
        if self.optimize:
            if self.cur_id not in self.frozen_scene_paths:
                self.optimizer.step()
                self.saving_needed = True
            self.generated_planes.clear()
            self.downsampled_planes.clear()
        self.steps_since_drawing += 1
        if hasattr(self.models['fine'],'SR_model'):
            self.models['fine'].SR_model.clear_SR_planes(all_planes=self.optimize)

        if self.steps_since_drawing==self.steps_per_buffer:
            self.draw_scenes(assign_LR_planes=not self.optimize)
            return self.cur_scenes
        else:
            return None

    def zero_grad(self):
        if self.optimize and self.cur_id not in self.frozen_scene_paths:   self.optimizer.zero_grad()

    def jump_start(self,config=None,on=True):
        items2memorize = ['steps_per_buffer']
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
    def __init__(self,in_channels,out_channels,hidden_size,n_blocks,scale_factor,padding,receptive_field_bound=np.iinfo(np.int32).max,**kwargs):
        super(EDSR, self).__init__()
        KERNEL_SIZE = 3
        self.required_padding,rf_factor = 0,1

        def kernel_size(num_layers=1):
            if (1+2*(self.required_padding+rf_factor*num_layers*((KERNEL_SIZE-1)//2)))<=receptive_field_bound:
                self.required_padding += rf_factor*num_layers*(KERNEL_SIZE//2)
                return KERNEL_SIZE
            else:
                return 1

        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=kernel_size(), stride=1, padding=padding, bias=False)
        self.residual = nn.Sequential()
        for _ in range(n_blocks):
            self.residual.append(_Residual_Block(hidden_size=hidden_size,padding=padding,kernel_size=kernel_size(2)))
        self.conv_mid = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size(), stride=1, padding=padding, bias=False)
        assert math.log2(scale_factor)==int(math.log2(scale_factor)),"Supperting only scale factors that are an integer power of 2."
        upscaling_layers = []
        for _ in range(int(math.log2(scale_factor))):
            upscaling_layers += [
                nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*4, kernel_size=kernel_size(), stride=1, padding=padding, bias=False),
                nn.PixelShuffle(2),
            ]
            rf_factor /= 2
        self.upscale = nn.Sequential(*upscaling_layers)
        self.conv_output = nn.Conv2d(in_channels=hidden_size, out_channels=out_channels, kernel_size=kernel_size(), stride=1, padding=padding, bias=False)

    def forward(self,x):
        out = self.conv_input(x)
        out = self.conv_mid(self.residual(out))
        out = self.upscale(out)
        return self.conv_output(out)

class PlanesSR(nn.Module):
    def __init__(self,model_arch,scale_factor,in_channels,out_channels,sr_config,plane_interp):
        super(PlanesSR, self).__init__()

        hidden_size = sr_config.model.hidden_size
        n_blocks = sr_config.model.n_blocks
        input_normalization = sr_config.get("input_normalization",False)
        self.scale_factor = scale_factor
        self.plane_interp = plane_interp
        self.input_noise = getattr(sr_config,'sr_input_noise',0)
        self.output_noise = getattr(sr_config,'sr_output_noise',0)
        self.per_channel_sr = getattr(sr_config.model,'per_channel_sr',False)
        if self.per_channel_sr:
            in_channels = out_channels = 1

        PADDING = 0
        self.inner_model = model_arch(in_channels=in_channels,out_channels=out_channels,hidden_size=hidden_size,n_blocks=n_blocks,
            scale_factor=scale_factor,padding=PADDING,receptive_field_bound=getattr(sr_config.model,'receptive_field_bound',np.iinfo(np.int32).max),
            edsr_init=getattr(sr_config.model,'edsr_init',False),no_bn=getattr(sr_config.model,'no_batch_norm',False))
        self.HR_overpadding = int(self.inner_model.required_padding*self.scale_factor)
        self.inner_model.required_padding = int(np.ceil(self.inner_model.required_padding))
        self.HR_overpadding = self.inner_model.required_padding*self.scale_factor-self.HR_overpadding
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
        self.LR_planes[id] = plane
        if save_interpolated:
            self.residual_planes[id] = self.interpolate_LR(id).cpu()

    def clear_SR_planes(self,all_planes=False):
        planes_2_clear = ['SR_planes']
        if all_planes:
            planes_2_clear += ['LR_planes','residual_planes']
        for attr in planes_2_clear:
            setattr(self,attr,{})

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
            pre_padding = np.minimum(plane_roi[0],self.inner_model.required_padding)
            post_padding = np.minimum(np.array(x.shape[2:])-plane_roi[1],self.inner_model.required_padding)
            take_last = lambda ind: -ind if ind>0 else None
            DEBUG = False
            if DEBUG:   print('!! WARNING !!!!')
            x = x[...,plane_roi[0,0]-pre_padding[0]:plane_roi[1,0]+post_padding[0],plane_roi[0,1]-pre_padding[1]:plane_roi[1,1]+post_padding[1]]
            x = torch.nn.functional.pad(x,
                pad=(self.inner_model.required_padding-pre_padding[1],self.inner_model.required_padding-post_padding[1],self.inner_model.required_padding-pre_padding[0],self.inner_model.required_padding-post_padding[0]),
                mode='replicate')
            if self.per_channel_sr:
                difference = torch.cat([self.inner_model(x[:,ch:ch+1,...]) for ch in range(x.shape[1])],1)[...,self.HR_overpadding:take_last(self.HR_overpadding),self.HR_overpadding:take_last(self.HR_overpadding)]
            else:
                difference = self.inner_model(x)[...,self.HR_overpadding:take_last(self.HR_overpadding),self.HR_overpadding:take_last(self.HR_overpadding)]
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
        return out

def get_scene_id(basedir,ds_factor,plane_res):
    return '%s_DS%d%s'%(basedir,ds_factor,'' if plane_res[0] is None else '_PlRes%d_%d'%(plane_res))

def extract_ds_and_res(scene_name):
    ds = int(search('(?<=_DS)(\d)+',scene_name).group(0))
    res = int(search('(?<=_PlRes)(\d)+(?=_)',scene_name).group(0)) if '_PlRes' in scene_name else None
    return ds,res

class SceneCoupler:
    def __init__(self,scenes_list,planes_res,num_pos_planes,training_scenes,multi_im_res) -> None:
        planes_model = num_pos_planes>0
        name_pattern = lambda name: '^'+name.split('_DS')[0]+'_DS'+ ('(\d)+_PlRes(\d)+_'+name.split('_')[-1] if planes_model else '')
        ds_ratios,res_ratios,res_vals,ds_vals = [],[],[],[]
        self.upsample_couples,self.downsample_couples, = {},{},
        scenes_list = list(set(scenes_list+training_scenes))
        assert planes_res in ['HR','LR','LRHR','HRLR','']
        if multi_im_res:
            for sc_num in range(len(scenes_list)):
                matching_scenes = [sc for sc in [s for i,s in enumerate(scenes_list) if i!=sc_num] if search(name_pattern(scenes_list[sc_num]),sc)]
                if len(matching_scenes)>0:
                    assert len(matching_scenes)==1 or 'HR' not in planes_res,'Not supporting HR planes with multiple scene matches'
                    org_vals = extract_ds_and_res(scenes_list[sc_num])
                    for match in matching_scenes:
                        found_vals = extract_ds_and_res(match)
                        res_vals.extend([found_vals[1],org_vals[1]])
                        res_ratio = found_vals[1]/org_vals[1] if planes_model else None
                        if res_ratio==1:
                            continue
                        res_ratios.append(res_ratio)
                        ds_ratios.append(found_vals[0]/org_vals[0])
                        ds_vals.extend([found_vals[0],org_vals[0]])
                        determining_ratio = res_ratios[-1] if planes_model else 1/ds_ratios[-1]
                        if determining_ratio<1:
                            if scenes_list[sc_num] in training_scenes:
                                self.upsample_couples[match] = scenes_list[sc_num]
                            self.downsample_couples[scenes_list[sc_num]] = match
                        elif determining_ratio>1:
                            self.downsample_couples[match] = scenes_list[sc_num]
                            if match in training_scenes:
                                self.upsample_couples[scenes_list[sc_num]] = match
        if len(self.downsample_couples)==0:
            self.ds_factor = 1
        else:
            self.plane_res_vals = set(res_vals)
            self.ds_vals = set(ds_vals)
            assert len(self.plane_res_vals)<=2,'Should look into this...'
            self.ds_factor = int(max(1/res_ratios[0],res_ratios[0])) if planes_model else int(max(1/ds_ratios[0],ds_ratios[0]))
        if planes_model:
            for match_num in range(len(ds_ratios)):
                if res_ratios[match_num]!=1/ds_ratios[match_num]:
                    assert ds_ratios[match_num]==1,"I expect to have the downsampling factor match the plane resolution ratio."
                assert res_ratios[match_num] in [self.ds_factor,1/self.ds_factor],"Not all plane resolution ratios/downsampling factors are the same"
        self.use_HR_planes = False
        if 'HR' in planes_res:
            raise Exception('Depricated.')
            self.HR_planes = [get_plane_name(k,d) for k in self.downsample_couples.keys() for d in range(num_pos_planes)]
            self.scene2saved = dict([(sc,self.upsample_couples[sc] if (sc in self.upsample_couples and 'LR' not in planes_res) else sc if sc in training_scenes else self.downsample_couples[sc])
                for sc in scenes_list])
        else:
            self.HR_planes = []
            self.scene2saved = dict([(sc,self.downsample_couples[sc] if sc in self.downsample_couples else sc) for sc in scenes_list])
        def plane2saved(plane_name):
            scene = plane_name2scene(plane_name)
            return plane_name.replace(scene,self.scene2saved[scene])
        self.plane2saved = plane2saved

    def coupled_scene(self,scene):
        couples = []
        if scene in self.downsample_couples:
            couples.append(self.downsample_couples[scene])
        if scene in self.upsample_couples:
            couples.append(self.upsample_couples[scene])
        assert len(couples)<=1,"Expecting to have up to 1 couple, since this function is called during training phase, where each actual scene should not have more than two 'virtual' scenes associated with it."
        return couples

    def scene_with_saved_plane(self,scene,plane_not_scene=False):
        if plane_not_scene:
            return self.plane2saved(scene)
        else:
            return self.scene2saved[scene]

    def should_SR(self,scene,plane_not_scene=False):
        if len(self.HR_planes)>0:
            return scene in (self.HR_planes if plane_not_scene else self.downsample_couples.keys())
        else:
            if plane_not_scene:
                return plane_name2scene(scene) in self.downsample_couples
            else:
                return scene in self.downsample_couples

    def should_downsample(self,plane_name,for_LR_loading=False):
        return len(self.HR_planes)>0 and self.plane2saved(plane_name) in self.HR_planes

class _ResidualConvBlock(nn.Module):
    """Implements residual conv function.

    Args:
        channels (int): Number of channels in the input image.
    """

    def __init__(self, channels: int,no_bn=False) -> None:
        super(_ResidualConvBlock, self).__init__()
        rcb_seq = [nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)]
        if no_bn==False:
            rcb_seq.insert(1,nn.BatchNorm2d(channels))
            rcb_seq.insert(4,nn.BatchNorm2d(channels))
        elif no_bn=='add_relu':
            rcb_seq.append(nn.ReLU(inplace=True))
        self.rcb = nn.Sequential(OrderedDict([(str(i),l) for i,l in enumerate(rcb_seq)]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        identity = x

        out = self.rcb(x)

        out = torch.add(out, identity)

        return out

class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, scale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * scale_factor * scale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.upsample_block(x)

        return out

class SRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_size: int,
            n_blocks: int,
            scale_factor: int,
            padding: int,
            receptive_field_bound: int,
            **kwargs
    ) -> None:
        super(SRResNet, self).__init__()
        # First conv layer.
        self.required_padding = 0
        no_bn = kwargs.get('no_bn',False)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # Features trunk blocks.
        trunk = []
        for _ in range(n_blocks):
            trunk.append(_ResidualConvBlock(hidden_size,no_bn))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer.
        conv_block2_seq = [nn.Conv2d(hidden_size, hidden_size, (3, 3), (1, 1), (1, 1), bias=False)]
        if no_bn==False:
            conv_block2_seq.append(nn.BatchNorm2d(hidden_size))
        self.conv_block2 = nn.Sequential(OrderedDict([(str(i),l) for i,l in enumerate(conv_block2_seq)]))

        # Upscale block
        upsampling = []
        if scale_factor == 2 or scale_factor == 4 or scale_factor == 8:
            for _ in range(int(math.log(scale_factor, 2))):
                upsampling.append(_UpsampleBlock(hidden_size, 2))
        elif scale_factor == 3:
            upsampling.append(_UpsampleBlock(hidden_size, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv_block3 = nn.Conv2d(hidden_size, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        self._initialize_weights(edsr_init=kwargs.get('edsr_init',False))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.tensor) -> torch.tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv_block3(out)


        return out

    def _initialize_weights(self,edsr_init=False) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if edsr_init:
                    n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                    module.weight.data.normal_(0, math.sqrt(2. / n)/10)
                    if module.bias is not None:
                        module.bias.data.zero_()
                else:
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                if edsr_init:
                    module.weight.data.fill_(1)
                    if module.bias is not None:
                        module.bias.data.zero_()
                else:
                    nn.init.constant_(module.weight, 1)
