import torch
import torch.nn as nn
import numpy as np
import math

# from pytorch3d.transforms import Transform3d


def cast_rays(t_vals, origins, directions, radii, ray_shape):
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    gaussian_fn = conical_frustum_to_gaussian

    means, covs = gaussian_fn(directions, t0, t1, radii)

    means = means + origins[..., None, :]

    return means, covs


def conical_frustum_to_gaussian(d, t0, t1, base_radius):
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
    t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                        (3 * mu ** 2 + hw ** 2) ** 2)
    r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                (hw ** 4) / (3 * mu ** 2 + hw ** 2))
    return lift_gaussian(d, t_mean, t_var, r_var)


def lift_gaussian(d, t_mean, t_var, r_var):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = torch.maximum(torch.tensor(1e-10).type(d.type()), torch.sum(d ** 2, axis=-1, keepdims=True))

    d_outer_diag = d ** 2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag


def get_mean_covs_obj(ray_bundle, sampling_pts_w, intersection_mask, n_samples,
                      trafos_w2o, rots_w2o, scales_w2o, origins_o, ray_node_inter_idx,
                      random_sampling):
    # For KITTI
    dx = 0.00135
    # Get t_vals in world to aply mipnerf
    radii = dx * 2 / np.sqrt(12.)

    ray_o_world = ray_bundle.origins[intersection_mask[:-2]]
    t_uniform_world = torch.norm(sampling_pts_w - ray_o_world, dim=-1)

    ray_o_world = ray_o_world.transpose(1, 0)
    ray_d_world = ray_bundle.directions[intersection_mask[:-2]].transpose(1, 0)

    t_1 = t_uniform_world[:, -1]
    t_0 = t_uniform_world[:, 0]

    t_segments = (t_1 - t_0)[:, None] \
                    * torch.linspace(0, 1, n_samples + 1).to(t_0.device)[None, :] \
                    + t_0[..., None]
    if random_sampling:
        seg_lengths = (t_segments[:,1]-t_segments[:,0])[:,None]
        t_segments[:,1:-1] +=\
            torch.maximum(
                torch.minimum(
                    (seg_lengths/6*torch.randn_like(t_segments[:,1:-1])),
                    seg_lengths/2),
                -seg_lengths/2
            )
    t_vals = [torch.cat([t_segments[:, i, None], t_segments[:, i + 1, None]], dim=-1) for i in
              range(n_samples)]
    t_vals = torch.stack(t_vals)

    # Extract mean and Sigmas here
    means, covs = cast_rays(t_vals, origins=ray_o_world, directions=ray_d_world, radii=radii, ray_shape=None)

    # Transform means and sigma into box coordinates
    means = means.reshape(n_samples, -1, 3).transpose(1, 0)
    covs = covs.reshape(n_samples, -1, 3).transpose(1, 0)

    obj_z_vals_world = torch.norm(ray_o_world.transpose(1, 0) - means, dim=-1)

    means_obj = torch.zeros_like(means, device=means.device)
    covs_obj = torch.zeros_like(covs, device=covs.device)

    ordered_start_ray = 0
    for i, (trafo, rot, scale) in enumerate(zip(trafos_w2o, rots_w2o, scales_w2o)):
        ordered_end_ray = origins_o[i].shape[1] + ordered_start_ray

        # Just select intersections from rays in this frame
        ordered_fr_bool_mask = ray_node_inter_idx[1].ge(
            ordered_start_ray
        ) & ray_node_inter_idx[1].le(ordered_end_ray - 1)
        ordered_fr_mask = tuple(
            [
                ray_node_inter_idx[0][ordered_fr_bool_mask],
                ray_node_inter_idx[1][ordered_fr_bool_mask]
                - ordered_start_ray,
            ]
        )

        # Transform intersection means back into obj space
        mean_wobj = (
            Transform3d(matrix=trafo).compose(Transform3d(matrix=scale))
        )

        frame_means_obj = torch.zeros(
            [len(trafo), ordered_end_ray - ordered_start_ray, n_samples, 3],
            device=means.device,
        )
        frame_means_obj[ordered_fr_mask] = means[ordered_fr_bool_mask]
        frame_means_obj = mean_wobj.transform_points(
            frame_means_obj.view(len(trafo), -1, 3)
        ).view(len(trafo), -1, n_samples, 3)

        means_obj[ordered_fr_bool_mask] = frame_means_obj[ordered_fr_mask]

        # Compute Covariance in scaled object axis
        sigma_wobj = (
            Transform3d(matrix=rot).compose(Transform3d(matrix=scale))
        )
        sigma_wobj_matrix = sigma_wobj.get_matrix().transpose(2,1)
        sigma_wobj_matrix_diag = sigma_wobj_matrix[:, :3, :3] * torch.eye(3).type(sigma_wobj_matrix.type())
        sigma_wobj_matrix_diag = sigma_wobj_matrix_diag ** 2

        frame_covs_obj = torch.zeros(
            [len(trafo), ordered_end_ray - ordered_start_ray, n_samples, 3],
            device=means.device,
        )
        frame_covs_obj[ordered_fr_mask] = covs[ordered_fr_bool_mask]

        frame_covs_obj = frame_covs_obj.view(len(trafo), -1, 3)

        frame_covs_obj = frame_covs_obj.bmm(
            sigma_wobj_matrix_diag.to(dtype=covs.dtype)).view(
            len(trafo), -1, n_samples, 3)

        covs_obj[ordered_fr_bool_mask] = frame_covs_obj[ordered_fr_mask]

        # Set starting point for next frame
        ordered_start_ray = ordered_end_ray


    means = means_obj
    covs = covs_obj
    return means, covs, obj_z_vals_world


class IntegratedPositionalEncoding(nn.Module):
    def __init__(self, input_dims=3, multires=10,include_input=False):
        super(IntegratedPositionalEncoding, self).__init__()
        self.out_dims = input_dims * 2 * (multires-1)

        self.max_freq = multires - 1
        # self.include_input = include_input


    def forward(self, x_coord):
        integrated_pos_enc = self.integrated_pos_enc(x_coord, min_deg=0, max_deg=self.max_freq)
        # if self.include_input:
        #     integrated_pos_enc = torch.cat((x_coord[0],integrated_pos_enc),-1)
        return  integrated_pos_enc


    def integrated_pos_enc(self, x_coord, min_deg=0, max_deg=16,):
        """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].
        Args:
          x_coord: a tuple containing: x, jnp.ndarray, variables to be encoded. Should
            be in [-pi, pi]. x_cov, jnp.ndarray, covariance matrices for `x`.
          min_deg: int, the min degree of the encoding.
          max_deg: int, the max degree of the encoding.
          diag: bool, if true, expects input covariances to be diagonal (full
            otherwise).
        Returns:
          encoded: jnp.ndarray, encoded variables.
        """
        x, x_cov_diag = x_coord
        scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=x.device)
        shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x[..., None, :] * scales[:, None], shape)
        y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None] ** 2, shape)

        return self.expected_sin(
            torch.cat([y, y + 0.5 * np.pi], dim=-1),
            torch.cat([y_var] * 2, dim=-1))[0]


    def expected_sin(self, x, x_var):
        """Estimates mean and variance of sin(z), z ~ N(x, var)."""
        # When the variance is wide, shrink sin towards zero.
        y = torch.exp(-0.5 * x_var) * torch.sin(x)
        y_var = torch.maximum(
            torch.tensor([0.], device=x.device), 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y ** 2)
        return y, y_var
