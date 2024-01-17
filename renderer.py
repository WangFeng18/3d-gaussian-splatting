import torch
import gaussian

class _Drawer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        gaussians_pos,
        gaussians_rgb,
        gaussians_opa,
        gaussians_cov,
        tile_n_point_accum,
        padded_height,
        padded_width,
        focal_x,
        focal_y,
        render_weight_normalize=False,
        sigmoid=False,
        use_sh_coeff=False,
        fast=False,
        rays_o=None,
        lefttop_pos=None,
        vec_dx=None,
        vec_dy=None
    ):
        rendered_image = torch.zeros(padded_height, padded_width, 3, device=gaussians_pos.device, dtype=torch.float32)
        gaussian.draw(
            gaussians_pos,
            gaussians_rgb,
            gaussians_opa,
            gaussians_cov,
            tile_n_point_accum,
            rendered_image,
            focal_x,
            focal_y,
            render_weight_normalize,
            sigmoid,
            fast,
            rays_o,
            lefttop_pos,
            vec_dx,
            vec_dy,
            use_sh_coeff,
        )
        ctx.save_for_backward(gaussians_pos, gaussians_rgb, gaussians_opa, gaussians_cov, tile_n_point_accum, rendered_image, rays_o, lefttop_pos, vec_dx, vec_dy)
        ctx.focal_x = focal_x
        ctx.focal_y = focal_y
        ctx.render_weight_normalize = render_weight_normalize
        ctx.sigmoid = sigmoid
        ctx.fast = fast
        ctx.use_sh_coeff = use_sh_coeff
        return rendered_image
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output
        gaussians_pos, gaussians_rgb, gaussians_opa, gaussians_cov, tile_n_point_accum, rendered_image, rays_o, lefttop_pos, vec_dx, vec_dy = ctx.saved_tensors
        grad_pos = torch.zeros_like(gaussians_pos)
        grad_rgb = torch.zeros_like(gaussians_rgb)
        grad_opa = torch.zeros_like(gaussians_opa)
        grad_cov = torch.zeros_like(gaussians_cov)
        gaussian.draw_backward(
            gaussians_pos,
            gaussians_rgb,
            gaussians_opa,
            gaussians_cov,
            tile_n_point_accum,
            rendered_image,
            grad_output,
            grad_pos,
            grad_rgb,
            grad_opa,
            grad_cov,
            ctx.focal_x,
            ctx.focal_y,
            ctx.render_weight_normalize,
            ctx.sigmoid,
            ctx.fast,
            rays_o,
            lefttop_pos,
            vec_dx,
            vec_dy,
            ctx.use_sh_coeff,
        )
        return grad_pos, grad_rgb, grad_opa, grad_cov, None, None, None, None, None, None, None, None, None, None, None, None, None

draw = _Drawer.apply

class _trunc_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-1, 1))

trunc_exp = _trunc_exp.apply


class _GlobalCulling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, quat, scale, current_rot, current_tran, near, half_width, half_height):
        res_pos = torch.zeros_like(pos)
        res_cov = torch.zeros((pos.shape[0], 2, 2), device=pos.device)
        culling_mask = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)

        gaussian.global_culling(
            pos, quat, scale, current_rot, current_tran, res_pos, res_cov, culling_mask, near, half_width, half_height
        )
        ctx.save_for_backward(culling_mask, pos, quat, scale, current_rot, current_tran)
        return res_pos, res_cov, culling_mask.detach()

    @staticmethod
    def backward(ctx, gradout_pos, gradout_cov, grad_culling_mask):
        culling_mask = ctx.saved_tensors[0]
        pos = ctx.saved_tensors[1]
        quat = ctx.saved_tensors[2]
        scale = ctx.saved_tensors[3]
        current_rot = ctx.saved_tensors[4]
        current_tran = ctx.saved_tensors[5]

        gradinput_pos = torch.zeros_like(gradout_pos)
        gradinput_quat = torch.zeros((gradout_pos.shape[0], 4), device=gradout_pos.device)
        gradinput_scale = torch.zeros((gradout_pos.shape[0], 3), device=gradout_pos.device)
        gaussian.global_culling_backward(
            pos, quat, scale, current_rot, current_tran,
            gradout_pos, 
            gradout_cov, 
            culling_mask,
            gradinput_pos, 
            gradinput_quat, 
            gradinput_scale, 
        )

        return gradinput_pos, gradinput_quat, gradinput_scale, None, None, None, None, None
    
global_culling = _GlobalCulling.apply
