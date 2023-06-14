import os
from tqdm import tqdm
import numpy as np 
import torch
import argparse
from splatter import Splatter
import cv2
from torchgeometry.losses import SSIM
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
from utils import Timer

def train(gaussian_splatter, opt):
    # lr = 0.05
    lr_opa = opt.lr * opt.lr_factor_for_opa
    lr_rgb = opt.lr * 1
    lr_pos = opt.lr * 1
    lr_quat = opt.lr * opt.lr_factor_for_quat
    lr_scale = opt.lr * opt.lr_factor_for_scale
    lrs = [lr_opa, lr_rgb, lr_pos, lr_scale, lr_quat]
    
    warmup_iters = 100
    lr_lambda = lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 1 * 0.2**((i_iter-warmup_iters) // 2000)
    optimizer = torch.optim.Adam([
            {"params": gaussian_splatter.gaussian_3ds.opa, "lr": lr_opa * lr_lambda(0)},
            {"params": gaussian_splatter.gaussian_3ds.rgb, "lr": lr_rgb * lr_lambda(0)},
            {"params": gaussian_splatter.gaussian_3ds.pos, "lr": lr_pos * lr_lambda(0)},
            {"params": gaussian_splatter.gaussian_3ds.scale, "lr": lr_scale * lr_lambda(0)},
            {"params": gaussian_splatter.gaussian_3ds.quat, "lr": lr_quat * lr_lambda(0)},
        ],
        betas=(0.9, 0.99),
    )
    # gaussian_splatter.parameters(), lr=lr_lambda(0), betas=(0.9, 0.99))

    # _gamma = (end_lr/lr)**(1/opt.n_iters)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.2)
    n_cameras = len(gaussian_splatter.imgs)
    bar = tqdm(range(0, opt.n_iters))
    ssim_criterion = SSIM(window_size=11, reduction='mean')
    l1_losses = np.zeros(opt.n_history_track)
    psnrs = np.zeros(opt.n_history_track)
    ssim_losses = np.zeros(opt.n_history_track)
    grads = {}
    def save_grad(name):
        def hook(grad):
            grads[name] = grad
        return hook

    for i_iter in bar:
        optimizer.zero_grad()
        camera_id = np.random.randint(0, n_cameras)
        #rendered_img = gaussian_splatter(camera_id, record_view_space_pos_grad=(i_iter % opt.n_adaptive_control == 0 and i_iter>0))
        rendered_img = gaussian_splatter(camera_id)
        # loss = ((rendered_img - gaussian_splatter.ground_truth)**2).mean()
        l1_loss = ((rendered_img - gaussian_splatter.ground_truth).abs()).mean()
        ssim_loss = ssim_criterion(rendered_img.unsqueeze(0).permute(0, 3, 1, 2), gaussian_splatter.ground_truth.unsqueeze(0).permute(0, 3, 1, 2).to(rendered_img))
        # ssim_loss = 0
        #loss = 0.99*l1_loss + 0.01*ssim_loss
        loss = 0.8*l1_loss + 0.2*ssim_loss
        if opt.scale_reg > 0:
            loss += opt.scale_reg * gaussian_splatter.gaussian_3ds.scale.abs().mean()
        handle = None
        if i_iter % opt.n_adaptive_control == 0 and i_iter > 0:
            # gaussian_splatter.gaussian_3ds_pos_camera_space_culled.register_hook(save_grad("grad"))
            handle = gaussian_splatter.gaussian_3ds.pos.register_hook(save_grad("grad"))
        else:
            if handle is not None:
                handle.remove()
            #gaussian_splatter.gaussian_3ds.pos.register_hook(save_grad("grad"))

        # ssim_loss = 0
        with Timer("psnr"):
            psnr = psnr_func(rendered_img, gaussian_splatter.ground_truth)
        # loss = l1_loss
        with Timer("backward", debug=opt.debug):
            loss.backward()
        with Timer("step", debug=opt.debug):
            optimizer.step()
        # scheduler.step()
        l1_losses = np.roll(l1_losses, 1)
        psnrs = np.roll(psnrs, 1)
        ssim_losses = np.roll(ssim_losses, 1)
        l1_losses[0] = l1_loss.item()
        psnrs[0] = psnr.item()
        ssim_losses[0] = 0 # ssim_loss.item()
        avg_l1_loss = l1_losses[:min(i_iter+1, l1_losses.shape[0])].mean()
        avg_ssim_loss = ssim_losses[:min(i_iter+1, ssim_losses.shape[0])].mean()
        avg_psnr = psnrs[:min(i_iter+1, psnrs.shape[0])].mean()
        bar.set_description(desc=f"loss: {avg_l1_loss:.6f}/{avg_ssim_loss:.6f}/{avg_psnr:.4f}/[{gaussian_splatter.n_tile_gaussians}/{gaussian_splatter.n_gaussians}]: lr: {optimizer.param_groups[0]['lr']:.6f}")
        if i_iter % opt.n_save_train_img == 0:
            img_npy = rendered_img.clip(0,1).detach().cpu().numpy()
            cv2.imwrite(f"imgs/train_{i_iter}.png", (img_npy*255).astype(np.uint8)[...,::-1])

        # print(gaussian_splatter.gaussian_3ds.opa.grad.abs().mean()) 
        # print(gaussian_splatter.gaussian_3ds.rgb.grad.abs().mean()) 
        # print(gaussian_splatter.gaussian_3ds.pos.grad.abs().mean()) 
        # print(gaussian_splatter.gaussian_3ds.quat.grad.abs().mean()) 
        # print(gaussian_splatter.gaussian_3ds.scale.grad.abs().mean()) 

        if i_iter % 100 == 0:
            Timer.show_recorder()
        if i_iter % opt.n_adaptive_control == 0 and i_iter > 0:
            # adaptive control for gaussians
            grad = grads["grad"]
            # print(grad)
            adaptive_number = (grad.abs().max(-1)[0] > 0.0002).sum()
            adaptive_ratio = adaptive_number / grad[..., 0].numel()
            # print(adaptive_number, adaptive_ratio)
            gaussian_splatter.gaussian_3ds.adaptive_control(grad, taus=opt.split_thresh, delete_thresh=opt.delete_thresh)
            optimizer = torch.optim.Adam(gaussian_splatter.parameters(), lr=lr_lambda(0), betas=(0.9, 0.99))

        # if i_iter % (opt.n_adaptive_control*5) == 0 and i_iter > 0:
            # gaussian_splatter.gaussian_3ds.reset_opa()

        for param_group, lr in zip(optimizer.param_groups, lrs):
            param_group['lr'] = lr_lambda(i_iter) * lr

if __name__ == "__main__":
    # python train.py --render_downsample 2 --scale_init_value 0.01 --opa_init_value 0.5 --lr 0.001
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=10000)
    parser.add_argument("--n_history_track", type=int, default=100)
    parser.add_argument("--n_save_train_img", type=int, default=100)
    parser.add_argument("--n_adaptive_control", type=int, default=40000)
    parser.add_argument("--render_downsample", type=int, default=2)
    parser.add_argument("--jacobian_track", type=int, default=0)
    parser.add_argument("--data", type=str, default="garden")
    parser.add_argument("--scale_init_value", type=float, default=0.02)
    parser.add_argument("--opa_init_value", type=float, default=0.1)
    parser.add_argument("--tile_culling_dist_thresh", type=float, default=0.5)
    parser.add_argument("--tile_culling_prob_thresh", type=float, default=0.1)
    parser.add_argument("--tile_culling_method", type=str, default="dist", choices=["dist", "prob", "prob2"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_factor_for_scale", type=float, default=1)
    parser.add_argument("--lr_factor_for_opa", type=float, default=1)
    parser.add_argument("--lr_factor_for_quat", type=float, default=1)
    parser.add_argument("--delete_thresh", type=float, default=1.5)
    parser.add_argument("--split_thresh", type=float, default=0.05)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--scale_reg", type=float, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    opt = parser.parse_args()
    np.random.seed(opt.seed)
    if opt.jacobian_track:
        jacobian_calc="torch"
    else:
        jacobian_calc="cuda"
    if opt.data == "garden":
        data_path = os.path.join("colmap_garden/sparse/0/")
        img_path = f"colmap_garden/images_{opt.render_downsample}/"
    elif opt.data == "fern":
        data_path = os.path.join("llff/colmap/sparse/0/") 
        img_path = f"llff/images_{opt.render_downsample}/"

    gaussian_splatter = Splatter(
        data_path,
        img_path,
        render_weight_normalize=False, 
        render_downsample=opt.render_downsample,
        scale_init_value=opt.scale_init_value,
        opa_init_value=opt.opa_init_value,
        tile_culling_method=opt.tile_culling_method,
        tile_culling_dist_thresh=opt.tile_culling_dist_thresh,
        tile_culling_prob_thresh=opt.tile_culling_prob_thresh,
        debug=opt.debug,
        #jacobian_calc="torch",
    )
    train(gaussian_splatter, opt)
