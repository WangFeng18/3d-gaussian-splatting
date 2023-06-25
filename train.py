import os
import time
from tqdm import tqdm
import numpy as np 
import torch
import argparse
from splatter import Splatter
import cv2
from torchgeometry.losses import SSIM
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
from utils import Timer
# from gui import NeRFGUI
from visergui import ViserViewer

class Trainer:
    def __init__(self, gaussian_splatter, opt):
        self.gaussian_splatter = gaussian_splatter
        self.opt = opt
        self.lr_opa = opt.lr * opt.lr_factor_for_opa
        self.lr_rgb = opt.lr * opt.lr_factor_for_rgb
        self.lr_pos = opt.lr * 1
        self.lr_quat = opt.lr * opt.lr_factor_for_quat
        self.lr_scale = opt.lr * opt.lr_factor_for_scale
        self.lrs = [self.lr_opa, self.lr_rgb, self.lr_pos, self.lr_scale, self.lr_quat]

        warmup_iters = opt.n_iters_warmup
        if self.opt.lr_decay == "official":
            _gamma = (0.01)**(1/(self.opt.n_iters-warmup_iters))
            self.lr_lambdas = [
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 1,
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 1,
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else _gamma**(i_iter-warmup_iters),
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 1,
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 1,
            ]
        elif self.opt.lr_decay == "none":
            self.lr_lambdas = [
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 0.2**((i_iter-warmup_iters) // 2000),
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 0.2**((i_iter-warmup_iters) // 2000),
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 0.2**((i_iter-warmup_iters) // 2000),
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 0.2**((i_iter-warmup_iters) // 2000),
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 0.2**((i_iter-warmup_iters) // 2000),
            ]
        else:
            assert self.opt.lr_decay == "exp"
            _gamma = (0.01)**(1/(self.opt.n_iters-warmup_iters))
            self.lr_lambdas = [
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else _gamma**(i_iter-warmup_iters),
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else _gamma**(i_iter-warmup_iters),
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else _gamma**(i_iter-warmup_iters),
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else _gamma**(i_iter-warmup_iters),
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else _gamma**(i_iter-warmup_iters),
            ]
        self.optimizer = torch.optim.Adam([
                {"params": gaussian_splatter.gaussian_3ds.opa, "lr": self.lr_opa * self.lr_lambdas[0](0)},
                {"params": gaussian_splatter.gaussian_3ds.rgb, "lr": self.lr_rgb * self.lr_lambdas[1](0)},
                {"params": gaussian_splatter.gaussian_3ds.pos, "lr": self.lr_pos * self.lr_lambdas[2](0)},
                {"params": gaussian_splatter.gaussian_3ds.scale, "lr": self.lr_scale * self.lr_lambdas[3](0)},
                {"params": gaussian_splatter.gaussian_3ds.quat, "lr": self.lr_quat * self.lr_lambdas[4](0)},
            ],
            betas=(0.9, 0.99),
        )

        if not opt.test:
            self.n_cameras = len(gaussian_splatter.imgs)
            self.test_split = np.arange(0, self.n_cameras, 8)
            self.train_split = np.array(list(set(np.arange(0, self.n_cameras, 1)) - set(self.test_split)))

        self.ssim_criterion = SSIM(window_size=11, reduction='mean')
        self.l1_losses = np.zeros(opt.n_history_track)
        self.psnrs = np.zeros(opt.n_history_track)
        self.ssim_losses = np.zeros(opt.n_history_track)

        self.grad_counter = 0
        self.clear_grad()

    def clear_grad(self):
        self.accum_max_grad = torch.zeros_like(self.gaussian_splatter.gaussian_3ds.pos)
        self.grad_counter = 0
    
    def train_step(self, i_iter, bar):
        opt = self.opt
        _adaptive_control = i_iter > 600 and i_iter % opt.n_adaptive_control == 0
        _adaptive_control_accum_start = i_iter > 600 and (i_iter + opt.grad_accum_iters - 1) % opt.n_adaptive_control == 0
        self.optimizer.zero_grad()

        # forward
        camera_id = np.random.choice(self.train_split, 1)[0]
        rendered_img = self.gaussian_splatter(camera_id)

        # loss
        l1_loss = ((rendered_img - self.gaussian_splatter.ground_truth).abs()).mean()
        ssim_loss = self.ssim_criterion(
            rendered_img.unsqueeze(0).permute(0, 3, 1, 2), 
            self.gaussian_splatter.ground_truth.unsqueeze(0).permute(0, 3, 1, 2).to(rendered_img)
        )
        loss = (1-opt.ssim_weight)*l1_loss + opt.ssim_weight*ssim_loss
        if opt.scale_reg > 0:
            loss += opt.scale_reg * self.gaussian_splatter.gaussian_3ds.scale.abs().mean()
        psnr = psnr_func(rendered_img, self.gaussian_splatter.ground_truth)

        # optimize
        with Timer("backward", debug=opt.debug):
            loss.backward()
        with Timer("step", debug=opt.debug):
            self.optimizer.step()

        # historical losses for smoothing
        self.l1_losses = np.roll(self.l1_losses, 1)
        self.psnrs = np.roll(self.psnrs, 1)
        self.ssim_losses = np.roll(self.ssim_losses, 1)
        self.l1_losses[0] = l1_loss.item()
        self.psnrs[0] = psnr.item()
        self.ssim_losses[0] = ssim_loss.item()

        avg_l1_loss = self.l1_losses[:min(i_iter+1, self.l1_losses.shape[0])].mean()
        avg_ssim_loss = self.ssim_losses[:min(i_iter+1, self.ssim_losses.shape[0])].mean()
        avg_psnr = self.psnrs[:min(i_iter+1, self.psnrs.shape[0])].mean()

        # grad info for debuging
        grad_info = [
            self.gaussian_splatter.gaussian_3ds.opa.grad.abs().mean(),
            self.gaussian_splatter.gaussian_3ds.rgb.grad.abs().mean(),
            self.gaussian_splatter.gaussian_3ds.pos.grad.abs().mean(),
            self.gaussian_splatter.gaussian_3ds.scale.grad.abs().mean(),
            self.gaussian_splatter.gaussian_3ds.quat.grad.abs().mean(),
        ]

        # log

        if _adaptive_control_accum_start:
            self.clear_grad()
        # self.accum_max_grad = torch.max(self.gaussian_splatter.gaussian_3ds.pos.grad, self.accum_max_grad)
        if opt.grad_accum_method == "mean":
            self.accum_max_grad += self.gaussian_splatter.gaussian_3ds.pos.grad.abs()
            self.grad_counter += 1
        else:
            assert opt.grad_accum_method == "max"
            self.accum_max_grad = torch.max(self.gaussian_splatter.gaussian_3ds.pos.grad.abs(), self.accum_max_grad)
            self.grad_counter = 1

        if _adaptive_control:
            # adaptive control for gaussians
            # grad = self.gaussian_splatter.gaussian_3ds.pos.grad
            # adaptive_number = (self.accum_max_grad.abs().max(-1)[0] > 0.0002).sum()
            # adaptive_ratio = adaptive_number / grad[..., 0].numel()
            self.gaussian_splatter.gaussian_3ds.adaptive_control(
                self.accum_max_grad/self.grad_counter, 
                taus=opt.split_thresh, 
                delete_thresh=opt.delete_thresh, 
                scale_activation=gaussian_splatter.scale_activation,
                grad_thresh=opt.grad_thresh,
                use_clone=opt.use_clone,     
                use_split=opt.use_split,
                grad_aggregation=opt.grad_aggregation,
                clone_dt=opt.clone_dt,
            )
            # optimizer = torch.optim.Adam(gaussian_splatter.parameters(), lr=lr_lambda(0), betas=(0.9, 0.99))
            self.optimizer = torch.optim.Adam([
                    {"params": self.gaussian_splatter.gaussian_3ds.opa, "lr": self.lr_opa * self.lr_lambdas[0](i_iter)},
                    {"params": self.gaussian_splatter.gaussian_3ds.rgb, "lr": self.lr_rgb * self.lr_lambdas[1](i_iter)},
                    {"params": self.gaussian_splatter.gaussian_3ds.pos, "lr": self.lr_pos * self.lr_lambdas[2](i_iter)},
                    {"params": self.gaussian_splatter.gaussian_3ds.scale, "lr": self.lr_scale * self.lr_lambdas[3](i_iter)},
                    {"params": self.gaussian_splatter.gaussian_3ds.quat, "lr": self.lr_quat * self.lr_lambdas[4](i_iter)},
                ],
                betas=(0.9, 0.99),
            )
            self.clear_grad()

        for i_opt, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, self.lrs)):
            param_group['lr'] = self.lr_lambdas[i_opt](i_iter) * lr
        
        if i_iter % (opt.n_opa_reset) == 0 and i_iter > 0:
            self.gaussian_splatter.gaussian_3ds.reset_opa()

        return {
            "image": rendered_img,
            "loss": (1-opt.ssim_weight) * avg_l1_loss + opt.ssim_weight * avg_ssim_loss,
            "avg_l1_loss": avg_l1_loss,
            "avg_ssim_loss": avg_ssim_loss,
            "avg_psnr": avg_psnr,
            "n_tile_gaussians": self.gaussian_splatter.n_tile_gaussians,
            "n_gaussians": self.gaussian_splatter.n_gaussians,
            "grad_info": grad_info,
        }

    def train(self):
        bar = tqdm(range(0, opt.n_iters))
        for i_iter in bar:
            output = self.train_step(i_iter, bar)
            avg_l1_loss = output["avg_l1_loss"]
            avg_ssim_loss = output["avg_ssim_loss"]
            avg_psnr = output["avg_psnr"]
            n_tile_gaussians = output["n_tile_gaussians"]
            n_gaussians = output["n_gaussians"]
            grad_info = output["grad_info"]

            grad_desc = "[{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}]".format(*grad_info)
            bar.set_description(
                desc=f"loss: {avg_l1_loss:.6f}/{avg_ssim_loss:.6f}/{avg_psnr:.4f}/[{n_tile_gaussians}/{n_gaussians}]:" +
                                    f"lr: {self.optimizer.param_groups[0]['lr']:.4f}|{self.optimizer.param_groups[1]['lr']:.4f}|{self.optimizer.param_groups[2]['lr']:.4f}|{self.optimizer.param_groups[3]['lr']:.4f}|{self.optimizer.param_groups[4]['lr']:.4f} " + 
                                    f"grad: {grad_desc}"
            )

            rendered_img = output["image"]
            # write img
            if i_iter % opt.n_save_train_img == 0:
                img_npy = rendered_img.clip(0,1).detach().cpu().numpy()
                dirpath = f"{opt.exp}/imgs/"
                os.makedirs(dirpath, exist_ok=True)
                cv2.imwrite(f"{opt.exp}/imgs/train_{i_iter}.png", (img_npy*255).astype(np.uint8)[...,::-1])
                self.save_checkpoint()

            if i_iter % 100 == 0 and opt.debug:
                Timer.show_recorder()

            if i_iter % (opt.n_iters_test) == 0:
                test_psnrs = []
                test_ssims = []
                time_start = time.time()
                for test_camera_id in self.test_split:
                    output = self.test(test_camera_id)
                    test_psnrs.append(output["psnr"])
                    test_ssims.append(output["ssim"])
                    # save imgs
                    dirpath = f"{opt.exp}/test_imgs/"
                    os.makedirs(dirpath, exist_ok=True)
                    img_npy = output["image"].clip(0,1).detach().cpu().numpy()
                    cv2.imwrite(f"{opt.exp}/test_imgs/iter_{i_iter}_cid_{test_camera_id}.png", (img_npy*255).astype(np.uint8)[...,::-1])
                time_end = time.time()
                print(test_psnrs)
                print(test_ssims)
                print("TEST SPLIT PSNR: {:.4f}".format(np.mean(test_psnrs)))
                print("TEST SPLIT SSIM: {:.4f}".format(np.mean(test_ssims)))
                print("REDNDERING SPEED: {:.4f}".format(len(self.test_split)/(time_end - time_start)))

    @torch.no_grad()
    def test(self, camera_id, extrinsics=None, intrinsics=None):
        self.gaussian_splatter.eval()
        rendered_img = self.gaussian_splatter(camera_id, extrinsics, intrinsics)
        if camera_id is not None:
            psnr = psnr_func(rendered_img, self.gaussian_splatter.ground_truth).item()
            ssim = self.ssim_criterion(
                rendered_img.unsqueeze(0).permute(0, 3, 1, 2), 
                self.gaussian_splatter.ground_truth.unsqueeze(0).permute(0, 3, 1, 2).to(rendered_img),
            ).item()
        self.gaussian_splatter.train()
        output = {"image": rendered_img}
        if camera_id is not None:
            output.update({
                "psnr": psnr,
                "ssim": ssim,
            })
        return output
    
    def save_checkpoint(self):
        ckpt = {
            "pos": self.gaussian_splatter.gaussian_3ds.pos,
            "opa": self.gaussian_splatter.gaussian_3ds.opa,
            "rgb": self.gaussian_splatter.gaussian_3ds.rgb,
            "quat": self.gaussian_splatter.gaussian_3ds.quat,
            "scale": self.gaussian_splatter.gaussian_3ds.scale,
        }
        torch.save(ckpt, "ckpt.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=7001)
    parser.add_argument("--n_iters_warmup", type=int, default=300)
    parser.add_argument("--n_iters_test", type=int, default=200)
    parser.add_argument("--n_history_track", type=int, default=100)
    parser.add_argument("--n_save_train_img", type=int, default=100)
    parser.add_argument("--n_adaptive_control", type=int, default=100)
    parser.add_argument("--render_downsample", type=int, default=4)
    parser.add_argument("--jacobian_track", type=int, default=0)
    parser.add_argument("--data", type=str, default="colmap_garden/")
    parser.add_argument("--scale_init_value", type=float, default=1)
    parser.add_argument("--opa_init_value", type=float, default=0.3)
    parser.add_argument("--tile_culling_dist_thresh", type=float, default=0.5)
    parser.add_argument("--tile_culling_prob_thresh", type=float, default=0.05)
    parser.add_argument("--tile_culling_method", type=str, default="prob2", choices=["dist", "prob", "prob2"])

    # learning rate
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lr_factor_for_scale", type=float, default=1)
    parser.add_argument("--lr_factor_for_rgb", type=float, default=10)
    parser.add_argument("--lr_factor_for_opa", type=float, default=10)
    parser.add_argument("--lr_factor_for_quat", type=float, default=1)
    parser.add_argument("--lr_decay", type=str, default="exp", choices=["none", "official", "exp"])

    parser.add_argument("--delete_thresh", type=float, default=1.5)
    parser.add_argument("--n_opa_reset", type=int, default=10000000)
    parser.add_argument("--split_thresh", type=float, default=0.05)
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--use_sh_coeff", type=int, default=0)
    parser.add_argument("--scale_reg", type=float, default=0)
    parser.add_argument("--cudaculling", type=int, default=1)
    parser.add_argument("--adaptive_lr", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--scale_activation", type=str, default="abs", choices=["abs", "exp"])
    parser.add_argument("--fast_drawing", type=int, default=1)
    parser.add_argument("--exp", type=str, default="default")

    # adaptive control
    # parser.add_argument("--grad_accum_iters", type=int, default=20)
    parser.add_argument("--grad_accum_iters", type=int, default=50)
    parser.add_argument("--grad_accum_method", type=str, default="max", choices=["mean", "max"])
    parser.add_argument("--grad_thresh", type=float, default=0.0001)
    parser.add_argument("--use_clone", type=int, default=0)
    parser.add_argument("--use_split", type=int, default=1)
    parser.add_argument("--clone_dt", type=float, default=0.01)
    parser.add_argument("--grad_aggregation", type=str, default="max", choices=["max", "mean"])

    # GUI related
    parser.add_argument("--gui", default=0, type=int)
    parser.add_argument("--test", default=0, type=int)
    parser.add_argument("--H", default=768, type=int)
    parser.add_argument("--W", default=1024, type=int)
    parser.add_argument("--radius", default=5.0, type=float)
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    #parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--bound', type=float, default=10, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")


    opt = parser.parse_args()
    np.random.seed(opt.seed)
    if opt.jacobian_track:
        jacobian_calc="torch"
    else:
        jacobian_calc="cuda"
    data_path = os.path.join(opt.data, 'sparse', '0')
    img_path = os.path.join(opt.data, f'images_{opt.render_downsample}')

    if opt.ckpt == "":
        opt.ckpt = None
    gaussian_splatter = Splatter(
        data_path,
        img_path,
        render_weight_normalize=False, 
        render_downsample=opt.render_downsample,
        use_sh_coeff=opt.use_sh_coeff,
        scale_init_value=opt.scale_init_value,
        opa_init_value=opt.opa_init_value,
        tile_culling_method=opt.tile_culling_method,
        tile_culling_dist_thresh=opt.tile_culling_dist_thresh,
        tile_culling_prob_thresh=opt.tile_culling_prob_thresh,
        debug=opt.debug,
        scale_activation=opt.scale_activation,
        cudaculling=opt.cudaculling,
        load_ckpt=opt.ckpt,
        fast_drawing=opt.fast_drawing,
        test=opt.test,
        #jacobian_calc="torch",
    )
    trainer = Trainer(gaussian_splatter, opt)
    if opt.gui:
        assert opt.test == 1
        # gui = NeRFGUI(opt, trainer)
        # gui.render()
        gui = ViserViewer(device=gaussian_splatter.device, viewer_port=6789)
        gui.set_renderer(trainer)
        while(True):
            gui.update()
    else:
        trainer.train()
