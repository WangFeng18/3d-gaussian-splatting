import os
import torch
from tqdm import tqdm
import cv2

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from utils import Timer

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
                # lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else 1,
                lambda i_iter: i_iter / warmup_iters if i_iter <= warmup_iters else _gamma**(i_iter-warmup_iters),
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

        # self.ssim_criterion = SSIM(window_size=11, reduction='mean')
        self.ssim_criterion = StructuralSimilarityIndexMeasure(data_range=1.0).to(gaussian_splatter.device)
        self.psnr_metrics = PeakSignalNoiseRatio().to(gaussian_splatter.device)
        self.l1_losses = np.zeros(opt.n_history_track)
        self.psnrs = np.zeros(opt.n_history_track)
        self.ssim_losses = np.zeros(opt.n_history_track)

        self.grad_counter = 0
        self.clear_grad()

    def clear_grad(self):
        self.accum_max_grad = torch.zeros_like(self.gaussian_splatter.gaussian_3ds.pos)
        self.grad_counter = 0
    
    def train_step(self, i_iter):
        opt = self.opt

        _in_reset_interval = (i_iter >= opt.n_opa_reset) and (i_iter % opt.n_opa_reset < opt.reset_interval)
        _adaptive_control_only_delete = (i_iter > 600 and i_iter % opt.n_adaptive_control == 0)
        _adaptive_control = (i_iter > 600 and i_iter < opt.adaptive_control_end_iter and i_iter % opt.n_adaptive_control == 0)
        _adaptive_control_accum_start = i_iter > 600 and (i_iter + opt.grad_accum_iters - 1) % opt.n_adaptive_control == 0
        self.optimizer.zero_grad()

        # forward
        camera_id = np.random.choice(self.train_split, 1)[0]
        rendered_img = self.gaussian_splatter(camera_id)

        # loss
        l1_loss = ((rendered_img - self.gaussian_splatter.ground_truth).abs()).mean()
        if opt.ssim_weight > 0:
            ssim_loss = 1. - self.ssim_criterion(
                rendered_img.unsqueeze(0).permute(0, 3, 1, 2), 
                self.gaussian_splatter.ground_truth.unsqueeze(0).permute(0, 3, 1, 2).to(rendered_img)
            )
        else:
            ssim_loss = torch.Tensor([0.0,]).to(l1_loss.device)
        loss = (1-opt.ssim_weight)*l1_loss + opt.ssim_weight*ssim_loss
        if opt.scale_reg > 0:
            loss += opt.scale_reg * self.gaussian_splatter.gaussian_3ds.scale.abs().mean()
        if opt.opa_reg > 0:
            opa_sigmoid = self.gaussian_splatter.gaussian_3ds.opa.sigmoid()
            loss += opt.opa_reg * (opa_sigmoid * (1-opa_sigmoid)).mean()

        psnr = self.psnr_metrics(rendered_img, self.gaussian_splatter.ground_truth)

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
            self.grad_counter += self.gaussian_splatter.culling_mask.to(torch.float32)
        else:
            assert opt.grad_accum_method == "max"
            self.accum_max_grad = torch.max(self.gaussian_splatter.gaussian_3ds.pos.grad.abs(), self.accum_max_grad)
            self.grad_counter = 1

        if (_adaptive_control or _adaptive_control_only_delete) and not _in_reset_interval:
            # adaptive control for gaussians
            # grad = self.gaussian_splatter.gaussian_3ds.pos.grad
            # adaptive_number = (self.accum_max_grad.abs().max(-1)[0] > 0.0002).sum()
            # adaptive_ratio = adaptive_number / grad[..., 0].numel()
            self.gaussian_splatter.gaussian_3ds.adaptive_control(
                self.accum_max_grad/(self.grad_counter+1e-3).unsqueeze(dim=-1),   
                taus=opt.split_thresh, 
                delete_thresh=opt.delete_thresh, 
                scale_activation=gaussian_splatter.scale_activation,
                grad_thresh=opt.grad_thresh,
                use_clone=opt.use_clone if (_adaptive_control) else False, 
                use_split=opt.use_split if (_adaptive_control) else False,
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
            # if _in_reset_interval and i_opt == 0:
                # param_group["lr"] = lr
        
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
        raw_metrics = {
            "loss": [],
            "n_gaussians": [],
        }

        bar = tqdm(range(0, opt.n_iters))
        for i_iter in bar:
            output = self.train_step(i_iter)
            avg_l1_loss = output["avg_l1_loss"]
            avg_ssim_loss = output["avg_ssim_loss"]
            avg_psnr = output["avg_psnr"]
            n_tile_gaussians = output["n_tile_gaussians"]
            n_gaussians = output["n_gaussians"]

            raw_metrics["loss"].append(output["loss"])
            raw_metrics["n_gaussians"].append(output["n_gaussians"])

            bar.set_description(
                desc=f"loss: {avg_l1_loss:.6f} ssim_loss: {avg_ssim_loss:.6f} psnr: {avg_psnr:.4f} n_tile_gaussians / n_gaussians: [{n_tile_gaussians}/{n_gaussians}]")

            rendered_img = output["image"]
            # write img
            if i_iter % opt.n_save_train_img == 0:
                img_npy = rendered_img.clip(0,1).detach().cpu().numpy()
                dirpath = f"{opt.experiment}/imgs/"
                os.makedirs(dirpath, exist_ok=True)
                cv2.imwrite(f"{opt.experiment}/imgs/train_{i_iter}.png", (img_npy*255).astype(np.uint8)[...,::-1])
                self.save_checkpoint()

            if i_iter % 100 == 0:
                Timer.show_recorder()

            # why render a downsample at 400 iterations? Maybe this is just a "unit test" 
            if i_iter == 400:
                gaussian_splatter.switch_resolution(opt.render_downsample)

            if i_iter == 4000 and not gaussian_splatter.use_sh_coeff:
                gaussian_splatter.add_sh_coeff()

            if i_iter % (opt.n_iters_test) == 0:
                test_psnrs = []
                test_ssims = []
                elapsed = 0
                for test_camera_id in self.test_split:
                    output = self.test(test_camera_id)
                    elapsed += output["render_time"]
                    test_psnrs.append(output["psnr"])
                    test_ssims.append(output["ssim"])
                    # save imgs
                    dirpath = f"{opt.experiment}/test_imgs/"
                    os.makedirs(dirpath, exist_ok=True)
                    img_npy = output["image"].clip(0,1).detach().cpu().numpy()
                    cv2.imwrite(f"{opt.experiment}/test_imgs/iter_{i_iter}_cid_{test_camera_id}.png", (img_npy*255).astype(np.uint8)[...,::-1])
                print("\tTEST SPLIT PSNR: {:.4f}".format(np.mean(test_psnrs)))
                print("\tTEST SPLIT SSIM: {:.4f}".format(np.mean(test_ssims)))
                print("\tRENDERING SPEED: {:.4f}".format(len(self.test_split)/elapsed))
        return raw_metrics

    @torch.no_grad()
    def test(self, camera_id, extrinsics=None, intrinsics=None):
        
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        tic.record()
        self.gaussian_splatter.eval()
        rendered_img = self.gaussian_splatter(camera_id, extrinsics, intrinsics)
        toc.record()
        torch.cuda.synchronize()
        render_time = tic.elapsed_time(toc)/1000
        if camera_id is not None:
            psnr = self.psnr_metrics(rendered_img, self.gaussian_splatter.ground_truth).item()
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
                "render_time": render_time,
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
        torch.save(ckpt, os.path.join(opt.experiment, "ckpt.pth"))

