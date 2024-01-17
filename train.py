import os
import time
import csv
import numpy as np 
import argparse
from splatter import Splatter
from trainer import Trainer

from visergui import ViserViewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=7001)
    parser.add_argument("--n_iters_warmup", type=int, default=300)

    # test every n_iters_test iterations
    parser.add_argument("--n_iters_test", type=int, default=200)

    # how many iterations to track psnr over
    parser.add_argument("--n_history_track", type=int, default=100)
    parser.add_argument("--n_save_train_img", type=int, default=500)

    # how frequently to run adaptive control
    parser.add_argument("--n_adaptive_control", type=int, default=100)
    parser.add_argument("--render_downsample_start", type=int, default=4)
    parser.add_argument("--render_downsample", type=int, default=4)

    ## DATA ARGS
    parser.add_argument("--data", type=str, default="colmap_garden/")
    parser.add_argument("--experiment", type=str, default="default_experiment")

    parser.add_argument("--scale_init_value", type=float, default=1)
    parser.add_argument("--opa_init_value", type=float, default=0.3)
    # in some tests 0.1/0.15 seems to yield higher PSNR 
    parser.add_argument("--tile_culling_prob_thresh", type=float, default=0.05)

    # learning rate
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lr_factor_for_scale", type=float, default=1)
    parser.add_argument("--lr_factor_for_rgb", type=float, default=10)
    parser.add_argument("--lr_factor_for_opa", type=float, default=10)
    parser.add_argument("--lr_factor_for_quat", type=float, default=1)


    # default 0.2 in paper (lamdba)
    parser.add_argument("--ssim_weight", type=float, default=0.2)

    # Using "official" seems to result in poor reconstruction
    parser.add_argument("--lr_decay", type=str, default="exp", choices=["none", "official", "exp"])
    
    parser.add_argument("--delete_thresh", type=float, default=1.5)
    parser.add_argument("--split_thresh", type=float, default=0.05)
    parser.add_argument("--scale_activation", type=str, default="abs", choices=["abs", "exp"])


    # Seems to be too aggressive at the moment, opacity has a hard time recovering after the resets
    # even with --reset_interval 1500
    parser.add_argument("--n_opa_reset", type=int, default=10000000)
    parser.add_argument("--reset_interval", type=int, default=500)


    parser.add_argument("--debug", type=int, default=0)

    # use spherical harmonics
    # NOTE this setting must match playback!
    # Seems like there is a bug where if sh_coeff > 0, gaussians start to get generated at a rapid rate after ~5k iterations
    # and artifacts / blocky patterns start to appear
    parser.add_argument("--use_sh_coeff", type=int, default=0) 
    parser.add_argument("--scale_reg", type=float, default=0)
    parser.add_argument("--opa_reg", type=float, default=0)
    parser.add_argument("--adaptive_lr", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--fast_drawing", type=int, default=1)

    # adaptive control
    parser.add_argument("--grad_accum_iters", type=int, default=50)

    # grad accum method max is broken it seems
    parser.add_argument("--grad_accum_method", type=str, default="mean", choices=["mean", "max"])

    # gradient threshold to clone / split
    # lower value = more gaussians
    parser.add_argument("--grad_thresh", type=float, default=0.0002)

    parser.add_argument("--use_clone", type=int, default=1)
    parser.add_argument("--use_split", type=int, default=1)
    parser.add_argument("--clone_dt", type=float, default=0.01)
    parser.add_argument("--grad_aggregation", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--adaptive_control_end_iter", type=int, default=1000000000)

    # GUI related
    parser.add_argument("--gui", default=0, type=int)
    parser.add_argument("--test", default=0, type=int)
    
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512"

    opt = parser.parse_args()
    np.random.seed(opt.seed)
    data_path = os.path.join(opt.data, 'sparse', '0')
    img_path = os.path.join(opt.data, f'images_{opt.render_downsample_start}')

    if opt.ckpt == "":
        opt.ckpt = None

    start_time = time.time()
    gaussian_splatter = Splatter(
        data_path,
        img_path,
        render_weight_normalize=False, 
        render_downsample=opt.render_downsample,
        use_sh_coeff=opt.use_sh_coeff,
        scale_init_value=opt.scale_init_value,
        opa_init_value=opt.opa_init_value,
        tile_culling_prob_thresh=opt.tile_culling_prob_thresh,
        debug=opt.debug,
        scale_activation=opt.scale_activation,
        load_ckpt=opt.ckpt,
        fast_drawing=opt.fast_drawing,
        test=opt.test,
    )
    trainer = Trainer(gaussian_splatter, opt)
    if opt.gui:
        assert opt.test == 1
        gui = ViserViewer(device=gaussian_splatter.device, viewer_port=6789)
        gui.set_renderer(trainer)
        while(True):
            gui.update()
    else:
        metrics = trainer.train()
        metrics_file = os.path.join(opt.experiment, "metrics.csv")
        with open(metrics_file, 'w') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["iteration", "loss", "n_gaussians"])
            for i in range(len(metrics["loss"])):
                writer.writerow([i, metrics["loss"][i],  metrics["n_gaussians"][i]])
        
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print("Total Elapsed Time: {}".format(total_elapsed_time))
    