# 3d-Gaussian-Splatting 
An unofficial Implementation of 3D Gaussian Splatting for Real-Time Radiance Field Rendering [SIGGRAPH 2023].

We implement the 3d gaussian splatting methods through PyTorch with CUDA extensions, including the global culling, tile-based culling and rendering forward/backward codes.

Work in progress.
#### Update
- 6/26/2023 Fix bugs of SSIM criterion, PSNR is improved from 24.28 to 24.85 (Garden Scene)
- 6/26/2023 Accelerate **Training** Speed from avg 4 it/s to 13 it/s, by (1) replacing part of atomicAdd by warp reduction primitive (2) fixing bugs for SSIM functions. The training costs 9 minutes for 7k iterations on Garden scene.

| Scene | PSNR from paper | PSNR from this repo | Rendering Speed (official) | Rendering Speed (Ours) |
| --- | --- | --- | --- | --- |
| Garden | 25.82(5k) | 24.91 (7k) | 160 FPS (avg MIPNeRF360) | 60 FPS |
| Garden | 25.82(5k) | 25.70 (7k) | 160 FPS (avg MIPNeRF360) | 25 FPS |



https://github.com/WangFeng18/3d-gaussian-splatting/assets/43294876/79703b5d-50ae-404b-96c9-c73690646f34



QuickStart

#### Install CUDA Extensions
```
# compile CUDA extension
pip install -e ./
```
#### Data Preparation
Put the colmap output in this folder, e.g., colmap_garden/sparse/0/, as well as the images.

### Training
```
python train.py --exp garden --grad_thresh 0.000004 --debug 1 --ssim_weight 0.1 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --split_thresh 0.08 # PSNR 24.75 SSIM 71.95 FPS 70 N_Gaussians 376467
python train.py --exp garden --grad_thresh 0.000004 --debug 1 --ssim_weight 0.1 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 # PSNR 25.03 SSIM 0.7541 FPS 40 N_GAUSSIANS 933918 
python train.py --exp garden --grad_thresh 0.000002 --debug 1 --ssim_weight 0.1 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --split_thresh 0.08 # PSNR 24.91 SSIM 73.18 FPS 64 N_GAUSSIANS 506627 GOOD

python train.py --exp garden2 --grad_thresh 0.000004 --debug 1 --ssim_weight 0.2 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --adaptive_control_end_iter 3000 --opa_init_value 0.05 --lr_factor_for_opa 20 # PSNR 25.55 SSIM 79.83 N_GAUSSIANS 2418528 FPS 24.68

CUDA_VISIBLE_DEVICES=3 python train.py --exp garden2 --grad_thresh 0.000004 --debug 1 --ssim_weight 0.2 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --adaptive_control_end_iter 3000 --opa_init_value 0.05 --lr_factor_for_opa 20 # PSNR 25.5586 SSIM 80.10 FPS 25.30 N_GAUSSIANS 2401413

python train.py --exp garden2 --grad_thresh 0.000004 --debug 1 --ssim_weight 0.2 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --adaptive_control_end_iter 3000 --opa_init_value 0.05 --lr_factor_for_opa 20 --lr_factor_for_scale 0.2 --lr_factor_for_quat 10 --split_thresh 0.05 #PSNR 24.896 SSIM 76.55 FPS 65 N_GAUSSIANS 765932

python train.py --exp garden2 --grad_thresh 0.000004 --debug 1 --ssim_weight 0.2 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --adaptive_control_end_iter 3000 --opa_init_value 0.05 --lr_factor_for_opa 20 --lr_factor_for_quat 10 # PSNR 25.6906 SSIM 80.66 FPS 24.68

python train.py --exp garden2 --grad_thresh 0.000004 --debug 1 --ssim_weight 0.2 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --adaptive_control_end_iter 3000 --opa_init_value 0.05 --lr_factor_for_opa 20 --lr_factor_for_scale 0.5 --lr_factor_for_quat 10 --split_thresh 0.05 # PSNR 25.3769 SSIM 0.7902 FPS 41.3186

CUDA_VISIBLE_DEVICES=3 python train.py --exp garden2 --grad_thresh 0.000004 --debug 1 --ssim_weight 0.2 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --adaptive_control_end_iter 3000 --opa_init_value 0.05 --lr_factor_for_opa 20 --lr_factor_for_quat 20 # PSNR 25.7021 SSIM 0.8052 FPS 25.3567

```

### Rendering With a GUI

```
python train.py --ckpt ckpt.pth --gui 1 --test 1
```
The GUI is based on [Viser](https://github.com/nerfstudio-project/viser) and written by [ZiLong Chen](https://github.com/heheyas).


The transforms folder are from [Viser](https://github.com/nerfstudio-project/viser)

### Link
Another good implementation for 3D gaussian splatting, by [Zilong Chen](https://github.com/heheyas/gaussian_splatting_3d)


### TODO List
[ ] Save optimization parameters and training metadata to text file for debugging
[ ] Move argument parser / options out of train.py
[ ] See if adaptive control can be controlled by change in filtered change in loss
[ ] Why does the number of gaussians scale with n_iters even on the same iteration? Seems like there are too many gaussians being generated leading to memory leaks etc. Most have low opacity?  I think issue is present in the base repository and isn't caused by my edits. 
[ ] Investigate / tune cloned gaussians
[ ] Investigate spherical harmonics and if they are incrementally added to the optimization
[ ] Are the gaussians radix sorted as described in the paper? 


### Configs

Current Best:
```
rm -rf default_experiment/ &&  python train.py --data ~/Downloads/garden --use_clone 0 --grad_thresh 0.000004 --debug 0 --ssim_weight 0.2 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --adaptive_control_end_iter 3000 --opa_init_value 0.05 --lr_factor_for_opa 20 --lr_factor_for_quat 20

N_GAUSSIAN: 2,164,080
PSNR: 25.16
~500s
```


Fast and still pretty good:
```
rm -rf default_experiment/ &&  python train.py --data ~/Downloads/garden --use_clone 0 --grad_thresh 0.00002 --debug 0 --ssim_weight 0.2 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --adaptive_control_end_iter 3000 --opa_init_value 0.05 --lr_factor_for_opa 20 --lr_factor_for_quat 20 --tile_culling_method prob2 --tile_culling_prob_thresh 0.15

N_GAUSSIANS: 587081
PSNR: 24.69
~200s 
```


Prob2 method is much faster since it does not need to recompute the 2d splat size for each tile
prob1: 
PSNR: 24.6573
Total Elapsed Time: 480.8974640369415
N_GAUSSIANS: 585076

prob2:
PSNR: 24.6649
Total Elapsed Time: 201.89806175231934
N_GAUSSIANS: 584628