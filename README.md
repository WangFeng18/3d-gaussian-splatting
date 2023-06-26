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



https://github.com/WangFeng18/3d-gaussian-splatting/assets/43294876/79703b5d-50ae-404b-96c9-c73690646f34



QuickStart

#### Install CUDA Extensions
```
# compile CUDA extension
pip install -e ./
```
#### Data Preparation
Put the colmap output in this folder, e.g., colmap_garden/sparse/0/, as well as the images.

### Traning
```
python train.py # for 7k
python train.py --exp garden_sh --grad_thresh 0.000004 --debug 1 --ssim_weight 0.1 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --split_thresh 0.08 # PSNR 24.75 SSIM 71.95 FPS 70 N_Gaussians 376467
python train.py --exp garden_sh --grad_thresh 0.000004 --debug 1 --ssim_weight 0.1 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 # PSNR 25.03 SSIM 0.7541 FPS 40 N_GAUSSIANS 933918 
CUDA_VISIBLE_DEVICES=3 python train.py --exp garden --grad_thresh 0.000002 --debug 1 --ssim_weight 0.1 --lr 0.002 --use_sh_coeff 0 --grad_accum_method mean --grad_accum_iters 300 --split_thresh 0.08 # PSNR 24.91 SSIM 73.18 FPS 64 N_GAUSSIANS 506627 GOOD
```

### Rendering With a GUI

```
python train.py --ckpt ckpt.pth --gui 1 --test 1
```
The GUI is based on [Viser](https://github.com/nerfstudio-project/viser) and written by [ZiLong Chen](https://github.com/heheyas).


The transforms folder are from [Viser](https://github.com/nerfstudio-project/viser)

### Link
Another good implementation for 3D gaussian splatting, by [Zilong Chen](https://github.com/heheyas/gaussian_splatting_3d)

