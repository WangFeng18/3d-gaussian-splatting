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
```

### Rendering With a GUI

```
python train.py --ckpt ckpt.pth --gui 1 --test 1
```
The GUI is based on [Viser](https://github.com/nerfstudio-project/viser) and written by [ZiLong Chen](https://github.com/heheyas).


The transforms folder are from [Viser](https://github.com/nerfstudio-project/viser)

### Link
Another good implementation for 3D gaussian splatting, by [Zilong Chen](https://github.com/heheyas/gaussian_splatting_3d)

