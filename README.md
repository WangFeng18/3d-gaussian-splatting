# 3d-Gaussian-Splatting 
An unofficial Implementation of 3D Gaussian Splatting for Real-Time Radiance Field Rendering [SIGGRAPH 2023].

We implement the 3d gaussian splatting methods through PyTorch with CUDA extensions, including the global culling, tile-based culling and rendering forward/backward codes.

| Scene | PSNR from paper | PSNR from this repo | Rendering Speed (official) | Rendering Speed (Ours) |
| --- | --- | --- | --- | --- |
| Garden | 25.82(5k) | 24.28 (7k) | 160 FPS (avg MIPNeRF360) | 60 FPS |

https://github.com/WangFeng18/3d-gaussian-splatting/assets/43294876/761951e6-b2b7-43d5-becf-1701a3504d0a

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
