import os
import math
import torch
import torch.nn as nn
import gaussian
from utils import read_points3d_binary, read_cameras_binary, read_images_binary, q2r, jacobian_torch, initialize_sh, inverse_sigmoid, inverse_sigmoid_torch, Timer, sample_two_point
from dataclasses import dataclass, field
import transforms as tf
import time
import math
import cv2
import numpy as np
from typing import Any, List
from einops import repeat
from renderer import draw, trunc_exp, global_culling
from tqdm import tqdm

@dataclass
class Vis:
    point3D_id: List = field(default_factory=list)
    xyz: List = field(default_factory=list)
    rgb: List = field(default_factory=list)
    error: List = field(default_factory=list)
    image_ids: List = field(default_factory=list)
    point2D_idxs: List = field(default_factory=list)

    def add_item(self, id, xyz, rgb, error, image_ids, point2D_idxs):
        self.point3D_id.append(id)
        self.xyz.append(xyz)
        self.rgb.append(rgb)
        self.error.append(error)
        self.image_ids.append(image_ids)
        self.point2D_idxs.append(point2D_idxs)
    
    def write(self, path, mask):
        lines = []
        for i in range(len(self.point3D_id)):
            if not mask[i]:
                continue
            s = ""
            s += str(self.point3D_id[i]) + " "
            s += " ".join([str(it) for it in self.xyz[i]]) + " "
            s += " ".join([str(it) for it in self.rgb[i]]) + " "
            s += str(self.error[i]) + " "
            _ids = list(self.image_ids[i]) 
            _ids2 = list(self.point2D_idxs[i])
            ids = []
            for j in range(len(_ids)):
                ids.append(_ids[j])
                ids.append(_ids2[j])
            s += " ".join([str(it) for it in ids]) + "\n"
            lines.append(s)
        with open(path, "w") as f:
            f.writelines(lines)

    
def world_to_camera(points, rot, tran):
    # r = torch.empty_like(points)
    # gaussian.world2camera(points, rot, tran, r)
    # return r
    _r = points @ rot.T + tran.unsqueeze(0)
    return _r

def camera_to_image(points_camera_space):
    points_image_space = [
        points_camera_space[:,0]/points_camera_space[:,2],
        points_camera_space[:,1]/points_camera_space[:,2],
        points_camera_space.norm(dim=-1),
    ]
    return torch.stack(points_image_space, dim=-1)


#TODO ensure if the camera coordinate system is normalized
class Gaussian3ds(nn.Module):
    def __init__(self, pos, rgb, opa, quat=None, scale=None, cov=None, init_values=False):
        super().__init__()
        self.init_values = init_values
        if init_values:
            self.pos = nn.parameter.Parameter(pos)
            self.rgb = nn.parameter.Parameter(rgb)
            self.opa = nn.parameter.Parameter(opa)
            self.quat = quat if quat is None else nn.parameter.Parameter(quat)
            self.scale = scale if scale is None else nn.parameter.Parameter(scale)
            self.cov = cov if cov is None else nn.parameter.Parameter(cov)
        else:
            self.pos = pos
            self.rgb = rgb
            self.opa = opa
            self.quat = quat
            self.scale = scale
            self.cov = cov

    def to_scale_matrix(self):
        return torch.diag_embed(self.scale)

    def _tocpp(self):
        _cobj = gaussian.Gaussian3ds()
        _cobj.pos = self.pos
        _cobj.rgb = self.rgb
        _cobj.opa = self.opa
        _cobj.cov = self.cov
        return _cobj

    def to(self, *args, **kwargs):
        self.pos.to(*args, **kwargs)
        self.rgb.to(*args, **kwargs)
        self.opa.to(*args, **kwargs)
        if self.quat is not None:
            self.quat.to(*args, **kwargs)
        if self.scale is not None:
            self.scale.to(*args, **kwargs)
        if self.cov is not None:
            self.cov.to(*args, **kwargs)
    
    def filte(self, mask):
        if self.quat is not None and self.scale is not None:
            assert self.cov is None
            return Gaussian3ds(
                pos=self.pos[mask],
                rgb=self.rgb[mask],
                opa=self.opa[mask],
                quat=self.quat[mask],
                scale=self.scale[mask],
            )
        else:
            assert self.cov is not None
            return Gaussian3ds(
                pos=self.pos[mask],
                rgb=self.rgb[mask],
                opa=self.opa[mask],
                cov=self.cov[mask],
            )
    
    def get_gaussian_3d_conv(self):
        R = q2r(self.quat)
        _scale = self.scale.abs()+1e-4
        # _scale = torch.clamp(_scale, min=1e-4, max=0.1)
        S = torch.diag_embed(_scale)
        # S = torch.diag_embed(trunc_exp(self.scale))
        RS = torch.bmm(R, S)
        RSSR = torch.bmm(RS, RS.permute(0,2,1))
        return RSSR
    
    # def stable_cov(self):
        # return self.cov + 1e-2*torch.eye(2).unsqueeze(dim=0).to(self.cov)
    
    def reset_opa(self):
        torch.nn.init.uniform_(self.opa, a=inverse_sigmoid(0.1), b=inverse_sigmoid(0.11))
    
    def adaptive_control(self, grad, taus, delete_thresh):
        # grad: B x 3
        # densification
        # 1. delete gaussians with small opacities
        assert self.init_values # only for the initial gaussians
        print(inverse_sigmoid(0.1))
        print(self.opa.min())
        print(self.opa.max())
        _mask = (self.opa > inverse_sigmoid(0.1)) & (self.scale.norm(dim=-1) < delete_thresh)
        self.pos = nn.parameter.Parameter(self.pos.detach()[_mask])
        self.rgb = nn.parameter.Parameter(self.rgb.detach()[_mask])
        self.opa = nn.parameter.Parameter(self.opa.detach()[_mask])
        self.quat = nn.parameter.Parameter(self.quat.detach()[_mask])
        self.scale = nn.parameter.Parameter(self.scale.detach()[_mask])
        grad = grad[_mask]
        print("DELETE: {} Gaussians".format((~_mask).sum()))
        # 2. clone or split
        densify_mask = grad.abs().max(-1)[0] > 0.0002
        cat_pos = [self.pos.detach()]
        cat_rgb = [self.rgb.detach()]
        cat_opa = [self.opa.detach()]
        cat_quat = [self.quat.detach()]
        cat_scale = [self.scale.detach()]
        if densify_mask.any():
            scale_norm = self.scale.norm(dim=-1)
            split_mask = scale_norm > taus
            clone_mask = scale_norm <= taus
            split_mask = split_mask & densify_mask
            clone_mask = clone_mask & densify_mask
            if clone_mask.any():
                cloned_pos = self.pos[clone_mask].clone().detach()
                # cloned_pos -= grad[clone_mask] * 0.01
                cloned_pos -= grad[clone_mask] * 10
                cloned_rgb = self.rgb[clone_mask].clone().detach()
                cloned_opa = self.opa[clone_mask].clone().detach()
                cloned_quat = self.quat[clone_mask].clone().detach()
                cloned_scale = self.scale[clone_mask].clone().detach()
                print("CLONE: {} Gaussians".format(cloned_pos.shape[0]))
                cat_pos.append(cloned_pos)
                cat_rgb.append(cloned_rgb)
                cat_opa.append(cloned_opa)
                cat_quat.append(cloned_quat)
                cat_scale.append(cloned_scale)

            if split_mask.any():
                _scale = self.scale.detach() 
                _scale[split_mask] /= 1.6

                cat_scale[0][split_mask] /= 1.6
                self.scale = nn.parameter.Parameter(_scale)

                # sampling two positions
                this_cov = self.get_gaussian_3d_conv()[split_mask]
                p1, p2 = sample_two_point(self.pos[split_mask], this_cov)

                # split_pos = self.pos[split_mask].clone().detach()
                # split_pos -= grad[split_mask] * 0.01
                cat_pos[0][split_mask] = p1.detach()
                split_pos = p2.detach()
                split_rgb = self.rgb[split_mask].clone().detach()
                split_opa = self.opa[split_mask].clone().detach()
                split_quat = self.quat[split_mask].clone().detach()
                split_scale = self.scale[split_mask].clone().detach()
                print("SPLIT : {} Gaussians".format(split_pos.shape[0]))
                cat_pos.append(split_pos)
                cat_rgb.append(split_rgb)
                cat_opa.append(split_opa)
                cat_quat.append(split_quat)
                cat_scale.append(split_scale)
            self.pos = nn.parameter.Parameter(torch.cat(cat_pos))
            self.rgb = nn.parameter.Parameter(torch.cat(cat_rgb))
            self.opa = nn.parameter.Parameter(torch.cat(cat_opa))
            self.quat = nn.parameter.Parameter(torch.cat(cat_quat))
            self.scale = nn.parameter.Parameter(torch.cat(cat_scale))
    
    def project(self, rot, tran, near, jacobian_calc):

        with Timer("        w2c"):
            pos_cam_space = world_to_camera(self.pos, rot, tran)
        with Timer("        c2i"):
            pos_img_space = camera_to_image(pos_cam_space)

        with Timer("        jacobian"):
            if jacobian_calc == "cuda":
                jacobian = torch.empty(pos_cam_space.shape[0], 3, 3).cuda()
                gaussian.jacobian(pos_cam_space, jacobian)
            else:
                jacobian = jacobian_torch(pos_cam_space)

        with Timer("        cov"):
            gaussian_3d_cov = self.get_gaussian_3d_conv()
            JW = torch.einsum("bij,bjk->bik", jacobian, rot.unsqueeze(dim=0))
            gaussian_2d_cov = torch.bmm(torch.bmm(JW, gaussian_3d_cov), JW.permute(0,2,1))[:, :2, :2]

        with Timer("        last"):
            gaussian_3ds_image_space = Gaussian3ds(
                pos=pos_img_space,
                rgb=self.rgb.sigmoid(),
                opa=self.opa.sigmoid(),
                cov=gaussian_2d_cov,
            )
        return gaussian_3ds_image_space
        
class Tiles:
    def __init__(self, width, height, focal_x, focal_y, device):
        self.width = width 
        self.height = height
        self.padded_width = int(math.ceil(self.width/16)) * 16
        self.padded_height = int(math.ceil(self.height/16)) * 16
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.n_tile_x = self.padded_width // 16
        self.n_tile_y = self.padded_height // 16
        self.device = device
    
    def crop(self, image):
        # image: padded_height x padded_width x 3
        # output: height x width x 3
        top = int(self.padded_height - self.height)//2 
        left = int(self.padded_width - self.width)//2 
        return image[top:top+int(self.height), left:left+int(self.width), :]
        # return image[top:top+int(self.height)-1, left:left+int(self.width)-1, :]
    
    def create_tiles(self):
        self.tiles_left = torch.linspace(-self.padded_width/2, self.padded_width/2, self.n_tile_x + 1, device=self.device)[:-1]
        self.tiles_right = self.tiles_left + 16
        self.tiles_top = torch.linspace(-self.padded_height/2, self.padded_height/2, self.n_tile_y + 1, device=self.device)[:-1]
        self.tiles_bottom = self.tiles_top + 16
        self.tile_geo_length_x = 16 / self.focal_x
        self.tile_geo_length_y = 16 / self.focal_y
        self.leftmost = -self.padded_width/2/self.focal_x
        self.topmost = -self.padded_height/2/self.focal_y

        self.tiles_left = self.tiles_left/self.focal_x
        self.tiles_top = self.tiles_top/self.focal_y
        self.tiles_right = self.tiles_right/self.focal_x
        self.tiles_bottom = self.tiles_bottom/self.focal_y

        self.tiles_left = repeat(self.tiles_left, "b -> (c b)", c=self.n_tile_y)
        self.tiles_right = repeat(self.tiles_right, "b -> (c b)", c=self.n_tile_y)

        self.tiles_top = repeat(self.tiles_top, "b -> (b c)", c=self.n_tile_x)
        self.tiles_bottom = repeat(self.tiles_bottom, "b -> (b c)", c=self.n_tile_x)

        _tile = gaussian.Tiles()
        _tile.top = self.tiles_top
        _tile.bottom = self.tiles_bottom
        _tile.left = self.tiles_left
        _tile.right = self.tiles_right
        return _tile
    
    def __len__(self):
        return self.tiles_top.shape[0]


class Splatter(nn.Module):
    def __init__(self, 
        colmap_path, 
        image_path, 
        near=0.3,
        #near=1.1,
        jacobian_calc="cuda",
        render_downsample=1,
        use_sh_coeff=False,
        render_weight_normalize=False,
        opa_init_value=0.1,
        scale_init_value=0.02,
        tile_culling_method="dist", # dist or prob
        tile_culling_dist_thresh=0.5,
        tile_culling_prob_thresh=0.1,
        debug=1,
    ):
        super().__init__()
        self.device = torch.device("cuda")
        self.use_sh_coeff = use_sh_coeff
        self.near = near
        self.jacobian_calc = jacobian_calc
        self.render_downsample = render_downsample
        self.render_weight_normalize = render_weight_normalize
        self.tile_culling_method = tile_culling_method
        self.tile_culling_dist_thresh = tile_culling_dist_thresh
        self.tile_culling_prob_thresh = tile_culling_prob_thresh
        self.debug = debug
        assert jacobian_calc == "cuda" or jacobian_calc == "torch"

        self.points3d = read_points3d_binary(os.path.join(colmap_path, "points3D.bin"))
        self.cameras = read_cameras_binary(os.path.join(colmap_path,"cameras.bin"))
        self.images_info = read_images_binary(os.path.join(colmap_path,"images.bin"))
        self.image_path = image_path
        self.parse_imgs()
        # self.vis_culling = Vis()
        self.tic = torch.cuda.Event(enable_timing=True)
        self.toc = torch.cuda.Event(enable_timing=True)

        _points = []
        _rgbs = []
        for pid, point in self.points3d.items():
            _points.append(torch.from_numpy(point.xyz))
            _rgbs.append(inverse_sigmoid_torch(torch.from_numpy(point.rgb/255.)))
            # _rgbs.append(torch.from_numpy(point.rgb))
            # self.vis_culling.add_item(point.id, point.xyz, point.rgb, point.error, point.image_ids, point.point2D_idxs)
        rgb = torch.stack(_rgbs).to(torch.float32).to(self.device) # B x 3
        if self.use_sh_coeff:
            rgb = initialize_sh(rgb)
        self.gaussian_3ds = Gaussian3ds(
            pos=torch.stack(_points).to(torch.float32).to(self.device), # B x 3
            rgb = rgb, # B x 3 or 27
            opa = torch.ones(len(_points)).to(torch.float32).to(self.device)*inverse_sigmoid(opa_init_value), # B
            quat = torch.Tensor([1, 0, 0, 0]).unsqueeze(dim=0).repeat(len(_points),1).to(torch.float32).to(self.device), # B x 4
            # quat = torch.Tensor([1, 1, 2, 1]).unsqueeze(dim=0).repeat(len(_points),1).to(torch.float32).to(self.device), # B x 4
            scale = torch.ones(len(_points), 3).to(torch.float32).to(self.device)*scale_init_value,
            init_values=True,
        )
        self.current_camera = None
        self.set_camera(0)
    
    def parse_imgs(self):
        img_ids = sorted([im.id for im in self.images_info.values()])
        self.w2c_quats = []
        self.w2c_rots = []
        self.w2c_trans = []
        self.cam_ids = []
        self.imgs = []
        for img_id in tqdm(img_ids):
            img_info = self.images_info[img_id]
            cam = self.cameras[img_info.camera_id]
            image_filename = os.path.join(self.image_path, img_info.name)
            if not os.path.exists(image_filename):
                continue
            _current_image = cv2.imread(image_filename)
            _current_image = cv2.cvtColor(_current_image, cv2.COLOR_BGR2RGB)
            self.imgs.append(torch.from_numpy(_current_image).to(torch.uint8).to(self.device))

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(img_info.qvec), img_info.tvec,
            )#.inverse()
            self.w2c_quats.append(torch.from_numpy(T_world_camera.rotation().wxyz).to(torch.float32).to(self.device))
            self.w2c_trans.append(torch.from_numpy(T_world_camera.translation()).to(torch.float32).to(self.device))
            self.w2c_rots.append(q2r(self.w2c_quats[-1].unsqueeze(0)).squeeze().to(torch.float32).to(self.device))
            self.cam_ids.append(img_info.camera_id)

    def set_camera(self, idx):
        with Timer("    set image"):
            with Timer("        set image 1"):
                self.current_w2c_quat = self.w2c_quats[idx]
            with Timer("        set image 2"):
                self.current_w2c_tran = self.w2c_trans[idx]
            with Timer("        set image 3"):
                self.current_w2c_rot = self.w2c_rots[idx]
            with Timer("        set image 4"):
                self.ground_truth = self.imgs[idx].to(torch.float16)/255.
        with Timer("    set camera"):
            if self.cameras[self.cam_ids[idx]] != self.current_camera:
                self.current_camera = self.cameras[self.cam_ids[idx]]
                width = self.current_camera.width / self.render_downsample
                height = self.current_camera.height / self.render_downsample
                focal_x = self.current_camera.params[0] / self.render_downsample
                focal_y = self.current_camera.params[1] / self.render_downsample
                self.tile_info = Tiles(math.ceil(width), math.ceil(height), focal_x, focal_y, self.device)
                self.tile_info_cpp = self.tile_info.create_tiles()
        # print("current_camera info")
        # print(self.current_camera)

    def project_and_culling(self, cudaculling=False):
        # project 3D to 2D
        # print(f"number of gaussians {len(self.gaussian_3ds.pos)}")
        # self.tic.record()
        if cudaculling:
            with Timer(" frustum cuda"):
                # self.gaussian_3ds.rgb = self.gaussian_3ds.rgb.sigmoid()
                # self.gaussian_3ds.opa = self.gaussian_3ds.opa.sigmoid()
                # self.gaussian_3ds.quat = self.gaussian_3ds.quat / self.gaussian_3ds.quat.norm(dim=-1, keepdim=True)
                # self.gaussian_3ds.scale = self.gaussian_3ds.scale.abs()+1e-5
                _pos, _cov, _culling_mask = global_culling(
                    self.gaussian_3ds.pos, 
                    self.gaussian_3ds.quat/self.gaussian_3ds.quat.norm(dim=-1, keepdim=True),
                    self.gaussian_3ds.scale.abs()+1e-4, 
                    self.current_w2c_rot, 
                    self.current_w2c_tran, 
                    self.near, 
                    self.current_camera.width/2/self.current_camera.params[0], 
                    self.current_camera.height/2/self.current_camera.params[1],
                )

                self.culling_gaussian_3d_image_space = Gaussian3ds(
                    pos=_pos[_culling_mask.bool()],
                    cov=_cov[_culling_mask.bool()],
                    rgb=self.gaussian_3ds.rgb[_culling_mask.bool()].sigmoid(),
                    opa=self.gaussian_3ds.opa[_culling_mask.bool()].sigmoid(),
                )
                print(len(self.culling_gaussian_3d_image_space.pos))
        else:    
            with Timer(" frustum pytorch", verbose=True):
                with Timer("    frustum 11"):
                    gaussian_3ds_pos_camera_space = world_to_camera(self.gaussian_3ds.pos, self.current_w2c_rot, self.current_w2c_tran)
                with Timer("    frustum 12"):
                    valid = gaussian_3ds_pos_camera_space[:,2] > self.near
                with Timer("    frustum 13"):
                    self.gaussian_3ds_valid = self.gaussian_3ds.filte(valid)
                with Timer("    frustum 2"):
                    self.gaussian_3ds_image_space = self.gaussian_3ds_valid.project(self.current_w2c_rot, self.current_w2c_tran, self.near, self.jacobian_calc)
                # culling
                with Timer("    frumstum 3"):
                    culling_mask = (self.gaussian_3ds_image_space.pos[:, 0].abs() < (self.current_camera.width/2/self.current_camera.params[0]))  & \
                                (self.gaussian_3ds_image_space.pos[:, 1].abs() < (self.current_camera.height/2/self.current_camera.params[1]))
                    self.culling_gaussian_3d_image_space = self.gaussian_3ds_image_space.filte(culling_mask)
            # print(_pos.shape)
            # print(_cov.shape)
            # print(_culling_mask.sum())
            # print(self.culling_gaussian_3d_image_space.pos.shape)
            # print((self.culling_gaussian_3d_image_space.cov - _cov[_culling_mask.bool()]).abs().mean())
            # print((self.culling_gaussian_3d_image_space.pos - _pos[_culling_mask.bool()]).abs().max())
            # print((self.culling_gaussian_3d_image_space.cov + _cov[_culling_mask.bool()]).abs().mean())
            # print((self.culling_gaussian_3d_image_space.pos + _pos[_culling_mask.bool()]).abs().max())
        
    def render(self, out_write=True):
        # self.tic.record()
        with Timer("     culling tiles", debug=self.debug):
            tile_n_point = torch.zeros(len(self.tile_info), device=self.device, dtype=torch.int32)
            tile_gaussian_list = torch.ones(len(self.tile_info), len(self.culling_gaussian_3d_image_space.pos)//10, device=self.device, dtype=torch.int32) * -1
            _method_config = {"dist": 0, "prob": 1, "prob2": 2}
            gaussian.calc_tile_list(
                    self.culling_gaussian_3d_image_space._tocpp(), 
                    self.tile_info_cpp,
                    tile_n_point,
                    tile_gaussian_list,
                    # (self.tile_info.tile_geo_length/1.0) ** 2,
                    # (self.tile_info.tile_geo_length/0.5) ** 2, training
                    (self.tile_info.tile_geo_length_x/self.tile_culling_dist_thresh) ** 2 if self.tile_culling_method == "dist" else self.tile_culling_prob_thresh,
                    _method_config[self.tile_culling_method],
                    self.tile_info.tile_geo_length_x,
                    self.tile_info.tile_geo_length_y,
                    self.tile_info.n_tile_x,
                    self.tile_info.n_tile_y,
                    self.tile_info.leftmost,
                    self.tile_info.topmost,
                    # 0.01,
            )
            
        with Timer("     gather culled tiles", debug=self.debug):
            gathered_list = torch.empty(tile_n_point.sum(), dtype=torch.int32, device=self.device)
            tile_ids_for_points = torch.empty(tile_n_point.sum(), dtype=torch.int32, device=self.device)
            tile_n_point_accum = torch.cat([torch.Tensor([0]).to(self.device), torch.cumsum(tile_n_point, 0)]).to(tile_n_point)
            max_points_for_tile = tile_n_point.max().item()
            # print(max_points_for_tile)
            gaussian.gather_gaussians(
                tile_n_point_accum,
                tile_gaussian_list,
                gathered_list,
                tile_ids_for_points,
                int(max_points_for_tile),
            )
            self.tile_gaussians = self.culling_gaussian_3d_image_space.filte(gathered_list.long())
            self.n_tile_gaussians = len(self.tile_gaussians.pos)
            self.n_gaussians = len(self.gaussian_3ds.pos)

        with Timer("     sorting", debug=self.debug):
            # cat id and sort
            BASE = self.tile_gaussians.pos[..., 2].max()
            id_and_depth = self.tile_gaussians.pos[..., 2].to(torch.float32) + tile_ids_for_points.to(torch.float32) * (BASE+1)
            _, sort_indices = torch.sort(id_and_depth)
            self.tile_gaussians = self.tile_gaussians.filte(sort_indices)

        with Timer("     rendering", debug=self.debug):
            rendered_image = draw(
                self.tile_gaussians.pos,
                self.tile_gaussians.rgb,
                self.tile_gaussians.opa,
                self.tile_gaussians.cov,
                tile_n_point_accum,
                self.tile_info.padded_height,
                self.tile_info.padded_width,
                self.tile_info.focal_x,
                self.tile_info.focal_y,
                self.render_weight_normalize,
                False,
                self.use_sh_coeff,
            ) 

        with Timer("    write out", debug=self.debug):
            if out_write:
                img_npy = rendered_image.clip(0,1).detach().cpu().numpy()
                cv2.imwrite("test.png", (img_npy*255).astype(np.uint8)[...,::-1])

        return rendered_image

    def forward(self, camera_id=None, record_view_space_pos_grad=False, cudaculling=False):
        # print(self.gaussian_3ds.opa.max())
        # print(self.gaussian_3ds.opa.min())
        with Timer("forward", debug=self.debug):
            with Timer("set camera"):
                self.set_camera(camera_id)
            with Timer("frustum culling", debug=self.debug):
                self.project_and_culling(cudaculling)
            with Timer("render function", debug=self.debug):
                padded_render_img = self.render(out_write=False)
            with Timer("crop", debug=self.debug):
                padded_render_img = torch.clamp(padded_render_img, 0, 1)
                ret = self.tile_info.crop(padded_render_img)
        return ret

if __name__ == "__main__":
    test = Splatter(os.path.join("colmap_garden/sparse/0/"), "colmap_garden/images_2/", render_weight_normalize=False, jacobian_calc="torch", render_downsample=2, opa_init_value=0.8, scale_init_value=0.2)
    test.forward(camera_id=0, cudaculling=False)
    loss = (test.ground_truth - test.forward(camera_id=0)).abs().mean()
    loss.backward()
    print(test.gaussian_3ds.rgb.grad.max())
