import os
import torch
import torch.nn as nn
import gaussian
from utils import read_points3d_binary, read_cameras_binary, read_images_binary, q2r, jacobian_torch
from dataclasses import dataclass, field
import transforms as tf
import time
import math
import cv2
import numpy as np
from typing import Any, List
from einops import repeat

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
@dataclass
class Gaussian3ds:
    pos: torch.Tensor
    rgb: torch.Tensor
    opa: torch.Tensor
    quat: torch.Tensor = None
    scale: torch.Tensor = None
    cov: torch.Tensor = None

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
            self.scale.to(*args, **kwargs)
    
    def filte(self, mask):
        if self.quat is not None and self.scale is not None:
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
        S = self.scale
        RS = torch.bmm(R, S)
        RSSR = torch.bmm(RS, RS.permute(0,2,1))
        return RSSR
    
    def project(self, rot, tran, near, jacobian_calc):
        pos_cam_space = world_to_camera(self.pos, rot, tran)
        pos_img_space = camera_to_image(pos_cam_space)

        if jacobian_calc == "cuda":
            jacobian = torch.empty(pos_cam_space.shape[0], 3, 3).cuda()
            gaussian.jacobian(pos_cam_space, jacobian)
        else:
            jacobian = jacobian_torch(pos_cam_space)

        gaussian_3d_cov = self.get_gaussian_3d_conv()
        JW = torch.einsum("bij,bjk->bik", jacobian, rot.unsqueeze(dim=0))
        gaussian_2d_cov = torch.bmm(torch.bmm(JW, gaussian_3d_cov), JW.permute(0,2,1))[:, :2, :2]
        gaussian_3ds_image_space = Gaussian3ds(
            pos=pos_img_space,
            rgb=self.rgb,
            opa=self.opa,
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
    
    def create_tiles(self):
        self.tiles_left = torch.linspace(-self.padded_width/2, self.padded_width/2, self.n_tile_x + 1, device=self.device)[:-1]
        self.tiles_right = self.tiles_left + 16
        self.tiles_top = torch.linspace(-self.padded_height/2, self.padded_height/2, self.n_tile_y + 1, device=self.device)[:-1]
        self.tiles_bottom = self.tiles_top + 16
        self.tile_geo_length = 16 / self.focal_x

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
        near=1.1,
        jacobian_calc="cuda",
        render_downsample=2,
    ):
        self.device = torch.device("cuda")
        self.near = near
        self.jacobian_calc = jacobian_calc
        self.render_downsample = render_downsample
        assert jacobian_calc == "cuda" or jacobian_calc == "torch"

        self.points3d = read_points3d_binary(os.path.join(colmap_path, "points3D.bin"))
        self.cameras = read_cameras_binary(os.path.join(colmap_path,"cameras.bin"))
        self.images = read_images_binary(os.path.join(colmap_path,"images.bin"))
        self.image_path = image_path
        self.parse_imgs()
        self.vis_culling = Vis()
        self.tic = torch.cuda.Event(enable_timing=True)
        self.toc = torch.cuda.Event(enable_timing=True)

        _points = []
        _rgbs = []
        for pid, point in self.points3d.items():
            _points.append(torch.from_numpy(point.xyz))
            _rgbs.append(torch.from_numpy(point.rgb/255.))
            # _rgbs.append(torch.from_numpy(point.rgb))
            self.vis_culling.add_item(point.id, point.xyz, point.rgb, point.error, point.image_ids, point.point2D_idxs)
        self.gaussian_3ds = Gaussian3ds(
            pos=torch.stack(_points).to(torch.float32).to(self.device), # B x 3
            rgb = torch.stack(_rgbs).to(torch.float32).to(self.device), # B x 3
            opa = torch.ones(len(_points)).to(torch.float32).to(self.device)*1, # B
            quat = torch.Tensor([1, 0, 0, 0]).unsqueeze(dim=0).repeat(len(_points),1).to(torch.float32).to(self.device), # B x 4
            scale = repeat(torch.eye(3).unsqueeze(0), "b c d -> (r b) c d", r=len(_points)).to(torch.float32).to(self.device)*0.05
        )
        self.current_camera = None

        self.set_camera(0)
    
    def parse_imgs(self):
        img_ids = sorted([im.id for im in self.images.values()])
        self.w2c_quats = []
        self.w2c_trans = []
        self.cam_ids = []
        for img_id in img_ids:
            img = self.images[img_id]
            cam = self.cameras[img.camera_id]
            image_filename = os.path.join(self.image_path, img.name)
            if not os.path.exists(image_filename):
                continue
            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(img.qvec), img.tvec,
            )#.inverse()
            self.w2c_quats.append(T_world_camera.rotation().wxyz)
            self.w2c_trans.append(T_world_camera.translation())
            self.cam_ids.append(img.camera_id)

    def set_camera(self, idx):
        self.current_w2c_quat = torch.from_numpy(self.w2c_quats[idx]).to(self.device).to(torch.float32)
        self.current_w2c_tran = torch.from_numpy(self.w2c_trans[idx]).to(self.device).to(torch.float32)
        self.current_w2c_rot = q2r(self.current_w2c_quat.unsqueeze(0)).squeeze().to(torch.float32).to(self.device)

        if self.cameras[self.cam_ids[idx]] != self.current_camera:
            self.current_camera = self.cameras[self.cam_ids[idx]]
            width = self.current_camera.width / self.render_downsample
            height = self.current_camera.height / self.render_downsample
            focal_x = self.current_camera.params[0] / self.render_downsample
            focal_y = self.current_camera.params[1] / self.render_downsample
            self.tile_info = Tiles(width, height, focal_x, focal_y, self.device)
            self.tile_info_cpp = self.tile_info.create_tiles()
        print("current_camera info")
        print(self.current_camera)

    def project_and_culling(self):
        # project 3D to 2D
        print(f"number of gaussians {len(self.gaussian_3ds.pos)}")
        # self.tic.record()
        # gaussian_3ds_pos_camera_space = torch.empty_like(self.gaussian_3ds.pos)
        # gaussian.world2camera(self.gaussian_3ds.pos, self.current_w2c_rot, self.current_w2c_tran, gaussian_3ds_pos_camera_space)
        gaussian_3ds_pos_camera_space = world_to_camera(self.gaussian_3ds.pos, self.current_w2c_rot, self.current_w2c_tran)
        valid = gaussian_3ds_pos_camera_space[:,2] > self.near
        self.gaussian_3ds_valid = self.gaussian_3ds.filte(valid)
        self.gaussian_3ds_image_space = self.gaussian_3ds_valid.project(self.current_w2c_rot, self.current_w2c_tran, self.near, self.jacobian_calc)
        # culling
        culling_mask = (self.gaussian_3ds_image_space.pos[:, 0].abs() < (self.current_camera.width/2/self.current_camera.params[0]))  & \
                       (self.gaussian_3ds_image_space.pos[:, 1].abs() < (self.current_camera.height/2/self.current_camera.params[1]))
        self.culling_gaussian_3d_image_space = self.gaussian_3ds_image_space.filte(culling_mask)
        print(self.culling_gaussian_3d_image_space.cov)
        # exit()
        # self.toc.record()
        # torch.cuda.synchronize()
        # elapse = self.tic.elapsed_time(self.toc) / 1000
        # print(f"culling frustum costs {elapse}")

        print(f"number of gaussians culled {len(self.culling_gaussian_3d_image_space.pos)}")

        self.totalmask = torch.ones(len(self.gaussian_3ds.pos))
        self.totalmask[~valid.cpu()] = 0
        _id = torch.nonzero(self.totalmask)[~culling_mask.cpu()]
        self.totalmask[_id] = 0
        
    def render(self):
        tile_n_point = torch.zeros(len(self.tile_info), device=self.device, dtype=torch.int32)
        tile_gaussian_list = torch.ones(len(self.tile_info), len(self.culling_gaussian_3d_image_space.pos)//20, device=self.device, dtype=torch.int32) * -1
        print(f"tile geo lenth {self.tile_info.tile_geo_length}")

        self.tic.record()
        gaussian.calc_tile_list(
                self.culling_gaussian_3d_image_space._tocpp(), 
                self.tile_info_cpp,
                tile_n_point,
                tile_gaussian_list,
                (self.tile_info.tile_geo_length/1) ** 2,
        )
        
        gathered_list = torch.empty(tile_n_point.sum(), dtype=torch.int32, device=self.device)
        tile_ids_for_points = torch.empty(tile_n_point.sum(), dtype=torch.int32, device=self.device)
        tile_n_point_accum = torch.cat([torch.Tensor([0]).to(self.device), torch.cumsum(tile_n_point, 0)]).to(tile_n_point)
        max_points_for_tile = tile_n_point.max().item()
        print(f"Max Points For Tile {max_points_for_tile}")

        gaussian.gather_gaussians(
            tile_n_point_accum,
            tile_gaussian_list,
            gathered_list,
            tile_ids_for_points,
            int(max_points_for_tile),
        )
        self.tile_gaussians = self.culling_gaussian_3d_image_space.filte(gathered_list.long())
        print(tile_ids_for_points)
        # cat id and sort
        BASE = self.tile_gaussians.pos[..., 2].max()
        id_and_depth = self.tile_gaussians.pos[..., 2].to(torch.float32) + tile_ids_for_points.to(torch.float32) * (BASE+1)
        _, sort_indices = torch.sort(id_and_depth)
        print(BASE)
        print(_[:100])
        self.tile_gaussians = self.tile_gaussians.filte(sort_indices)
        
        rendered_image = torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)
        gaussian.draw(
            self.tile_gaussians._tocpp(),
            tile_n_point_accum,
            rendered_image,
            self.tile_info.focal_x,
            self.tile_info.focal_y,
        )
        self.toc.record()
        torch.cuda.synchronize()
        elapse = self.tic.elapsed_time(self.toc) / 1000
        print(elapse)

        print(rendered_image.shape)
        print(self.tile_info.focal_x)
        print(self.tile_info.focal_y)
        img_npy = rendered_image.clip(0,1).cpu().numpy()
        cv2.imwrite("test.png", (img_npy*255).astype(np.uint8)[...,::-1])
        
        # vis for tile culling
        _vis_tile_id = 550
        # order
        #_id = torch.nonzero(self.totalmask)[(tile_gaussian_list[_vis_tile_id][:tile_n_point[_vis_tile_id]]).cpu().long()]
        _id = torch.nonzero(self.totalmask)[(tile_gaussian_list[_vis_tile_id][:3]).cpu().long()]
        self.totalmask = torch.zeros(len(self.gaussian_3ds.pos))
        self.totalmask[_id] = 1
        print(self.totalmask.sum())
        print(tile_n_point[_vis_tile_id])
        self.vis_culling.write("point_filter.txt", self.totalmask)
        # totalmask = torch.ones(len(self.gaussian_3ds.pos))
        # totalmask[~valid.cpu()] = 0
        # _id = torch.nonzero(totalmask)[~culling_mask.cpu()]
        # totalmask[_id] = 0
        # self.vis_culling.write("point_filter.txt", totalmask)
        # print(totalmask.sum())

        # print(self.tile_gaussians.pos.size())
        # print(sort_indices[300:400])
        # print(tile_ids_for_points[300:400])
        # print(self.tile_gaussians.pos[300:400, 2])
        # print(id_and_depth[300:400])

if __name__ == "__main__":
    test = Splatter(os.path.join("colmap_garden/sparse/0/"), "colmap_garden/images_8/")
    t1 = time.time()
    test.project_and_culling()
    test.render()
