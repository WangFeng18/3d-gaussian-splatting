#include <torch/extension.h>
// #include "include/gaussian.h"
#include "include/common.hpp"

void culling(torch::Tensor pos, torch::Tensor rgb, torch::Tensor quatenions, torch::Tensor scales, torch::Tensor w2c_quat, torch::Tensor w2c_tran);
void world2camera(torch::Tensor pos, torch::Tensor rot, torch::Tensor trans, torch::Tensor res);
void world2camera_backward(torch::Tensor grad_out, torch::Tensor rot, torch::Tensor grad_inp);
void jacobian(torch::Tensor pos_camera_space, torch::Tensor jacobian);
void calc_tile_list(Gaussian3ds & gaussians_image_space, Tiles & tile_info, torch::Tensor tile_n_point, torch::Tensor tile_gaussian_list, float thresh, int method, float tile_length_x, float tile_length_y, int n_tiles_x, int n_tiles_y, float leftmost, float topmost);
void gather_gaussians(torch::Tensor tile_n_point_accum, torch::Tensor tile_gaussian_list, torch::Tensor gathered_list, torch::Tensor tile_ids_for_points, int max_points_for_tile);
// void draw(Gaussian3ds & tile_sorted_gaussians, torch::Tensor tile_n_point_accum, torch::Tensor res, float focal_x, float focal_y);
// void draw(torch::Tensor gaussian_pos, torch::Tensor gaussian_rgb, torch::Tensor gaussian_opa, torch::Tensor gaussian_cov, torch::Tensor tile_n_point_accum, torch::Tensor res, float focal_x, float focal_y, bool weight_normalize, bool sigmoid, bool fast); //, torch::Tensor rays_o, torch::Tensor lefttop_pos, torch::Tensor vec_dx, torch::Tensor vec_dy);
// void draw_backward(torch::Tensor gaussian_pos, torch::Tensor gaussian_rgb, torch::Tensor gaussian_opa, torch::Tensor gaussian_cov, torch::Tensor tile_n_point_accum, torch::Tensor output, torch::Tensor grad_output, torch::Tensor grad_pos, torch::Tensor grad_rgb, torch::Tensor grad_opa, torch::Tensor grad_cov, float focal_x, float focal_y, bool weight_normalize, bool sigmoid, bool fast);//, torch::Tensor rays_o, torch::Tensor lefttop_pos, torch::Tensor vec_dx, torch::Tensor vec_dy);
void draw(torch::Tensor gaussian_pos, torch::Tensor gaussian_rgb, torch::Tensor gaussian_opa, torch::Tensor gaussian_cov, torch::Tensor tile_n_point_accum, torch::Tensor res, float focal_x, float focal_y, bool weight_normalize, bool sigmoid, bool fast, torch::Tensor rays_o, torch::Tensor lefttop_pos, torch::Tensor vec_dx, torch::Tensor vec_dy, bool use_sh_coeff);
void draw_backward(torch::Tensor gaussian_pos, torch::Tensor gaussian_rgb, torch::Tensor gaussian_opa, torch::Tensor gaussian_cov, torch::Tensor tile_n_point_accum, torch::Tensor output, torch::Tensor grad_output, torch::Tensor grad_pos, torch::Tensor grad_rgb, torch::Tensor grad_opa, torch::Tensor grad_cov, float focal_x, float focal_y, bool weight_normalize, bool sigmoid, bool fast, torch::Tensor rays_o, torch::Tensor lefttop_pos, torch::Tensor vec_dx, torch::Tensor vec_dy, bool use_sh_coeff);

void global_culling(torch::Tensor pos, torch::Tensor quat, torch::Tensor scale, torch::Tensor current_rot, torch::Tensor current_tran, torch::Tensor res_pos, torch::Tensor res_cov, torch::Tensor culling_mask, float near, float half_width, float half_height);
void global_culling_backward(torch::Tensor pos, torch::Tensor quat, torch::Tensor scale, torch::Tensor current_rot, torch::Tensor current_tran, torch::Tensor gradout_pos, torch::Tensor gradout_cov, torch::Tensor culling_mask, torch::Tensor gradinput_pos, torch::Tensor gradinput_quat, torch::Tensor gradinput_scale);
// void cat_ids(torch::Tensor depth, torch::Tensor ids, torch::Tensor res);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("matmul", &matmul, "matmul forward (CUDA)");
    m.def("culling", &culling, "gaussian culling (CUDA)");
    m.def("world2camera", &world2camera, "world to camera fast (CUDA)");
    m.def("world2camera_backward", &world2camera_backward, "world to camera fast backward (CUDA)");
    m.def("jacobian", &jacobian, "jacobian (CUDA)");

    py::class_<Tiles>(m, "Tiles")
      .def(py::init<>())
      .def_readwrite("top", &Tiles::top)
      .def_readwrite("bottom", &Tiles::bottom)
      .def_readwrite("left", &Tiles::left)
      .def_readwrite("right", &Tiles::right);

    py::class_<Gaussian3ds>(m, "Gaussian3ds")
      .def(py::init<>())
      .def_readwrite("pos", &Gaussian3ds::pos)
      .def_readwrite("rgb", &Gaussian3ds::rgb)
      .def_readwrite("opa", &Gaussian3ds::opa)
      .def_readwrite("quat", &Gaussian3ds::quat)
      .def_readwrite("scale", &Gaussian3ds::scale)
      .def_readwrite("cov", &Gaussian3ds::cov);

    m.def("calc_tile_list", &calc_tile_list, "calc tile list (CUDA)");
    m.def("gather_gaussians", &gather_gaussians, "gather gaussian (CUDA)");
    m.def("draw", &draw, "draw (CUDA)");
    m.def("draw_backward", &draw_backward, "draw backward (CUDA)");
    m.def("global_culling", &global_culling, "global culling (CUDA)");
    m.def("global_culling_backward", &global_culling_backward, "global culling backward (CUDA)");
}