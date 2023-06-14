#include <stdio.h>
#include "include/common.hpp"
#include <torch/torch.h>
#include <cuda_runtime.h>

// struct Tile{

// };

void culling(torch::Tensor pos, torch::Tensor rgb, torch::Tensor quatenions, torch::Tensor scales, torch::Tensor w2c_quat, torch::Tensor w2c_tran){
    printf("hellow\n");
}

__global__ void jacobian_kernel(
    const float* pos_camera_space, 
    float* jacobian,
    uint32_t B
){
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= B) return;
    pos_camera_space += tid * 3;
    jacobian += tid * 9;
    float _res[9];
    float u0 = pos_camera_space[0];
    float u1 = pos_camera_space[1];
    float u2 = pos_camera_space[2];

    _res[0] = 1/u2;
    _res[1] = 0;
    _res[2] = -u0/(u2*u2);
    _res[3] = 0;
    _res[4] = 1/u2;
    _res[5] = -u1/(u2*u2);
    float _rsqr = rsqrtf(u0*u0+u1*u1+u2*u2);
    _res[6] = _rsqr * u0;
    _res[7] = _rsqr * u1;
    _res[8] = _rsqr * u2;

    #pragma unroll 
    for(uint32_t i=0; i<9; ++i){
        jacobian[i] = _res[i];
    }
}

void jacobian(torch::Tensor pos_camera_space, torch::Tensor jacobian){
    // pos_camera_space B x 3
    // jacobian B x 3 x 3
    uint32_t B = pos_camera_space.size(0);
    uint32_t gridsize = DIV_ROUND_UP(B, 1024);
    jacobian_kernel<<<gridsize, 1024>>>(pos_camera_space.data_ptr<float>(), jacobian.data_ptr<float>(), B);
}

__global__ void world2camera_kernel(const float * pos, const float * rot, const float * trans, float * res, uint32_t B){
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= B) return;
    pos += tid * 3;
    float _rot[9];
    float _trans[3];
    #pragma unroll
    for(int i=0;i<9;++i){
        _rot[i] = rot[i];
    }

    #pragma unroll
    for(int i=0;i<3;++i){
        _trans[i] = trans[i];
    }

    res += tid * 3;
    res[0] = pos[0] * _rot[0] + pos[1] * _rot[3] + pos[2] * _rot[6] + _trans[0];
    res[1] = pos[0] * _rot[1] + pos[1] * _rot[4] + pos[2] * _rot[7] + _trans[1];
    res[2] = pos[0] * _rot[2] + pos[1] * _rot[5] + pos[2] * _rot[8] + _trans[2];
}

void world2camera(torch::Tensor pos, torch::Tensor rot, torch::Tensor trans, torch::Tensor res){
    uint32_t B = pos.size(0);
    uint32_t gridsize = DIV_ROUND_UP(B, 1024);
    world2camera_kernel<<<gridsize, 1024>>>(pos.data_ptr<float>(), rot.data_ptr<float>(), trans.data_ptr<float>(), res.data_ptr<float>(), B);

}

__global__ void calc_tile_info_kernel(
    float * gaussian_pos,
    float * top,
    float * bottom, 
    float * left, 
    float * right,
    int * tile_n_point,
    int * tile_gaussian_list,
    uint32_t n_point,
    uint32_t n_tiles,
    uint32_t max_points_per_tile,
    float thresh_dis
){
    uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if(pid >= n_point) return;
    uint32_t tid = blockDim.y * blockIdx.y + threadIdx.y;
    if(tid >= n_tiles) return;
    gaussian_pos += pid * 3;
    top += tid;
    bottom += tid;
    left += tid;
    right += tid;
    // simple test
    float center_y = (*top + *bottom) / 2;
    float center_x = (*left + *right) / 2;
    float d1 = gaussian_pos[0] - center_x;
    float d2 = gaussian_pos[1] - center_y;
    if(d1*d1 + d2*d2 < thresh_dis){
        // write pid -> append
        uint32_t old = atomicAdd(tile_n_point + tid, 1);
        if(old<max_points_per_tile){
            tile_gaussian_list += max_points_per_tile * tid;
            tile_gaussian_list[old] = pid;
        }
    }
}

__global__ void calc_tile_info_kernel2(
    float * gaussian_pos,
    float * gaussian_cov,
    float * top,
    float * bottom, 
    float * left, 
    float * right,
    int * tile_n_point,
    int * tile_gaussian_list,
    uint32_t n_point,
    uint32_t n_tiles,
    uint32_t max_points_per_tile,
    float thresh_dis
){
    //
    uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if(pid >= n_point) return;
    uint32_t tid = blockDim.y * blockIdx.y + threadIdx.y;
    if(tid >= n_tiles) return;
    gaussian_pos += pid * 3;
    top += tid;
    bottom += tid;
    left += tid;
    right += tid;
    // simple test
    float _a, _b, _c, _d;
    _a = gaussian_cov[pid*4+0];
    _b = gaussian_cov[pid*4+1];
    _c = gaussian_cov[pid*4+2];
    _d = gaussian_cov[pid*4+3];

    float _center_x = gaussian_pos[0];
    float _center_y = gaussian_pos[1];

    float det = (_a * _d - _b * _c);
    if(det<=0) return;

    float _ai = _d / (det + 1e-14);
    float _bi = -_b / (det + 1e-14);
    float _ci = -_c / (det + 1e-14);
    float _di = _a / (det + 1e-14);
    float thresh_dis_log = -2 * logf(thresh_dis);
    float shift_x = sqrtf(_di * thresh_dis_log * det);
    float shift_y = sqrtf(_ai * thresh_dis_log * det);
    float bbx_right = _center_x + shift_x;
    float bbx_left  = _center_x - shift_x;
    float bbx_top = _center_y - shift_y;
    float bbx_bottom = _center_y + shift_y;

    if(! (*right < bbx_left || bbx_right < *left || *bottom < bbx_top || bbx_bottom < *top)){
        // uint32_t old = atomicAdd(tile_n_point + tid, 1);
        if(tile_n_point[tid]<max_points_per_tile){
            uint32_t old = atomicAdd(tile_n_point + tid, 1);
            tile_gaussian_list += max_points_per_tile * tid;
            tile_gaussian_list[old] = pid;
        }
    }
}

__global__ void calc_tile_info_kernel3(
    float * gaussian_pos,
    float * gaussian_cov,
    float tile_length_x,
    float tile_length_y,
    int * tile_n_point,
    int * tile_gaussian_list,
    uint32_t n_point,
    uint32_t n_tiles_x,
    uint32_t n_tiles_y,
    uint32_t max_points_per_tile,
    float thresh_dis,
    float leftmost,
    float topmost
){
    //
    uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if(pid >= n_point) return;
    gaussian_pos += pid * 3;
    // simple test
    float _a, _b, _c, _d;
    _a = gaussian_cov[pid*4+0];
    _b = gaussian_cov[pid*4+1];
    _c = gaussian_cov[pid*4+2];
    _d = gaussian_cov[pid*4+3];

    float _center_x = gaussian_pos[0];
    float _center_y = gaussian_pos[1];

    float det = (_a * _d - _b * _c);
    if(det<=0) return;

    float _ai = _d / (det + 1e-14);
    float _bi = -_b / (det + 1e-14);
    float _ci = -_c / (det + 1e-14);
    float _di = _a / (det + 1e-14);
    float thresh_dis_log = -2 * logf(thresh_dis);
    float shift_x = sqrtf(_di * thresh_dis_log * det);
    float shift_y = sqrtf(_ai * thresh_dis_log * det);
    float bbx_right = _center_x + shift_x;
    float bbx_left  = _center_x - shift_x;
    float bbx_top = _center_y - shift_y;
    float bbx_bottom = _center_y + shift_y;

    for(uint32_t i_top=fmaxf((bbx_top-topmost)/tile_length_y, 0); i_top<(uint32_t)((bbx_bottom-topmost)/tile_length_y+1) && i_top<n_tiles_y; ++i_top){
        for(uint32_t i_left=fmaxf((bbx_left-leftmost)/tile_length_x, 0); i_left<(uint32_t)((bbx_right-leftmost)/tile_length_x+1) && i_left<n_tiles_x; ++i_left){
            uint32_t tid = i_left + i_top * n_tiles_x; 
            if(tile_n_point[tid]<max_points_per_tile){
                uint32_t old = atomicAdd(tile_n_point + tid, 1);
                tile_gaussian_list[max_points_per_tile * tid + old] = pid;
            }
        }
    }
}



void calc_tile_list(
    Gaussian3ds & gaussians_image_space,
    Tiles & tile_info,
    torch::Tensor tile_n_point,
    torch::Tensor tile_gaussian_list,
    float thresh,
    int method, //0,1,2
    float tile_length_x,
    float tile_length_y,
    int n_tiles_x,
    int n_tiles_y,
    float leftmost,
    float topmost
)
{
    // create intersect info
    uint32_t n_point = gaussians_image_space.len();
    uint32_t n_tiles = tile_info.len();
    // printf("The Point Number: %d\n", n_point);
    // printf("The Tile Number: %d\n", n_tiles);
    uint32_t max_points_per_tile = tile_gaussian_list.size(1);

    if(method == 0){
        uint32_t gridsize_x = DIV_ROUND_UP(n_point, 32);
        uint32_t gridsize_y = DIV_ROUND_UP(n_tiles, 32);
        dim3 gridsize(gridsize_x, gridsize_y, 1);
        dim3 blocksize(32, 32, 1);
        calc_tile_info_kernel<<<gridsize, blocksize>>>(
            gaussians_image_space.pos.data_ptr<float>(), 
            tile_info.top.data_ptr<float>(), 
            tile_info.bottom.data_ptr<float>(), 
            tile_info.left.data_ptr<float>(), 
            tile_info.right.data_ptr<float>(), 
            tile_n_point.data_ptr<int>(),
            tile_gaussian_list.data_ptr<int>(),
            n_point, 
            n_tiles,
            max_points_per_tile,
            thresh
        );
    } 
    else if(method == 1){
        uint32_t gridsize_x = DIV_ROUND_UP(n_point, 32);
        uint32_t gridsize_y = DIV_ROUND_UP(n_tiles, 32);
        dim3 gridsize(gridsize_x, gridsize_y, 1);
        dim3 blocksize(32, 32, 1);
        calc_tile_info_kernel2<<<gridsize, blocksize>>>(
            gaussians_image_space.pos.data_ptr<float>(), 
            gaussians_image_space.cov.data_ptr<float>(), 
            tile_info.top.data_ptr<float>(), 
            tile_info.bottom.data_ptr<float>(), 
            tile_info.left.data_ptr<float>(), 
            tile_info.right.data_ptr<float>(), 
            tile_n_point.data_ptr<int>(),
            tile_gaussian_list.data_ptr<int>(),
            n_point, 
            n_tiles,
            max_points_per_tile,
            thresh
        );
    }
    else{
        uint32_t gridsize_x = DIV_ROUND_UP(n_point, 1024);
        dim3 gridsize(gridsize_x, 1, 1);
        dim3 blocksize(1024, 1, 1);
        calc_tile_info_kernel3<<<gridsize, blocksize>>>(
            gaussians_image_space.pos.data_ptr<float>(), 
            gaussians_image_space.cov.data_ptr<float>(), 
            tile_length_x,
            tile_length_y,
            tile_n_point.data_ptr<int>(),
            tile_gaussian_list.data_ptr<int>(),
            n_point, 
            n_tiles_x,
            n_tiles_y,
            max_points_per_tile,
            thresh,
            leftmost,
            topmost
        );
    }
}

__global__ void gather_gaussians_kernel(
    const int * tile_n_point_accum,
    const int * gaussian_list,
    int * gather_list,
    int * tile_ids_for_points,
    int n_tiles,
    int max_points_for_tile,
    int gaussian_list_size
){
    uint32_t tid = blockDim.y * blockIdx.y + threadIdx.y;
    if(tid >= n_tiles) return;
    uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t shift_res = tile_n_point_accum[tid];

    uint32_t n_point_this_tile = tile_n_point_accum[tid+1] - shift_res;
    if(pid >= n_point_this_tile) return;

    gaussian_list += tid * gaussian_list_size;
    gather_list[shift_res+pid] = gaussian_list[pid];
    tile_ids_for_points[shift_res+pid] = tid;
}

void gather_gaussians(
    torch::Tensor tile_n_point_accum, 
    torch::Tensor tile_gaussian_list, 
    torch::Tensor gathered_list, 
    torch::Tensor tile_ids_for_points, 
    int max_points_for_tile
){
    uint32_t n_tiles = tile_n_point_accum.size(0) - 1;
    uint32_t gridsize_x = DIV_ROUND_UP(max_points_for_tile, 32);
    uint32_t gridsize_y = DIV_ROUND_UP(n_tiles, 32);
    dim3 gridsize(gridsize_x, gridsize_y, 1);
    dim3 blocksize(32, 32, 1);
    uint32_t gaussian_list_size = tile_gaussian_list.size(1);
    gather_gaussians_kernel<<<gridsize, blocksize>>>(
        tile_n_point_accum.data_ptr<int>(),
        tile_gaussian_list.data_ptr<int>(),
        gathered_list.data_ptr<int>(),
        tile_ids_for_points.data_ptr<int>(),
        n_tiles,
        max_points_for_tile,
        gaussian_list_size
    );

}

template<uint32_t SMSIZE>
__global__ void draw_backward_kernel(
    const float * gaussian_pos,
    const float * gaussian_rgb,
    const float * gaussian_opa,
    const float * gaussian_cov,
    const int * tile_n_point_accum, 
    const float * output,
    const float * grad_output,
    float * grad_pos, 
    float * grad_rgb,
    float * grad_opa,
    float * grad_cov,
    const float focal_x, 
    const float focal_y,
    const uint32_t w,
    const uint32_t h,
    const bool weight_normalize,
    const bool sigmoid
){
    uint32_t id_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t id_y = blockDim.y * blockIdx.y + threadIdx.y;
    if(id_x>=w || id_y>=h) return;
    uint32_t id_tile = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t start_idx = tile_n_point_accum[id_tile];
    uint32_t end_idx = tile_n_point_accum[id_tile+1];
    // const uint32_t interval_length = end_idx - start_idx;
    __shared__ float _gaussian_pos[SMSIZE*2];
    __shared__ float _gaussian_rgb[SMSIZE*3];
    __shared__ float _gaussian_opa[SMSIZE*1];
    __shared__ float _gaussian_cov[SMSIZE*4];
    __shared__ float _grad_pos[SMSIZE*2];
    __shared__ float _grad_rgb[SMSIZE*3];
    __shared__ float _grad_opa[SMSIZE*1];
    __shared__ float _grad_cov[SMSIZE*4];
    uint32_t id_thread = threadIdx.x + threadIdx.y * blockDim.x;
    uint32_t blocksize = blockDim.x * blockDim.y;

    // initialize shared memory for gradients
    for(uint32_t i=id_thread; i<SMSIZE; i+=blocksize){
        #pragma unroll
        for(uint32_t _m=0; _m<2; ++_m){
            _grad_pos[i*2+_m] = 0;
        }
        #pragma unroll
        for(uint32_t _m=0; _m<3; ++_m){
            _grad_rgb[i*3+_m] = 0;
        }
        _grad_opa[i] = 0;
        #pragma unroll
        for(uint32_t _m=0; _m<4; ++_m){
            _grad_cov[i*4+_m] = 0;
        }
    }

    //draw: access all point with early stop
    float pixel_x = (id_x + 0.5 - w/2)/focal_x;
    float pixel_y = (id_y + 0.5 - h/2)/focal_y;
    float color[] = {0, 0, 0};
    double current_prob = 0.0;
    double current_prob_c0, current_prob0, current_prob1;
    float accum = 1.0;
    float accum_weight = 0.0;
    double _a, _b, _c, _d, _x, _y, det;
    float alpha, weight;

    // load to memory
    uint32_t n_loadings = DIV_ROUND_UP(end_idx - start_idx, SMSIZE);
    uint32_t global_idx;
    
    //output gradient
    output += (id_x + id_y * w) * 3;
    grad_output += (id_x + id_y * w) * 3;
    float cur_grad_out[3];
    float cur_out[3];
    #pragma unroll
    for(int _m=0; _m<3; ++_m){
        cur_grad_out[_m] = grad_output[_m];
        cur_out[_m] = output[_m];
    }

    for(uint32_t i_loadings=0; i_loadings<n_loadings; ++i_loadings){
        for(uint32_t i=id_thread; i<SMSIZE; i+=blocksize){
            // position
            global_idx = i_loadings*SMSIZE + i;
            if(global_idx>=(end_idx - start_idx)){
                break;
            }
            _gaussian_pos[i*2 + 0] = gaussian_pos[(start_idx + i_loadings*SMSIZE + i)*3 + 0];
            _gaussian_pos[i*2 + 1] = gaussian_pos[(start_idx + i_loadings*SMSIZE + i)*3 + 1];
            // rgb
            _gaussian_rgb[i*3 + 0] = gaussian_rgb[(start_idx + i_loadings*SMSIZE + i)*3 + 0];
            _gaussian_rgb[i*3 + 1] = gaussian_rgb[(start_idx + i_loadings*SMSIZE + i)*3 + 1];
            _gaussian_rgb[i*3 + 2] = gaussian_rgb[(start_idx + i_loadings*SMSIZE + i)*3 + 2];
            // opa
            _gaussian_opa[i] = gaussian_opa[start_idx + i_loadings*SMSIZE + i];
            // cov
            _gaussian_cov[i*4 + 0] = gaussian_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 0];
            _gaussian_cov[i*4 + 1] = gaussian_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 1];
            _gaussian_cov[i*4 + 2] = gaussian_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 2];
            _gaussian_cov[i*4 + 3] = gaussian_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 3];
        }
        __syncthreads();
        
        for(uint32_t i=0; i<SMSIZE; ++i){
            global_idx = i_loadings*SMSIZE + i;

            // check bound and early stop
            if(global_idx>=(end_idx-start_idx)||accum < 0.01){
                break;
            }
            _a = _gaussian_cov[i*4+0];
            _b = _gaussian_cov[i*4+1];
            _c = _gaussian_cov[i*4+2];
            _d = _gaussian_cov[i*4+3];
            _x = pixel_x - _gaussian_pos[i*2 + 0];
            _y = pixel_y - _gaussian_pos[i*2 + 1];
            det = (_a * _d - _b * _c);

            double Pm = -(_d * _x * _x - (_b + _c) * _x * _y + _a * _y * _y);
            double Pn = (2 * det + 1e-14);

            current_prob_c0 = sigmoid ? 1.0/2*3.1415926536 : 1.0;
            current_prob0 = sigmoid ? current_prob_c0 * rsqrtf(det+1e-7) : 1.0;
            current_prob1 = exp(Pm / Pn);
            current_prob = current_prob0 * current_prob1;
            
            // gradient primitives for 2d gaussian
            double dPm_da = - (_y * _y);
            double dPm_db = _x * _y;
            double dPm_dc = _x * _y;
            double dPm_dd = - (_x * _x);
            double dPn_da = 2 * _d;
            double dPn_db = -2 * _c;
            double dPn_dc = -2 * _b;
            double dPn_dd = 2 * _a;
            double dP1_da = current_prob1 * (dPm_da*Pn - dPn_da*Pm) / (Pn*Pn);
            double dP1_db = current_prob1 * (dPm_db*Pn - dPn_db*Pm) / (Pn*Pn);
            double dP1_dc = current_prob1 * (dPm_dc*Pn - dPn_dc*Pm) / (Pn*Pn);
            double dP1_dd = current_prob1 * (dPm_dd*Pn - dPn_dd*Pm) / (Pn*Pn);

            double dP0_da = sigmoid ? -0.5 * (current_prob0 * current_prob0 * current_prob0) / (current_prob_c0 * current_prob_c0) * _d : 0.0;
            double dP0_db = sigmoid ? 0.5 * (current_prob0 * current_prob0 * current_prob0) / (current_prob_c0 * current_prob_c0) * _c : 0.0;
            double dP0_dc = sigmoid ? 0.5 * (current_prob0 * current_prob0 * current_prob0) / (current_prob_c0 * current_prob_c0) * _b : 0.0;
            double dP0_dd = sigmoid ? -0.5 * (current_prob0 * current_prob0 * current_prob0) / (current_prob_c0 * current_prob_c0) * _a : 0.0;

            double dP_da = current_prob0 * dP1_da + current_prob1 * dP0_da;
            double dP_db = current_prob0 * dP1_db + current_prob1 * dP0_db;
            double dP_dc = current_prob0 * dP1_dc + current_prob1 * dP0_dc;
            double dP_dd = current_prob0 * dP1_dd + current_prob1 * dP0_dd;
            // gradient w.r.t position (not _x, _y, with a minus)
            double dP_dx = current_prob / Pn * (2*_d*_x - _b*_y - _c*_y);
            double dP_dy = current_prob / Pn * (2*_a*_y - _b*_x - _c*_x);

            alpha = current_prob * _gaussian_opa[i];
            // sigmoid + scale -> 0-1
            if(sigmoid){
                alpha = 2./(exp(-alpha)+1) - 1;
            }
            weight = alpha * accum;
            float cur_point_color[3];
            cur_point_color[0] = _gaussian_rgb[i*3 + 0];
            cur_point_color[1] = _gaussian_rgb[i*3 + 1];
            cur_point_color[2] = _gaussian_rgb[i*3 + 2];
            color[0] += cur_point_color[0] * weight;
            color[1] += cur_point_color[1] * weight;
            color[2] += cur_point_color[2] * weight;
            accum_weight += weight;

            // grad w.r.t gaussian rgb
            atomicAdd(_grad_rgb+i*3+0, cur_grad_out[0]*weight);
            atomicAdd(_grad_rgb+i*3+1, cur_grad_out[1]*weight);
            atomicAdd(_grad_rgb+i*3+2, cur_grad_out[2]*weight);

            // grad w.r.t pos opa cov -> grad w.r.t alpha
            float d_alpha = 0;
            #pragma unroll
            for(int _m=0; _m<3; ++_m){
                d_alpha += cur_grad_out[_m] * cur_point_color[_m];
            }
            d_alpha *= accum;
            float _d_alpha_acc = 0;
            #pragma unroll
            for(int _m=0; _m<3; ++_m){
                _d_alpha_acc += cur_grad_out[_m] * (cur_out[_m]-color[_m]);
            }
            _d_alpha_acc /= (1-alpha+1e-7);
            d_alpha -= _d_alpha_acc;
            // backward to activation function
            if(sigmoid){
                d_alpha = d_alpha * (alpha + 1 - 0.5*(alpha+1)*(alpha+1));
            }
            // grad w.r.t opa
            atomicAdd(_grad_opa+i*1+0, (float)(d_alpha*current_prob));
            float d_current_prob = d_alpha * _gaussian_opa[i];
            // grad w.r.t pos
            atomicAdd(_grad_pos+i*2+0, (float)(d_current_prob * dP_dx));
            atomicAdd(_grad_pos+i*2+1, (float)(d_current_prob * dP_dy));
            // grad w.r.t cov
            atomicAdd(_grad_cov+i*4+0, (float)(d_current_prob * dP_da));
            atomicAdd(_grad_cov+i*4+1, (float)(d_current_prob * dP_db));
            atomicAdd(_grad_cov+i*4+2, (float)(d_current_prob * dP_dc));
            atomicAdd(_grad_cov+i*4+3, (float)(d_current_prob * dP_dd));

            accum *= (1-alpha);
        }
        __syncthreads();

        // write gradients back to global memory
        for(uint32_t i=id_thread; i<SMSIZE; i+=blocksize){
            // position
            global_idx = i_loadings*SMSIZE + i;
            if(global_idx>=(end_idx - start_idx)){
                break;
            }
            grad_pos[(start_idx + i_loadings*SMSIZE + i)*3 + 0] = _grad_pos[i*2 + 0];
            grad_pos[(start_idx + i_loadings*SMSIZE + i)*3 + 1] = _grad_pos[i*2 + 1];

            grad_rgb[(start_idx + i_loadings*SMSIZE + i)*3 + 0] = _grad_rgb[i*3 + 0];
            grad_rgb[(start_idx + i_loadings*SMSIZE + i)*3 + 1] = _grad_rgb[i*3 + 1];
            grad_rgb[(start_idx + i_loadings*SMSIZE + i)*3 + 2] = _grad_rgb[i*3 + 2];

            grad_opa[(start_idx + i_loadings*SMSIZE + i)*1 + 0] = _grad_opa[i];

            grad_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 0] = _grad_cov[i*4 + 0];
            grad_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 1] = _grad_cov[i*4 + 1];
            grad_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 2] = _grad_cov[i*4 + 2];
            grad_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 3] = _grad_cov[i*4 + 3];
        }
        // if this sync is necessary ?
        // __syncthreads();
    }
}


template<uint32_t SMSIZE>
__global__ void draw_kernel(
    // Gaussian3ds & tile_sorted_gaussians, 
    const float * gaussian_pos,
    const float * gaussian_rgb,
    const float * gaussian_opa,
    const float * gaussian_cov,
    const int * tile_n_point_accum, 
    float * res, 
    const float focal_x, 
    const float focal_y,
    const uint32_t w,
    const uint32_t h,
    const bool weight_normalize,
    const bool sigmoid
){
    uint32_t id_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t id_y = blockDim.y * blockIdx.y + threadIdx.y;
    if(id_x>=w || id_y>=h) return;
    uint32_t id_tile = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t start_idx = tile_n_point_accum[id_tile];
    uint32_t end_idx = tile_n_point_accum[id_tile+1];
    // const uint32_t interval_length = end_idx - start_idx;
    __shared__ float _gaussian_pos[SMSIZE*2];
    __shared__ float _gaussian_rgb[SMSIZE*3];
    __shared__ float _gaussian_opa[SMSIZE*1];
    __shared__ float _gaussian_cov[SMSIZE*4];
    uint32_t id_thread = threadIdx.x + threadIdx.y * blockDim.x;
    uint32_t blocksize = blockDim.x * blockDim.y;
    //draw: access all point with early stop
    float pixel_x = (id_x + 0.5 - w/2)/focal_x;
    float pixel_y = (id_y + 0.5 - h/2)/focal_y;
    float color[] = {0, 0, 0};
    double current_prob = 0.0;
    float accum = 1.0;
    float accum_weight = 0.0;
    double _a, _b, _c, _d, _x, _y, det;
    float alpha, weight;

    // load to memory
    uint32_t n_loadings = DIV_ROUND_UP(end_idx - start_idx, SMSIZE);
    uint32_t global_idx;
    for(uint32_t i_loadings=0; i_loadings<n_loadings; ++i_loadings){
        for(uint32_t i=id_thread; i<SMSIZE; i+=blocksize){
            // position
            global_idx = i_loadings*SMSIZE + i;
            if(global_idx>=(end_idx - start_idx)){
                break;
            }
            _gaussian_pos[i*2 + 0] = gaussian_pos[(start_idx + i_loadings*SMSIZE + i)*3 + 0];
            _gaussian_pos[i*2 + 1] = gaussian_pos[(start_idx + i_loadings*SMSIZE + i)*3 + 1];
            // rgb
            _gaussian_rgb[i*3 + 0] = gaussian_rgb[(start_idx + i_loadings*SMSIZE + i)*3 + 0];
            _gaussian_rgb[i*3 + 1] = gaussian_rgb[(start_idx + i_loadings*SMSIZE + i)*3 + 1];
            _gaussian_rgb[i*3 + 2] = gaussian_rgb[(start_idx + i_loadings*SMSIZE + i)*3 + 2];
            // opa
            _gaussian_opa[i] = gaussian_opa[start_idx + i_loadings*SMSIZE + i];
            // cov
            _gaussian_cov[i*4 + 0] = gaussian_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 0];
            _gaussian_cov[i*4 + 1] = gaussian_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 1];
            _gaussian_cov[i*4 + 2] = gaussian_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 2];
            _gaussian_cov[i*4 + 3] = gaussian_cov[(start_idx + i_loadings*SMSIZE + i)*4 + 3];
        }
        __syncthreads();
        
        for(uint32_t i=0; i<SMSIZE; ++i){
            global_idx = i_loadings*SMSIZE + i;

            // check bound and early stop
            if(global_idx>=(end_idx-start_idx)||accum < 0.01){
                break;
            }
            _a = _gaussian_cov[i*4+0];
            _b = _gaussian_cov[i*4+1];
            _c = _gaussian_cov[i*4+2];
            _d = _gaussian_cov[i*4+3];
            _x = pixel_x - _gaussian_pos[i*2 + 0];
            _y = pixel_y - _gaussian_pos[i*2 + 1];
            det = (_a * _d - _b * _c);

            current_prob = sigmoid ? 1.0/2*3.1415926536 * rsqrtf(det+1e-7) : 1;
            //current_prob *= exp(-(_d * _x * _x - (_b + _c) * _x * _y + _a * _y * _y) / (2 * det + 1e-7));
            current_prob *= exp(-(_d * _x * _x - (_b + _c) * _x * _y + _a * _y * _y) / (2 * det+1e-14));

            alpha = current_prob * _gaussian_opa[i];
            // printf("current_alpha: %f\n", alpha);
            // sigmoid + scale -> 0-1
            if(sigmoid){
                alpha = 2./(exp(-alpha)+1) - 1;
            }
            weight = alpha * accum;
            // printf("a: %f, b: %f, c: %f, d: %f, x: %f, y: %f, det: %f, current_prob: %f, weight: %f\n", _a, _b, _c, _d, _x, _y, det, current_prob, weight);
            color[0] += _gaussian_rgb[i*3 + 0] * weight;
            color[1] += _gaussian_rgb[i*3 + 1] * weight;
            color[2] += _gaussian_rgb[i*3 + 2] * weight;
            accum_weight += weight;
            accum *= (1-alpha);
        }
    }

    if(accum_weight < 0.01 || !weight_normalize){
        accum_weight = 1;
    }
    res[(id_x + id_y * w)*3 + 0] = color[0] / accum_weight;
    res[(id_x + id_y * w)*3 + 1] = color[1] / accum_weight;
    res[(id_x + id_y * w)*3 + 2] = color[2] / accum_weight;
}

// void draw(Gaussian3ds & tile_sorted_gaussians, torch::Tensor tile_n_point_accum, torch::Tensor res, float focal_x, float focal_y){
void draw(
    torch::Tensor gaussian_pos, 
    torch::Tensor gaussian_rgb, 
    torch::Tensor gaussian_opa, 
    torch::Tensor gaussian_cov, 
    torch::Tensor tile_n_point_accum, 
    torch::Tensor res, 
    float focal_x, 
    float focal_y,
    bool weight_normalize,
    bool sigmoid
){
    uint32_t h = res.size(0);
    uint32_t w = res.size(1); 
    uint32_t gridsize_x = DIV_ROUND_UP(w, 16);
    uint32_t gridsize_y = DIV_ROUND_UP(h, 16);
    dim3 gridsize(gridsize_x, gridsize_y, 1);
    dim3 blocksize(16, 16, 1);
    draw_kernel<1200><<<gridsize, blocksize>>>(
        gaussian_pos.data_ptr<float>(),
        gaussian_rgb.data_ptr<float>(),
        gaussian_opa.data_ptr<float>(),
        gaussian_cov.data_ptr<float>(),
        tile_n_point_accum.data_ptr<int>(),
        res.data_ptr<float>(),
        focal_x,
        focal_y,
        w,
        h,
        weight_normalize,
        sigmoid
    );
}

void draw_backward(
    torch::Tensor gaussian_pos, 
    torch::Tensor gaussian_rgb, 
    torch::Tensor gaussian_opa, 
    torch::Tensor gaussian_cov, 
    torch::Tensor tile_n_point_accum, 
    torch::Tensor output, 
    torch::Tensor grad_output,
    torch::Tensor grad_pos,
    torch::Tensor grad_rgb,
    torch::Tensor grad_opa,
    torch::Tensor grad_cov,
    float focal_x, 
    float focal_y,
    bool weight_normalize,
    bool sigmoid
){
    uint32_t h = output.size(0);
    uint32_t w = output.size(1); 
    uint32_t gridsize_x = DIV_ROUND_UP(w, 16);
    uint32_t gridsize_y = DIV_ROUND_UP(h, 16);
    dim3 gridsize(gridsize_x, gridsize_y, 1);
    dim3 blocksize(16, 16, 1);
    draw_backward_kernel<512><<<gridsize, blocksize>>>(
        gaussian_pos.data_ptr<float>(),
        gaussian_rgb.data_ptr<float>(),
        gaussian_opa.data_ptr<float>(),
        gaussian_cov.data_ptr<float>(),
        tile_n_point_accum.data_ptr<int>(),
        output.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_pos.data_ptr<float>(),
        grad_rgb.data_ptr<float>(),
        grad_opa.data_ptr<float>(),
        grad_cov.data_ptr<float>(),
        focal_x,
        focal_y,
        w,
        h,
        weight_normalize,
        sigmoid
    );
}

__device__ void world_to_camera(
    const float* pos_w,
    const float* current_rot,
    const float* current_tran,
    float* pos_c
){
    // pos_w 3 current_rot 3*3
    float _rot[9];
    float _trans[3];
    #pragma unroll
    for(int i=0;i<9;++i){
        _rot[i] = current_rot[i];
    }

    #pragma unroll
    for(int i=0;i<3;++i){
        _trans[i] = current_tran[i];
    }

    #pragma unroll
    for(int i=0;i<3;++i){
        pos_c[i] = _rot[i*3+0] * pos_w[0] + _rot[i*3+1] * pos_w[1] + _rot[i*3+2] * pos_w[2] + _trans[i];
    }
}

__device__ void calc_jacobian(
    const float* pos_camera_space,
    float* jacobian
){
    float _res[9];
    float u0 = pos_camera_space[0];
    float u1 = pos_camera_space[1];
    float u2 = pos_camera_space[2];

    _res[0] = 1/u2;
    _res[1] = 0;
    _res[2] = -u0/(u2*u2);
    _res[3] = 0;
    _res[4] = 1/u2;
    _res[5] = -u1/(u2*u2);
    float _rsqr = rsqrtf(u0*u0+u1*u1+u2*u2);
    _res[6] = _rsqr * u0;
    _res[7] = _rsqr * u1;
    _res[8] = _rsqr * u2;

    #pragma unroll 
    for(uint32_t i=0; i<9; ++i){
        jacobian[i] = _res[i];
    }
}

__global__ void global_culling_kernel(
    const float* pos,
    const float* quat,
    const float* scale,
    const float* current_rot,
    const float* current_tran,
    const uint32_t n_point,
    const float near,
    const float half_width,
    const float half_height,
    float* res_pos,
    float* res_cov,
    long* culling_mask
    // int* res_size
){
    uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if(pid >= n_point) return;
    pos += pid*3;

    // 1. calculate the camera space coordinate
    float pos_c[3];
    world_to_camera(pos, current_rot, current_tran, pos_c);

    // printf("z: %f\n", pos_c[2]);

    // 2. check if the point is before the near plane
    if(pos_c[2] < near){
        // culling_mask[pid] = 0;
        return;
    }

    // 3. image space transform
    float pos_i[3];
    pos_i[0] = pos_c[0] / pos_c[2];
    pos_i[1] = pos_c[1] / pos_c[2];
    pos_i[2] = sqrtf(pos_c[0]*pos_c[0] + pos_c[1]*pos_c[1] + pos_c[2]*pos_c[2]);

    // 4. frustum culling
    if(abs(pos_i[0]) > half_width || abs(pos_i[1]) > half_height){
        // culling_mask[pid] = 0;
        return;
    }
    culling_mask[pid] = 1;
    res_pos[pid*3 + 0] = pos_i[0];
    res_pos[pid*3 + 1] = pos_i[1];
    res_pos[pid*3 + 2] = pos_i[2];


    // 5. calculate the covariance matrix and jacobian
    float w = quat[4*pid+0];
    float x = quat[4*pid+1];
    float y = quat[4*pid+2];
    float z = quat[4*pid+3];

    float R[9];
    R[0] = 1 - 2*y*y - 2*z*z;
    R[1] = 2*x*y - 2*z*w;
    R[2] = 2*x*z + 2*y*w;
    R[3] = 2*x*y + 2*z*w;
    R[4] = 1 - 2*x*x - 2*z*z;
    R[5] = 2*y*z - 2*x*w;
    R[6] = 2*x*z - 2*y*w;
    R[7] = 2*y*z + 2*x*w;
    R[8] = 1 - 2*x*x - 2*y*y;

    float S[9];
    S[0] = scale[pid*3+0];
    S[1] = 0;
    S[2] = 0;
    S[3] = 0;
    S[4] = scale[pid*3+1];
    S[5] = 0;
    S[6] = 0;
    S[7] = 0;
    S[8] = scale[pid*3+2];

    // RS
    float RS[9];
    #pragma unroll
    for(uint32_t i_r=0; i_r<3; ++i_r){
        #pragma unroll
        for(uint32_t i_c=0; i_c<3; ++i_c){
            RS[i_r*3+i_c] = 0;
            #pragma unroll
            for(uint32_t i_k=0; i_k<3; ++i_k){
                RS[i_r*3+i_c] += R[i_r*3+i_k] * S[i_k*3+i_c];
            }
        }
    }
    float RSSR[9];
    #pragma unroll
    for(uint32_t i_r=0; i_r<3; ++i_r){
        #pragma unroll
        for(uint32_t i_c=0; i_c<3; ++i_c){
            RSSR[i_r*3+i_c] = 0;
            #pragma unroll
            for(uint32_t i_k=0; i_k<3; ++i_k){
                RSSR[i_r*3+i_c] += RS[i_r*3+i_k] * RS[i_c*3+i_k];
            }
        }
    }
    
    float jacobian[9];
    calc_jacobian(pos_c, jacobian);
    // jacobian is required to multiplied by rotation matrix, in the form of jwRSSRw'j'

    float JW[9];
    #pragma unroll
    for(int i_r=0; i_r<3; ++i_r){
        #pragma unroll
        for(int i_c=0; i_c<3; ++i_c){
            JW[i_r*3+i_c] = 0;
            #pragma unroll
            for(int i_k=0; i_k<3; ++i_k){
                JW[i_r*3+i_c] += jacobian[i_r*3+i_k] * current_rot[i_k*3+i_c];
            }
        }
    }
    float JWC[9];
    #pragma unroll
    for(int i_r=0; i_r<3; ++i_r){
        #pragma unroll
        for(int i_c=0; i_c<3; ++i_c){
            JWC[i_r*3+i_c] = 0;
            #pragma unroll
            for(int i_k=0; i_k<3; ++i_k){
                JWC[i_r*3+i_c] += JW[i_r*3+i_k] * RSSR[i_k*3+i_c];
            }
        }
    }

    float JWCWJ[9];
    #pragma unroll
    for(int i_r=0; i_r<3; ++i_r){
        #pragma unroll
        for(int i_c=0; i_c<3; ++i_c){
            JWCWJ[i_r*3+i_c] = 0;
            #pragma unroll
            for(int i_k=0; i_k<3; ++i_k){
                JWCWJ[i_r*3+i_c] += JWC[i_r*3+i_k] * JW[i_c*3+i_k];
            }
        }
    }

    // write back to the covariance matrix to the res_cov variables.
    res_cov[pid*4+0] = JWCWJ[0];
    res_cov[pid*4+1] = JWCWJ[1];
    res_cov[pid*4+2] = JWCWJ[3];
    res_cov[pid*4+3] = JWCWJ[4];
}

void global_culling(
    torch::Tensor pos, 
    torch::Tensor quat, 
    torch::Tensor scale, 
    torch::Tensor current_rot, 
    torch::Tensor current_tran, 
    torch::Tensor res_pos,
    torch::Tensor res_cov,
    torch::Tensor culling_mask,
    float near, 
    float half_width, 
    float half_height
){
    uint32_t n_point = pos.size(0);
    uint32_t gridsize_x = DIV_ROUND_UP(n_point, 1024);
    dim3 gridsize(gridsize_x, 1, 1);
    dim3 blocksize(1024, 1, 1);
    global_culling_kernel<<<gridsize, blocksize>>>(
        pos.data_ptr<float>(),
        quat.data_ptr<float>(),
        scale.data_ptr<float>(),
        current_rot.data_ptr<float>(),
        current_tran.data_ptr<float>(),
        n_point,
        near,
        half_width,
        half_height,
        res_pos.data_ptr<float>(),
        res_cov.data_ptr<float>(),
        culling_mask.data_ptr<long>()
    );
}

__global__ void global_culling_backward_kernel(
    const float* pos,
    const float* quat,
    const float* scale,
    const float* current_rot,
    const float* current_tran,
    const uint32_t n_point,
    const float* gradout_pos,
    const float* gradout_cov,
    const long* culling_mask,
    float* gradinput_pos,
    float* gradinput_quat,
    float* gradinput_scale
){
    uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if(pid >= n_point) return;
    pos += pid*3;

    if(culling_mask[pid]==0){
        return;
    }
    
    // forward pass pos 0,1,2 -> pos_c 0,1,2 -> pos_i 0,1,2
    float pos_c[3];
    world_to_camera(pos, current_rot, current_tran, pos_c);

    float grad_c[3];
    float pos_i_z = sqrtf(pos_c[0]*pos_c[0] + pos_c[1]*pos_c[1] + pos_c[2]*pos_c[2]);
    float grad_i[3];
    grad_i[0] = gradout_pos[pid*3+0];
    grad_i[1] = gradout_pos[pid*3+1];
    grad_i[2] = gradout_pos[pid*3+2];

    grad_c[0] = grad_i[0] / pos_c[2] + grad_i[2] * pos_c[0] / pos_i_z;
    grad_c[1] = grad_i[1] / pos_c[2] + grad_i[2] * pos_c[1] / pos_i_z;
    grad_c[2] = - grad_i[0] * pos_c[0] / (pos_c[2] * pos_c[2]) - grad_i[1] * pos_c[1] / (pos_c[2] * pos_c[2]) + grad_i[2] * pos_c[2] / pos_i_z;

    float grad_w[3];

    #pragma unroll
    for(int i_r=0; i_r<3; ++i_r){
        #pragma unroll
        grad_w[i_r] = 0;
        for(int i_k=0; i_k<3; ++i_k){
            grad_w[i_r] += current_rot[i_k*3+i_r] * grad_c[i_k];
        }
    }
    // write back the pos gradient
    gradinput_pos[pid*3+0] = grad_w[0];
    gradinput_pos[pid*3+1] = grad_w[1];
    gradinput_pos[pid*3+2] = grad_w[2];

    float jacobian[9];
    calc_jacobian(pos_c, jacobian);
    // jacobian is required to multiplied by rotation matrix, in the form of jwRSSRw'j'

    float JW[9];
    #pragma unroll
    for(int i_r=0; i_r<3; ++i_r){
        #pragma unroll
        for(int i_c=0; i_c<3; ++i_c){
            JW[i_r*3+i_c] = 0;
            #pragma unroll
            for(int i_k=0; i_k<3; ++i_k){
                JW[i_r*3+i_c] += jacobian[i_r*3+i_k] * current_rot[i_k*3+i_c];
            }
        }
    }
    // calc grad_3d_cov
    float grad_3d_cov[9];

    // move to register
    float grad_2d_cov[4];
    #pragma unroll
    for(uint32_t i=0;i<4;++i){
        grad_2d_cov[i] = gradout_cov[pid*4+i];
    }

    #pragma unroll
    for(uint32_t i_r=0; i_r<3; ++i_r){
        for(uint32_t i_c=0; i_c<3; ++i_c){
            grad_3d_cov[i_r*3+i_c] = 0;
            #pragma unroll
            for(uint32_t i_i=0; i_i<2; ++i_i){
                #pragma unroll
                for(uint32_t i_j=0; i_j<2; ++i_j){
                    grad_3d_cov[i_r*3+i_c] += grad_2d_cov[i_i*2+i_j] * JW[i_i*3+i_r] * JW[i_j*3+i_c];
                }
            }
        }
    }

    // 5. calculate the covariance matrix and jacobian
    float w = quat[4*pid+0];
    float x = quat[4*pid+1];
    float y = quat[4*pid+2];
    float z = quat[4*pid+3];

    float R[9];
    R[0] = 1 - 2*y*y - 2*z*z;
    R[1] = 2*x*y - 2*z*w;
    R[2] = 2*x*z + 2*y*w;
    R[3] = 2*x*y + 2*z*w;
    R[4] = 1 - 2*x*x - 2*z*z;
    R[5] = 2*y*z - 2*x*w;
    R[6] = 2*x*z - 2*y*w;
    R[7] = 2*y*z + 2*x*w;
    R[8] = 1 - 2*x*x - 2*y*y;

    float S[9];
    S[0] = scale[pid*3+0];
    S[1] = 0;
    S[2] = 0;
    S[3] = 0;
    S[4] = scale[pid*3+1];
    S[5] = 0;
    S[6] = 0;
    S[7] = 0;
    S[8] = scale[pid*3+2];

    // RS
    float RS[9];
    #pragma unroll
    for(uint32_t i_r=0; i_r<3; ++i_r){
        #pragma unroll
        for(uint32_t i_c=0; i_c<3; ++i_c){
            RS[i_r*3+i_c] = 0;
            #pragma unroll
            for(uint32_t i_k=0; i_k<3; ++i_k){
                RS[i_r*3+i_c] += R[i_r*3+i_k] * S[i_k*3+i_c];
            }
        }
    }

    // gradient w.r.t M=RS
    float grad_RS[9];
    #pragma unroll
    for(uint32_t i_r=0; i_r<3; ++i_r){
        #pragma unroll
        for(uint32_t i_c=0; i_c<3; ++i_c){
            grad_RS[i_r*3+i_c] = 0;
            #pragma unroll
            for(uint32_t i_k=0; i_k<3; ++i_k){
                grad_RS[i_r*3+i_c] += 2 * grad_3d_cov[i_r*3+i_k] * RS[i_c*3+i_k];
            }
        }
    }

    //gradient w.r.t scale
    float grad_scale[3];
    #pragma unroll
    for(uint32_t i=0; i<3; ++i){
        grad_scale[i] = grad_RS[0*3+i]*R[0*3+i] + grad_RS[1*3+i]*R[1*3+i] + grad_RS[2*3+i]*R[2*3+i];
    }

    float sx = S[0];
    float sy = S[4];
    float sz = S[8];
    float qr = w;
    float qi = x;
    float qj = y;
    float qk = z;
    float gradcoeff_qr[9] = {
        0, -2*sy*qk, 2*sz*qj,
        2*sx*qk, 0, -2*sz*qi,
        -2*sx*qj, 2*sy*qi, 0
    };
    float gradcoeff_qi[9] = {
        0, 2*sy*qj, 2*sz*qk,
        2*sx*qj, -4*sy*qi, -2*sz*qr,
        2*sx*qk, 2*sy*qr, -4*sz*qi
    };
    float gradcoeff_qj[9] ={
        -4*sx*qj, 2*sy*qi, 2*sz*qr,
        2*sx*qi, 0, 2*sz*qk,
        -2*sx*qr, 2*sy*qk, -4*sz*qj
    };
    float gradcoeff_qk[9] = {
        -4*sx*qk, -2*sy*qr, 2*sz*qi,
        2*sx*qr, -4*sy*qk, 2*sz*qj,
        2*sx*qi, 2*sy*qj, 0
    };
    float grad_qr = 0;
    float grad_qi = 0;
    float grad_qj = 0;
    float grad_qk = 0;

    #pragma unroll
    for(uint32_t i=0; i<9; ++i){
        grad_qr += gradcoeff_qr[i] * grad_RS[i]; 
        grad_qi += gradcoeff_qi[i] * grad_RS[i];
        grad_qj += gradcoeff_qj[i] * grad_RS[i];
        grad_qk += gradcoeff_qk[i] * grad_RS[i];
    }
    //write back to global memory
    gradinput_quat[pid*4+0] = grad_qr;
    gradinput_quat[pid*4+1] = grad_qi;
    gradinput_quat[pid*4+2] = grad_qj;
    gradinput_quat[pid*4+3] = grad_qk;
    
    gradinput_scale[pid*3+0] = grad_scale[0];
    gradinput_scale[pid*3+1] = grad_scale[1];
    gradinput_scale[pid*3+2] = grad_scale[2];
}

void global_culling_backward(
    torch::Tensor pos, 
    torch::Tensor quat, 
    torch::Tensor scale, 
    torch::Tensor current_rot, 
    torch::Tensor current_tran, 
    torch::Tensor gradout_pos, 
    torch::Tensor gradout_cov, 
    torch::Tensor culling_mask, 
    torch::Tensor gradinput_pos, 
    torch::Tensor gradinput_quat, 
    torch::Tensor gradinput_scale
){
    uint32_t n_point = pos.size(0);
    uint32_t gridsize_x = DIV_ROUND_UP(n_point, 1024);
    dim3 gridsize(gridsize_x, 1, 1);
    dim3 blocksize(1024, 1, 1);
    global_culling_backward_kernel<<<gridsize, blocksize>>>(
        pos.data_ptr<float>(),
        quat.data_ptr<float>(),
        scale.data_ptr<float>(),
        current_rot.data_ptr<float>(),
        current_tran.data_ptr<float>(),
        n_point,
        gradout_pos.data_ptr<float>(),
        gradout_cov.data_ptr<float>(),
        culling_mask.data_ptr<long>(),
        gradinput_pos.data_ptr<float>(),
        gradinput_quat.data_ptr<float>(),
        gradinput_scale.data_ptr<float>()
    );
}