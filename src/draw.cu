#include <stdio.h>
#include "include/common.hpp"
#include <torch/torch.h>
#include <cuda_runtime.h>


// Spherical functions from svox2
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

__device__ __inline__ void calc_sh(
    const int basis_dim,
    const float* __restrict__ dir,
    float* __restrict__ out) {
    out[0] = C0;
    const float x = dir[0], y = dir[1], z = dir[2];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    switch (basis_dim) {
        case 9:
            out[4] = C2[0] * xy;
            out[5] = C2[1] * yz;
            out[6] = C2[2] * (2.0 * zz - xx - yy);
            out[7] = C2[3] * xz;
            out[8] = C2[4] * (xx - yy);
            [[fallthrough]];
        case 4:
            out[1] = -C1 * y;
            out[2] = C1 * z;
            out[3] = -C1 * x;
    }
}

__device__ uint32_t find_min_active_lane_id(uint32_t active_mask){
    uint32_t res = 0;
    for(; res<32; ++res){
        if(active_mask & 0x00000001 == 1){
            return res;
        }
        active_mask = active_mask >> 1;
    }
    return res;
}
#define FULL_MASK 0xffffffff

template<uint32_t SMSIZE, typename T, uint32_t D>
__global__ void draw_backward_kernel(
    const float * gaussian_pos,
    const float * gaussian_rgb_coeff,
    const float * gaussian_opa,
    const float * gaussian_cov,
    const int * tile_n_point_accum, 
    const float * output,
    const float * grad_output,
    float * grad_pos, 
    float * grad_rgb_coeff,
    float * grad_opa,
    float * grad_cov,
    const float focal_x, 
    const float focal_y,
    const uint32_t w,
    const uint32_t h,
    const bool weight_normalize,
    const bool sigmoid,
    const bool fast,
    const float* rays_o,
    const float* lefttop_pos,
    const float* vec_dx,
    const float* vec_dy,
    bool use_sh_coeff
){
    uint32_t id_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t id_y = blockDim.y * blockIdx.y + threadIdx.y;
    bool is_valid = !(id_x>=w || id_y>=h);
    if(id_x>=w || id_y>=h) return;
    uint32_t id_tile = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t start_idx = tile_n_point_accum[id_tile];
    uint32_t end_idx = tile_n_point_accum[id_tile+1];
    // const uint32_t interval_length = end_idx - start_idx;
   
    uint32_t id_thread = threadIdx.x + threadIdx.y * blockDim.x;
    uint32_t blocksize = blockDim.x * blockDim.y;

    uint32_t lane_id = id_thread % 32;

    // direction calculation
    float current_dir[3];
    float SH[9];
    float _norm = 0.0f;
    if(use_sh_coeff){
        #pragma unroll
        for(uint32_t _i=0; _i<3; ++_i){
            current_dir[_i] = lefttop_pos[_i] + id_x * vec_dx[_i] + id_y * vec_dy[_i] - rays_o[_i];
            _norm += current_dir[_i] * current_dir[_i];
        }
        _norm = sqrtf(_norm);
        #pragma unroll
        for(uint32_t _i=0; _i<3;++_i){
            current_dir[_i] = current_dir[_i] / (_norm + 1e-7);
        }
        calc_sh(9, current_dir, SH);
    }

    __shared__ float _gaussian_pos[SMSIZE*2];
    __shared__ float _gaussian_rgb_coeff[SMSIZE*D];
    __shared__ float _gaussian_opa[SMSIZE*1];
    __shared__ float _gaussian_cov[SMSIZE*4];
    __shared__ float _grad_pos[SMSIZE*2];
    __shared__ float _grad_rgb_coeff[SMSIZE*D];
    __shared__ float _grad_opa[SMSIZE*1];
    __shared__ float _grad_cov[SMSIZE*4];

    // initialize shared memory for gradients
    for(uint32_t i=id_thread; i<SMSIZE; i+=blocksize){
        #pragma unroll
        for(uint32_t _m=0; _m<2; ++_m){
            _grad_pos[i*2+_m] = 0;
        }
        #pragma unroll
        for(uint32_t _m=0; _m<D; ++_m){
            _grad_rgb_coeff[i*D+_m] = 0;
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
    T current_prob = 0.0;
    T current_prob_c0, current_prob0, current_prob1;
    float accum = 1.0;
    float accum_weight = 0.0;
    T _a, _b, _c, _d, _x, _y, det;
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
            #pragma unroll
            for(uint32_t _i_rgb_coeff=0; _i_rgb_coeff<D; ++_i_rgb_coeff){
                _gaussian_rgb_coeff[i*D + _i_rgb_coeff] = gaussian_rgb_coeff[(start_idx + i_loadings*SMSIZE + i)*D + _i_rgb_coeff];
            }
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
            if(global_idx>=(end_idx-start_idx)||accum < 0.0001){
                break;
            }

            _a = _gaussian_cov[i*4+0];
            _b = _gaussian_cov[i*4+1];
            _c = _gaussian_cov[i*4+2];
            _d = _gaussian_cov[i*4+3];
            _x = pixel_x - _gaussian_pos[i*2 + 0];
            _y = pixel_y - _gaussian_pos[i*2 + 1];
            det = (_a * _d - _b * _c);

            T Pm = -(_d * _x * _x - (_b + _c) * _x * _y + _a * _y * _y);
            T Pn = (2 * det + 1e-14);

            current_prob_c0 = sigmoid ? 1.0/2*3.1415926536 : 1.0;
            current_prob0 = sigmoid ? current_prob_c0 * rsqrtf(det+1e-7) : 1.0;
            if(fast){
                current_prob1 = __expf(Pm / Pn);
            }
            else{
                current_prob1 = exp(Pm / Pn);
            }
            current_prob = current_prob0 * current_prob1;

            alpha = current_prob * _gaussian_opa[i];
            // sigmoid + scale -> 0-1
            if(sigmoid){
                alpha = 2./(exp(-alpha)+1) - 1;
            }

            // gradient primitives for 2d gaussian
            T dPm_da = - (_y * _y);
            T dPm_db = _x * _y;
            T dPm_dc = _x * _y;
            T dPm_dd = - (_x * _x);
            T dPn_da = 2 * _d;
            T dPn_db = -2 * _c;
            T dPn_dc = -2 * _b;
            T dPn_dd = 2 * _a;
            T dP1_da = current_prob1 * (dPm_da*Pn - dPn_da*Pm) / (Pn*Pn);
            T dP1_db = current_prob1 * (dPm_db*Pn - dPn_db*Pm) / (Pn*Pn);
            T dP1_dc = current_prob1 * (dPm_dc*Pn - dPn_dc*Pm) / (Pn*Pn);
            T dP1_dd = current_prob1 * (dPm_dd*Pn - dPn_dd*Pm) / (Pn*Pn);

            T dP0_da = sigmoid ? -0.5 * (current_prob0 * current_prob0 * current_prob0) / (current_prob_c0 * current_prob_c0) * _d : 0.0;
            T dP0_db = sigmoid ? 0.5 * (current_prob0 * current_prob0 * current_prob0) / (current_prob_c0 * current_prob_c0) * _c : 0.0;
            T dP0_dc = sigmoid ? 0.5 * (current_prob0 * current_prob0 * current_prob0) / (current_prob_c0 * current_prob_c0) * _b : 0.0;
            T dP0_dd = sigmoid ? -0.5 * (current_prob0 * current_prob0 * current_prob0) / (current_prob_c0 * current_prob_c0) * _a : 0.0;

            T dP_da = current_prob0 * dP1_da + current_prob1 * dP0_da;
            T dP_db = current_prob0 * dP1_db + current_prob1 * dP0_db;
            T dP_dc = current_prob0 * dP1_dc + current_prob1 * dP0_dc;
            T dP_dd = current_prob0 * dP1_dd + current_prob1 * dP0_dd;
            // gradient w.r.t position (not _x, _y, with a minus)
            T dP_dx = current_prob / Pn * (2*_d*_x - _b*_y - _c*_y);
            T dP_dy = current_prob / Pn * (2*_a*_y - _b*_x - _c*_x);
            
            weight = alpha * accum;

            float cur_point_color[] = {0.0, 0.0, 0.0};
            if(use_sh_coeff){
                #pragma unroll
                for(uint32_t _i_channel=0; _i_channel<3; ++_i_channel){
                    #pragma unroll
                    for(uint32_t _i_sh=0; _i_sh<9; ++_i_sh){
                        cur_point_color[_i_channel] += SH[_i_sh] * _gaussian_rgb_coeff[i*27 + _i_channel*9 + _i_sh];
                    }
                }

                #pragma unroll
                for(uint32_t _i_channel=0; _i_channel<3; ++_i_channel){
                    cur_point_color[_i_channel] = 1./(1 + __expf(-cur_point_color[_i_channel]));
                }
            }
            else{
                cur_point_color[0] = _gaussian_rgb_coeff[i*3 + 0];
                cur_point_color[1] = _gaussian_rgb_coeff[i*3 + 1];
                cur_point_color[2] = _gaussian_rgb_coeff[i*3 + 2];
            }

            color[0] += cur_point_color[0] * weight;
            color[1] += cur_point_color[1] * weight;
            color[2] += cur_point_color[2] * weight;
            accum_weight += weight;

            // grad w.r.t gaussian rgb
            if(use_sh_coeff){
                float D0 = cur_grad_out[0]*weight*(cur_point_color[0]*(1-cur_point_color[0]));
                float D1 = cur_grad_out[1]*weight*(cur_point_color[1]*(1-cur_point_color[1]));
                float D2 = cur_grad_out[2]*weight*(cur_point_color[2]*(1-cur_point_color[2]));
                #pragma unroll
                for(uint32_t _i_sh=0; _i_sh<9; ++_i_sh){
                    float grad_wrt_color[3];
                    grad_wrt_color[0] = D0*SH[_i_sh];
                    grad_wrt_color[1] = D1*SH[_i_sh];
                    grad_wrt_color[2] = D2*SH[_i_sh];
                    uint32_t active_mask = __activemask();
                    uint32_t first_lane_id = find_min_active_lane_id(active_mask);
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2){
                        grad_wrt_color[0] += __shfl_down_sync(active_mask, grad_wrt_color[0], offset);
                        grad_wrt_color[1] += __shfl_down_sync(active_mask, grad_wrt_color[1], offset);
                        grad_wrt_color[2] += __shfl_down_sync(active_mask, grad_wrt_color[2], offset);
                    }
                    if(first_lane_id < 32 && lane_id == first_lane_id){
                        atomicAdd(_grad_rgb_coeff+i*27+0+_i_sh, grad_wrt_color[0]);
                        atomicAdd(_grad_rgb_coeff+i*27+9+_i_sh, grad_wrt_color[1]);
                        atomicAdd(_grad_rgb_coeff+i*27+18+_i_sh, grad_wrt_color[2]);
                    }
                }
            }
            else{
                float grad_wrt_color_0 = cur_grad_out[0]*weight;
                float grad_wrt_color_1 = cur_grad_out[1]*weight;
                float grad_wrt_color_2 = cur_grad_out[2]*weight;
                uint32_t active_mask = __activemask();
                uint32_t first_lane_id = find_min_active_lane_id(active_mask);
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2){
                    grad_wrt_color_0 += __shfl_down_sync(active_mask, grad_wrt_color_0, offset);
                    grad_wrt_color_1 += __shfl_down_sync(active_mask, grad_wrt_color_1, offset);
                    grad_wrt_color_2 += __shfl_down_sync(active_mask, grad_wrt_color_2, offset);
                }
                if(first_lane_id < 32 && lane_id == first_lane_id){
                    atomicAdd(_grad_rgb_coeff+i*3+0, grad_wrt_color_0);
                    atomicAdd(_grad_rgb_coeff+i*3+1, grad_wrt_color_1);
                    atomicAdd(_grad_rgb_coeff+i*3+2, grad_wrt_color_2);
                }
            }

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
            float grad_wrt_opa = (float)(d_alpha*current_prob);
            uint32_t active_mask = __activemask();
            uint32_t first_lane_id = find_min_active_lane_id(active_mask);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                grad_wrt_opa += __shfl_down_sync(active_mask, grad_wrt_opa, offset);
            }
            if(first_lane_id < 32 && lane_id == first_lane_id){
                atomicAdd(_grad_opa+i*1+0, grad_wrt_opa);
            }

            float d_current_prob = d_alpha * _gaussian_opa[i];
            // grad w.r.t pos
            float grad_wrt_pos[2];
            grad_wrt_pos[0] = (float)(d_current_prob * dP_dx);
            grad_wrt_pos[1] = (float)(d_current_prob * dP_dy);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                grad_wrt_pos[0] += __shfl_down_sync(active_mask, grad_wrt_pos[0], offset);
                grad_wrt_pos[1] += __shfl_down_sync(active_mask, grad_wrt_pos[1], offset);
            }
            if(first_lane_id < 32 && lane_id == first_lane_id){
                atomicAdd(_grad_pos+i*2+0, grad_wrt_pos[0]);
                atomicAdd(_grad_pos+i*2+1, grad_wrt_pos[1]);
            }
            // grad w.r.t cov
            float grad_wrt_cov[4];
            grad_wrt_cov[0] = (float)(d_current_prob * dP_da);
            grad_wrt_cov[1] = (float)(d_current_prob * dP_db);
            grad_wrt_cov[2] = (float)(d_current_prob * dP_dc);
            grad_wrt_cov[3] = (float)(d_current_prob * dP_dd);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2){
                grad_wrt_cov[0] += __shfl_down_sync(active_mask, grad_wrt_cov[0], offset);
                grad_wrt_cov[1] += __shfl_down_sync(active_mask, grad_wrt_cov[1], offset);
                grad_wrt_cov[2] += __shfl_down_sync(active_mask, grad_wrt_cov[2], offset);
                grad_wrt_cov[3] += __shfl_down_sync(active_mask, grad_wrt_cov[3], offset);
            }
            if(first_lane_id < 32 && lane_id == first_lane_id){
                atomicAdd(_grad_cov+i*4+0, grad_wrt_cov[0]);
                atomicAdd(_grad_cov+i*4+1, grad_wrt_cov[1]);
                atomicAdd(_grad_cov+i*4+2, grad_wrt_cov[2]);
                atomicAdd(_grad_cov+i*4+3, grad_wrt_cov[3]);
            }

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

            #pragma unroll
            for(uint32_t _i_rgb_coeff=0; _i_rgb_coeff<D; ++_i_rgb_coeff){
                grad_rgb_coeff[(start_idx + i_loadings*SMSIZE + i)*D + _i_rgb_coeff] = _grad_rgb_coeff[i*D + _i_rgb_coeff];
            }

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


template<uint32_t SMSIZE, typename T, uint32_t D>
__global__ void draw_kernel(
    // Gaussian3ds & tile_sorted_gaussians, 
    const float * gaussian_pos,
    const float * gaussian_rgb_coeff,
    const float * gaussian_opa,
    const float * gaussian_cov,
    const int * tile_n_point_accum, 
    float * res, 
    const float focal_x, 
    const float focal_y,
    const uint32_t w,
    const uint32_t h,
    const bool weight_normalize,
    const bool sigmoid,
    const bool fast,
    // for direction
    float* rays_o,
    const float* lefttop_pos,
    const float* vec_dx,
    const float* vec_dy,
    bool use_sh_coeff
){
    uint32_t id_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t id_y = blockDim.y * blockIdx.y + threadIdx.y;
    if(id_x>=w || id_y>=h) return;
    uint32_t id_tile = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t start_idx = tile_n_point_accum[id_tile];
    uint32_t end_idx = tile_n_point_accum[id_tile+1];
    // const uint32_t interval_length = end_idx - start_idx;
    uint32_t id_thread = threadIdx.x + threadIdx.y * blockDim.x;
    uint32_t blocksize = blockDim.x * blockDim.y;
    //draw: access all point with early stop
    float pixel_x = (id_x + 0.5 - w/2)/focal_x;
    float pixel_y = (id_y + 0.5 - h/2)/focal_y;
    float color[] = {0, 0, 0};
    float accum = 1.0;
    float accum_weight = 0.0;

    // direction calculation
    float current_dir[3];
    float _norm = 0.0f;
    float SH[9];
    if(use_sh_coeff){
        #pragma unroll
        for(uint32_t _i=0; _i<3; ++_i){
            current_dir[_i] = lefttop_pos[_i] + id_x * vec_dx[_i] + id_y * vec_dy[_i] - rays_o[_i];
            _norm += current_dir[_i] * current_dir[_i];
        }
        _norm = sqrtf(_norm);
        #pragma unroll
        for(uint32_t _i=0; _i<3; ++_i){
            current_dir[_i] = current_dir[_i] / (_norm + 1e-7);
        }
        calc_sh(9, current_dir, SH);
    }

    __shared__ float _gaussian_pos[SMSIZE*2];
    __shared__ float _gaussian_rgb_coeff[SMSIZE*D];
    __shared__ float _gaussian_opa[SMSIZE*1];
    __shared__ float _gaussian_cov[SMSIZE*4];

    // double current_prob = 0.0;
    // double _a, _b, _c, _d, _x, _y, det;
    T current_prob = 0.0;
    T _a, _b, _c, _d, _x, _y, det;

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
            #pragma unroll
            for(int _channel=0; _channel<D; ++_channel){
                _gaussian_rgb_coeff[i*D + _channel] = gaussian_rgb_coeff[(start_idx + i_loadings*SMSIZE + i)*D + _channel];
            }
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
            if(global_idx>=(end_idx-start_idx)||accum < 0.0001){
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
            if(fast){
                current_prob *= __expf(-(_d * _x * _x - (_b + _c) * _x * _y + _a * _y * _y) / (2 * det+1e-14));
            }
            else{
                current_prob *= exp(-(_d * _x * _x - (_b + _c) * _x * _y + _a * _y * _y) / (2 * det+1e-14));
            }

            alpha = current_prob * _gaussian_opa[i];
            // printf("current_alpha: %f\n", alpha);
            // sigmoid + scale -> 0-1
            if(sigmoid){
                alpha = 2./(exp(-alpha)+1) - 1;
            }
            weight = alpha * accum;
            // printf("a: %f, b: %f, c: %f, d: %f, x: %f, y: %f, det: %f, current_prob: %f, weight: %f\n", _a, _b, _c, _d, _x, _y, det, current_prob, weight);
            // color related

            if(use_sh_coeff){
                float current_gaussian_rgb[] = {0.0, 0.0, 0.0};
                #pragma unroll
                for(uint32_t _i_channel=0; _i_channel<3; ++_i_channel){
                    #pragma unroll
                    for(uint32_t _i_sh=0; _i_sh<9; ++_i_sh){
                        current_gaussian_rgb[_i_channel] += SH[_i_sh] * _gaussian_rgb_coeff[i*27 + _i_channel*9 + _i_sh];
                    }
                }
                #pragma unroll
                for(uint32_t _i_channel=0; _i_channel<3; ++_i_channel){
                    current_gaussian_rgb[_i_channel] = 1./(1 + __expf(-current_gaussian_rgb[_i_channel]));
                }
                color[0] += current_gaussian_rgb[0] * weight;
                color[1] += current_gaussian_rgb[1] * weight;
                color[2] += current_gaussian_rgb[2] * weight;
            }
            else{
                color[0] += _gaussian_rgb_coeff[i*3 + 0] * weight;
                color[1] += _gaussian_rgb_coeff[i*3 + 1] * weight;
                color[2] += _gaussian_rgb_coeff[i*3 + 2] * weight;
            }

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
    bool sigmoid,
    bool fast,
    torch::Tensor rays_o,
    torch::Tensor lefttop_pos,
    torch::Tensor vec_dx,
    torch::Tensor vec_dy,
    bool use_sh_coeff
){
    uint32_t h = res.size(0);
    uint32_t w = res.size(1); 
    uint32_t gridsize_x = DIV_ROUND_UP(w, 16);
    uint32_t gridsize_y = DIV_ROUND_UP(h, 16);
    dim3 gridsize(gridsize_x, gridsize_y, 1);
    dim3 blocksize(16, 16, 1);
    // uint32_t SMSIZE;
    // SMSIZE = use_sh_coeff ? 340 : 1200;
    if(use_sh_coeff){
        draw_kernel<340, float, 27><<<gridsize, blocksize>>>(
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
            sigmoid,
            fast,
            rays_o.data_ptr<float>(),
            lefttop_pos.data_ptr<float>(),
            vec_dx.data_ptr<float>(),
            vec_dy.data_ptr<float>(),
            use_sh_coeff
        );
    }
    else{
        draw_kernel<1200, float, 3><<<gridsize, blocksize>>>(
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
            sigmoid,
            fast,
            rays_o.data_ptr<float>(),
            lefttop_pos.data_ptr<float>(),
            vec_dx.data_ptr<float>(),
            vec_dy.data_ptr<float>(),
            use_sh_coeff
        );
    }
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
    bool sigmoid,
    bool fast,
    torch::Tensor rays_o,
    torch::Tensor lefttop_pos,
    torch::Tensor vec_dx,
    torch::Tensor vec_dy,
    bool use_sh_coeff
){
    uint32_t h = output.size(0);
    uint32_t w = output.size(1); 
    uint32_t gridsize_x = DIV_ROUND_UP(w, 16);
    uint32_t gridsize_y = DIV_ROUND_UP(h, 16);
    dim3 gridsize(gridsize_x, gridsize_y, 1);
    dim3 blocksize(16, 16, 1);
    if(use_sh_coeff){
        draw_backward_kernel<160, float, 27><<<gridsize, blocksize>>>(
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
            sigmoid,
            fast,
            rays_o.data_ptr<float>(),
            lefttop_pos.data_ptr<float>(),
            vec_dx.data_ptr<float>(),
            vec_dy.data_ptr<float>(),
            use_sh_coeff
        );
    }
    else{
        draw_backward_kernel<500, float, 3><<<gridsize, blocksize>>>(
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
            sigmoid,
            fast,
            rays_o.data_ptr<float>(),
            lefttop_pos.data_ptr<float>(),
            vec_dx.data_ptr<float>(),
            vec_dy.data_ptr<float>(),
            use_sh_coeff
        );
    }
    
}
