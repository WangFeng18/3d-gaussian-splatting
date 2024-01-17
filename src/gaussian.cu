#include <stdio.h>
#include "include/common.hpp"
#include <torch/torch.h>
#include <cuda_runtime.h>


__global__ void calc_tile_info_kernel(
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
    uint32_t point_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(point_id >= n_point) return;
    gaussian_pos += point_id * 3;

    // determinant check
    float _a, _b, _c, _d;
    _a = gaussian_cov[point_id * 4 + 0];
    _b = gaussian_cov[point_id * 4 + 1];
    _c = gaussian_cov[point_id * 4 + 2];
    _d = gaussian_cov[point_id * 4 + 3];
    float det = (_a * _d - _b * _c);
    if(det <= 0) {return;}

    float _center_x = gaussian_pos[0];
    float _center_y = gaussian_pos[1];

    float _ai = _d / (det + 1e-14);
    float _bi = -_b / (det + 1e-14);
    float _ci = -_c / (det + 1e-14);
    float _di = _a / (det + 1e-14);

    // compute scaling factor from thresh_dist scaled by det of cov (variance)
    float thresh_dis_log = -2 * logf(thresh_dis);
    float shift_x = sqrtf(_di * thresh_dis_log * det);
    float shift_y = sqrtf(_ai * thresh_dis_log * det);
    float bbx_right = _center_x + shift_x;
    float bbx_left  = _center_x - shift_x;
    float bbx_top = _center_y - shift_y;
    float bbx_bottom = _center_y + shift_y;

    // iterate through tiles and see if gaussian splats onto the tile
    // similar algorithm to prob1 but more efficient
    for(
        uint32_t i_top=fmaxf((bbx_top - topmost) / tile_length_y, 0);
        i_top < (uint32_t)((bbx_bottom - topmost) / tile_length_y + 1) && i_top<n_tiles_y;
        ++i_top
    ){
        for(
            uint32_t i_left=fmaxf((bbx_left-leftmost) / tile_length_x, 0);
            i_left<(uint32_t)((bbx_right-leftmost) / tile_length_x + 1) && i_left<n_tiles_x;
            ++i_left
        ){
            uint32_t tile_id = i_left + i_top * n_tiles_x; 
            if(tile_n_point[tile_id] < max_points_per_tile){
                uint32_t old = atomicAdd(tile_n_point + tile_id, 1);
                tile_gaussian_list[max_points_per_tile * tile_id + old] = point_id;
            }
        }
    }
}


void calc_tile_list(
    Gaussian3ds & gaussians_image_space, // 3D Gaussians
    Tiles & tile_info,                // Tile Info Struct
    torch::Tensor tile_n_point,       // number of points per tile
    torch::Tensor tile_gaussian_list, // point ids per tile
    float thresh, // threshold factor 
    float tile_length_x, // tile length in x in pixels
    float tile_length_y, // tile length in y in pixels 
    int n_tiles_x,  // n_tiles in x 
    int n_tiles_y,  // n_tiles in y   
    float leftmost, // leftmost coordinate of all tiles
    float topmost   // topmost coordinate of all tiles
)
{
    // create intersect info
    uint32_t n_point = gaussians_image_space.len();
    uint32_t n_tiles = tile_info.len();
    uint32_t max_points_per_tile = tile_gaussian_list.size(1);
    uint32_t gridsize_x = DIV_ROUND_UP(n_point, 1024);
    dim3 gridsize(gridsize_x, 1, 1);
    dim3 blocksize(1024, 1, 1);
    calc_tile_info_kernel<<<gridsize, blocksize>>>(
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


// stack gaussian indices from multiple tiles into a single list
__global__ void gather_gaussians_kernel(
    const int * tile_n_point_accum, // start index for gaussians by tile_id
    const int * gaussian_list,      
    int * gather_list,             // output: stacked gaussian indices
    int * tile_ids_for_points,
    int n_tiles,                   // number of tiles
    int max_points_for_tile,       // max points per tile
    int gaussian_list_size         // 
){
    uint32_t tile_id = blockDim.y * blockIdx.y + threadIdx.y;
    if(tile_id >= n_tiles) return;
    uint32_t point_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t shift_res = tile_n_point_accum[tile_id];

    uint32_t n_point_this_tile = tile_n_point_accum[tile_id + 1] - shift_res;
    if(point_id >= n_point_this_tile) return;

    gaussian_list += tile_id * gaussian_list_size;
    gather_list[shift_res + point_id] = gaussian_list[point_id];
    tile_ids_for_points[shift_res + point_id] = tile_id;
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
    uint32_t point_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(point_id >= n_point) return;
    pos += point_id*3;

    // 1. calculate the camera space coordinate
    float pos_c[3];
    world_to_camera(pos, current_rot, current_tran, pos_c);

    // printf("z: %f\n", pos_c[2]);

    // 2. check if the point is before the near plane
    if(pos_c[2] <= near){
        // culling_mask[point_id] = 0;
        return;
    }

    // 3. image space transform
    float pos_i[3];
    pos_i[0] = pos_c[0] / pos_c[2];
    pos_i[1] = pos_c[1] / pos_c[2];
    pos_i[2] = sqrtf(pos_c[0]*pos_c[0] + pos_c[1]*pos_c[1] + pos_c[2]*pos_c[2]);

    // 4. frustum culling
    if(abs(pos_i[0]) >= half_width || abs(pos_i[1]) >= half_height){
        // culling_mask[point_id] = 0;
        return;
    }
    culling_mask[point_id] = 1;
    res_pos[point_id*3 + 0] = pos_i[0];
    res_pos[point_id*3 + 1] = pos_i[1];
    res_pos[point_id*3 + 2] = pos_i[2];


    // 5. calculate the covariance matrix and jacobian
    float w = quat[4*point_id+0];
    float x = quat[4*point_id+1];
    float y = quat[4*point_id+2];
    float z = quat[4*point_id+3];

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
    S[0] = scale[point_id*3+0];
    S[1] = 0;
    S[2] = 0;
    S[3] = 0;
    S[4] = scale[point_id*3+1];
    S[5] = 0;
    S[6] = 0;
    S[7] = 0;
    S[8] = scale[point_id*3+2];

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
    res_cov[point_id*4+0] = JWCWJ[0];
    res_cov[point_id*4+1] = JWCWJ[1];
    res_cov[point_id*4+2] = JWCWJ[3];
    res_cov[point_id*4+3] = JWCWJ[4];
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
    uint32_t point_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(point_id >= n_point) return;
    pos += point_id*3;

    if(culling_mask[point_id]==0){
        return;
    }
    
    // forward pass pos 0,1,2 -> pos_c 0,1,2 -> pos_i 0,1,2
    float pos_c[3];
    world_to_camera(pos, current_rot, current_tran, pos_c);

    float grad_c[3];
    float pos_i_z = sqrtf(pos_c[0]*pos_c[0] + pos_c[1]*pos_c[1] + pos_c[2]*pos_c[2]);
    float grad_i[3];
    grad_i[0] = gradout_pos[point_id*3+0];
    grad_i[1] = gradout_pos[point_id*3+1];
    grad_i[2] = gradout_pos[point_id*3+2];

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
    gradinput_pos[point_id*3+0] = grad_w[0];
    gradinput_pos[point_id*3+1] = grad_w[1];
    gradinput_pos[point_id*3+2] = grad_w[2];

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
        grad_2d_cov[i] = gradout_cov[point_id*4+i];
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
    float w = quat[4*point_id+0];
    float x = quat[4*point_id+1];
    float y = quat[4*point_id+2];
    float z = quat[4*point_id+3];

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
    S[0] = scale[point_id*3+0];
    S[1] = 0;
    S[2] = 0;
    S[3] = 0;
    S[4] = scale[point_id*3+1];
    S[5] = 0;
    S[6] = 0;
    S[7] = 0;
    S[8] = scale[point_id*3+2];

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
                // grad_RS[i_r*3+i_c] += 2 * grad_3d_cov[i_r*3+i_k] * RS[i_c*3+i_k];
                // grad_RS[i_r*3+i_c] += (grad_3d_cov[i_k*3+i_r] + grad_3d_cov[i_r*3+i_k]) * RS[i_c*3+i_k];
                grad_RS[i_r*3+i_c] += (grad_3d_cov[i_k*3+i_r] + grad_3d_cov[i_r*3+i_k]) * RS[i_k*3+i_c];
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
    gradinput_quat[point_id*4+0] = grad_qr;
    gradinput_quat[point_id*4+1] = grad_qi;
    gradinput_quat[point_id*4+2] = grad_qj;
    gradinput_quat[point_id*4+3] = grad_qk;
    
    gradinput_scale[point_id*3+0] = grad_scale[0];
    gradinput_scale[point_id*3+1] = grad_scale[1];
    gradinput_scale[point_id*3+2] = grad_scale[2];
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
