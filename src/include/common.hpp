#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <torch/torch.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x)                                                     \
  CHECK_CPU(x);                                                                \
  CHECK_CONTIGUOUS(x)

#if defined(__CUDACC__)
// #define _EXP(x) expf(x) // SLOW EXP
#define _EXP(x) __expf(x) // FAST EXP
#define _SIGMOID(x) (1 / (1 + _EXP(-(x))))

#else

#define _EXP(x) expf(x)
#define _SIGMOID(x) (1 / (1 + expf(-(x))))
#endif
#define _SQR(x) ((x) * (x))

#define DIV_ROUND_UP(X, Y) ((X) + (Y) - 1) / (Y)

struct Tiles{
  torch::Tensor top;
  torch::Tensor bottom;
  torch::Tensor left;
  torch::Tensor right;

  inline void check() {
    CHECK_INPUT(top);
    CHECK_INPUT(bottom);
    CHECK_INPUT(left);
    CHECK_INPUT(right);
  }
  
  uint32_t len(){
    return top.size(0);
  }
};

struct Gaussian3ds {
  torch::Tensor pos;
  torch::Tensor rgb;
  torch::Tensor opa;
  torch::Tensor quat;
  torch::Tensor scale;
  torch::Tensor cov;

  inline void check() {
    CHECK_INPUT(pos);
    CHECK_INPUT(rgb);
    CHECK_INPUT(opa);
    CHECK_INPUT(quat);
    CHECK_INPUT(scale);
    CHECK_INPUT(cov);
  }

  uint32_t len(){
    return pos.size(0);
  }
};

#endif