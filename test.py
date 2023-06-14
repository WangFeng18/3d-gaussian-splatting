import torch
import fmatmul as mul
import time

K = 4096
scale = 1
version = 4
dtype=torch.float
a = scale * torch.randn(K, K, dtype=dtype, device='cuda')
b = scale * torch.randn(K, K, dtype=dtype, device='cuda')
c = torch.zeros(K, K, dtype=dtype, device='cuda')

tic = torch.cuda.Event(enable_timing=True)
toc = torch.cuda.Event(enable_timing=True)
_rep = 10
mul.matmul(a, b, c, version)
tic.record()
for i in range(_rep):
    mul.matmul(a, b, c, version)
toc.record()
torch.cuda.synchronize()
elapse = tic.elapsed_time(toc) / 1000
print(elapse)
flops = (2*K**3 * 1e-9 * _rep)/elapse
print(flops)
print(((c-a@b).abs()>1e-2).sum())