import torch
import gaussian

def jacobian_torch(a):
    _rsqr = 1./(a[:, 0]**2 + a[:, 1]**2 + a[:, 2]**2).sqrt()
    _res = [
        1/a[:,2], torch.zeros_like(a[:,0]), -a[:,0]/(a[:,2]**2),
        torch.zeros_like(a[:,0]), 1/a[:,2], -a[:,1]/(a[:,2]**2),
        _rsqr * a[:, 0], _rsqr * a[:, 1], _rsqr * a[:, 2]
    ]
    return torch.stack(_res, dim=-1).reshape(-1, 3, 3)

def jacobian_cuda(a):
    res = torch.empty(a.shape[0], 3, 3).to(a.device)
    gaussian.jacobian(a, res)
    return res

tic = torch.cuda.Event(enable_timing=True)
toc = torch.cuda.Event(enable_timing=True)

a = torch.randn(1000000, 3).abs() + 0.1
a = a.cuda()

tic.record()
torch_jacobian = jacobian_torch(a)
toc.record()
torch.cuda.synchronize()
elapse = tic.elapsed_time(toc) / 1000
print(elapse)

tic.record()
cuda_jacobian = jacobian_cuda(a)
toc.record()
torch.cuda.synchronize()
elapse = tic.elapsed_time(toc) / 1000
print(elapse)
print(((cuda_jacobian-torch_jacobian).abs()>0.01).sum())