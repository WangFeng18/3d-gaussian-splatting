import torch
from utils import Timer
import gaussian


a = torch.randn(130000, 3).cuda()
b = torch.randn(1,3,3).cuda()
for i in range(10):
    with Timer("@", verbose=True):
        c = gaussian.world2camera(a,b)
