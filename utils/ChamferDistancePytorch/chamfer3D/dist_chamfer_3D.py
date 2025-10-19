from torch import nn
from torch.autograd import Function
import torch
import importlib.util
import os

# 🔧 Python 3.12 compatible check for compiled chamfer_3D extension
chamfer_found = importlib.util.find_spec("chamfer_3D") is not None

if not chamfer_found:
    print("⚠️ Chamfer 3D CUDA extension not found — attempting JIT build.")
    from torch.utils.cpp_extension import load

    chamfer_3D = load(
        name="chamfer_3D",
        sources=[
            "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
            "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
        ],
    )
    print("✅ Loaded JIT 3D CUDA chamfer distance.")
else:
    import chamfer_3D
    print("✅ Loaded compiled 3D CUDA chamfer distance.")

# Chamfer's distance module
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n, device=device)
        dist2 = torch.zeros(batchsize, m, device=device)
        idx1 = torch.zeros(batchsize, n, dtype=torch.int32, device=device)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int32, device=device)

        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros_like(xyz1, device=device)
        gradxyz2 = torch.zeros_like(xyz2, device=device)

        chamfer_3D.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2

class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)
