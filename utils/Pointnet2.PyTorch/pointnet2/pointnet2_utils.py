import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

# Safe import of CUDA backend
try:
    import pointnet2_cuda as pointnet2
except Exception:
    pointnet2 = None


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return: (B, npoint) indices
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        if pointnet2 is None or not hasattr(pointnet2, "furthest_point_sampling_wrapper"):
            raise RuntimeError("furthest_point_sampling requires CUDA extension (pointnet2_cuda).")

        pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.empty(B, C, npoint, device=features.device, dtype=features.dtype)

        if pointnet2 is None or not hasattr(pointnet2, "gather_points_wrapper"):
            # Fallback: torch.gather along last dim
            idx_exp = idx.unsqueeze(1).expand(B, C, npoint)              # (B,C,npoint)
            feats_exp = features                                        # (B,C,N)
            output = torch.gather(feats_exp, 2, idx_exp).contiguous()   # (B,C,npoint)
            ctx.for_backwards = (idx, C, N, False)  # False => fallback path
            return output

        pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)
        ctx.for_backwards = (idx, C, N, True)  # True => used CUDA path
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N, used_cuda = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)

        if used_cuda and pointnet2 is not None and hasattr(pointnet2, "gather_points_grad_wrapper"):
            pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out.contiguous(), idx, grad_features)
            return grad_features, None

        # Fallback grad for gather: scatter_add back into (B,C,N)
        # grad_out: (B,C,npoint)
        tmp = torch.zeros(B, C, npoint, N, device=grad_out.device, dtype=grad_out.dtype)
        idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(B, C, npoint, 1)  # (B,C,npoint,1)
        tmp.scatter_add_(3, idx_exp, grad_out.unsqueeze(-1))              # add along last dim
        grad_features = tmp.sum(dim=2)                                    # sum over npoint -> (B,C,N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param unknown: (B, N, 3)
        :param known:   (B, M, 3)
        :return: (dist: B,N,3), (idx: B,N,3)
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        if pointnet2 is None or not hasattr(pointnet2, "three_nn_wrapper"):
            raise RuntimeError("three_nn requires CUDA extension (pointnet2_cuda).")

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Weighted linear interpolation on 3 features
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx:      (B, n, 3) three nearest neighbors of target features
        :param weight:   (B, n, 3) weights
        :return:         (B, C, n)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        if pointnet2 is None or not hasattr(pointnet2, "three_interpolate_wrapper"):
            raise RuntimeError("three_interpolate requires CUDA extension (pointnet2_cuda).")

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        if pointnet2 is None or not hasattr(pointnet2, "three_interpolate_grad_wrapper"):
            raise RuntimeError("three_interpolate backward requires CUDA extension (pointnet2_cuda).")

        grad_features = torch.cuda.FloatTensor(B, c, m).zero_()
        pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out.contiguous(), idx, weight, grad_features)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param features: (B, C, N) tensor of features to group
        :param idx:      (B, npoint, nsample) indices into N
        :return:         (B, C, npoint, nsample)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, C, N = features.size()
        _, npoint, nsample = idx.size()

        # Try CUDA path first
        if (pointnet2 is not None) and hasattr(pointnet2, "group_points_wrapper"):
            output = torch.empty(B, C, npoint, nsample, device=features.device, dtype=features.dtype)
            pointnet2.group_points_wrapper(B, C, N, npoint, nsample, features, idx, output)
            ctx.for_backwards = (idx, N, True)  # used CUDA
            return output

        # ---- Pure PyTorch fallback (forward) ----
        feats_exp = features.unsqueeze(2).expand(B, C, npoint, N)      # (B,C,npoint,N)
        idx_exp   = idx.unsqueeze(1).expand(B, C, npoint, nsample)     # (B,C,npoint,nsample)
        output = torch.gather(feats_exp, 3, idx_exp).contiguous()      # (B,C,npoint,nsample)
        ctx.for_backwards = (idx, N, False)  # used fallback
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param grad_out: (B, C, npoint, nsample)
        :return: grad_features: (B, C, N)
        """
        idx, N, used_cuda = ctx.for_backwards
        B, C, npoint, nsample = grad_out.size()

        # CUDA backward if available
        if used_cuda and (pointnet2 is not None) and hasattr(pointnet2, "group_points_grad_wrapper"):
            grad_features = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)
            pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out.contiguous(), idx, grad_features)
            return grad_features, None

        # ---- Pure PyTorch fallback (backward) ----
        # We invert gather by scatter_add along the gathered dimension and summing over npoint
        grad_features = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)
        tmp = torch.zeros(B, C, npoint, N, device=grad_out.device, dtype=grad_out.dtype)
        idx_exp = idx.unsqueeze(1).expand(B, C, npoint, nsample)         # (B,C,npoint,nsample)
        tmp.scatter_add_(3, idx_exp, grad_out)                           # add along last dim (N)
        grad_features = tmp.sum(dim=2)                                   # sum over npoint
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return: (B, npoint, nsample) indices
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        if pointnet2 is None or not hasattr(pointnet2, "ball_query_wrapper"):
            raise RuntimeError("ball_query requires CUDA extension (pointnet2_cuda).")

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have no features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
