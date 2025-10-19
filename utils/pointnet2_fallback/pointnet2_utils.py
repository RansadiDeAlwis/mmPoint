import torch

def grouping_operation(points, idx):
    # Fallback using simple gather
    # points: (B, C, N)
    # idx: (B, npoints, nsample)
    B, C, N = points.shape
    _, npoints, nsample = idx.shape
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)
    grouped = torch.gather(points.unsqueeze(2).expand(-1, -1, npoints, -1), 3, idx_expanded)
    return grouped.contiguous()

def knn_point(k, xyz, new_xyz):
    # Simple brute-force KNN fallback
    dist = torch.cdist(new_xyz.transpose(1, 2), xyz.transpose(1, 2))
    idx = dist.topk(k, largest=False)[1]
    return idx, dist
