from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops
from util import block
from model.pointnet2 import paconv

from time import time
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_four(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, N, C]
        new_points: sampled points data, [B, N, k, D]
        grouped_xyz: grouped points data, [B, N, k, C]
        fps_points: FPS points data, [B, N, D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        #grouped_points = index_points(points, idx)
        fps_points = index_points(points, fps_idx)
        #fps_points = torch.cat([new_xyz, fps_points], dim=-1)
        new_points = index_points(points, idx)
        #new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
        fps_points = new_xyz
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points
    else:
        return new_xyz, new_points



class GraphAttention(nn.Module):
    def __init__(self,batchsize,npoint,feature_dim,dropout=0.6, alpha=0.2):
        super(GraphAttention, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.ones(size=(feature_dim+3, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.b = nn.Parameter(torch.ones(size=(2, 1, npoint, feature_dim)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        #self.c1 = nn.Parameter(torch.zeros(size=(feature_dim, 1)))
        #nn.init.xavier_uniform_(self.c1.data, gain=1.414)
        #self.c2 = nn.Parameter(torch.zeros(size=(3, 1)))
        #nn.init.xavier_uniform_(self.c2.data, gain=1.414)
        #self.c11 = nn.Parameter(torch.zeros(size=(feature_dim, feature_dim)))
        #nn.init.xavier_uniform_(self.c11.data, gain=1.414)
        #self.c22 = nn.Parameter(torch.zeros(size=(3, feature_dim)))
        #nn.init.xavier_uniform_(self.c22.data, gain=1.414)
        self.c = nn.Parameter(torch.zeros(size=(1, npoint, feature_dim)))
        nn.init.xavier_uniform_(self.c.data, gain=1.414)
        self.bn = nn.BatchNorm2d(feature_dim)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.convf = nn.Conv2d(feature_dim, 1, 1, 1)
        self.convd = nn.Conv2d(3, 1, 1, 1)
        self.conva = nn.Conv1d(feature_dim * 3, feature_dim, 1, 1)
        self.bnf = nn.BatchNorm2d(1)
        self.bnd = nn.BatchNorm2d(1)
        self.bna = nn.BatchNorm1d(feature_dim)
        #self.convc = block.Conv2d(feature_dim * 2 + 3, 2, bn=True, activation=None)



    def forward(self, center_feature, grouped_feature, delta, delta_xyz):
        '''
        Input:
            center_feature: centered point feature [B, npoint, nsample, D]
            grouped_feature: sampled points feature [B, npoint, nsample, D]
            delta_feature: sampled points feature [B, npoint, nsample, D]
            delta_xyz: group xyz data [B, npoint, nsample, C]
        Return:
            graph_pooling: results of graph pooling [B, npoint, D]
        '''
        #e = self.leakyrelu(torch.matmul(delta, self.a)) # [B, npoint, nsample,D]
        e = self.leakyrelu(torch.matmul(torch.cat([delta,delta_xyz],dim=-1), self.a))  # [B, npoint, nsample,D]
        #e2 = self.leakyrelu(torch.matmul(delta, self.a2))  # [B, npoint, nsample,D]
        attention1 = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        attention1 = F.dropout(attention1, self.dropout,training=self.training)  # [B, npoint, nsample,D]
        attention2 = 1 - attention1 # [B, npoint, nsample,D]
        #f = self.leakyrelu(torch.matmul(delta, self.c1))
        #d = self.leakyrelu(torch.matmul(delta_xyz, self.c2))

        f = F.softmax(self.bnf(self.convf(delta.permute(0,3,2,1))).permute(0,3,2,1),dim=2) + 1          # [B, npoint, nsample,1]
        d = F.softmax(self.bnd(self.convd(delta_xyz.permute(0,3,2,1))).permute(0,3,2,1),dim=2) + 1    # [B, npoint, nsample,1]
        df = (d+0.0001)/(f+0.0001)                         # [B, npoint, nsample,1]
        fd = (f+0.0001)/(d+0.0001)                         # [B, npoint, nsample,1]
        mdf = torch.mean(df,dim = 2).unsqueeze(-1).expand_as(df)
        mfd = torch.mean(fd,dim = 2).unsqueeze(-1).expand_as(fd)
        df = df/(mdf+0.0001)
        fd = fd/(mfd+0.0001)
        attention1 = df * attention1    # [B, npoint, nsample,D]
        attention2 = fd * attention2    # [B, npoint, nsample,D]

        """
        a3 = F.softmax(self.convc(torch.cat((center_feature,grouped_feature,delta_xyz),dim = -1).permute(0,3,2,1)).permute(0,3,2,1),dim=3).permute(3,0,1,2) # [B, npoint, nsample,2]
        a31 = a3[0, ..., ..., ...].unsqueeze(-1)
        a32 = a3[1, ..., ..., ...].unsqueeze(-1)
        attention1 = a31 * attention1
        attention1sum = torch.unsqueeze(torch.sum(attention1,dim = 2), -1).repeat(1, 1, 1, 32).permute(0, 1, 3, 2)
        attention1 = attention1/attention1sum
        attention2 = a32 * attention2
        attention2sum = torch.unsqueeze(torch.sum(attention2,dim = 2), -1).repeat(1, 1, 1, 32).permute(0, 1, 3, 2)
        attention2 = attention2/attention2sum
        """

        graph_pooling1 = torch.sum(torch.mul(attention1, grouped_feature), dim = 2)  # [B, npoint, D]
        graph_pooling2 = torch.sum(torch.mul(attention2, delta), dim = 2)  # [B, npoint, D]

        #center_feature = center_feature.mean(dim = 2)
        #graph_pooling = self.bna(self.conva(torch.cat((center_feature,graph_pooling1,graph_pooling2),dim = -1).permute(0,2,1))).permute(0,2,1)
        #graph_pooling = graph_pooling + self.c.expand_as(graph_pooling)

        b = F.softmax(self.b, dim=0)  # [B, npoint, nsample,D]
        b1 = b[0, ..., ..., ...]
        b2 = b[1, ..., ..., ...]
        graph_pooling = b1.expand_as(graph_pooling1) * graph_pooling1 + b2.expand_as(graph_pooling2) * graph_pooling2
        graph_pooling = graph_pooling + self.c.expand_as(graph_pooling) # [B, npoint, D]


        #print(graph_pooling.shape)
        return graph_pooling


class _PointNet2SAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N0, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, Cin, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, N1, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, Cout, N1)) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        if self.npoint is None:
            self.npoint = xyz.shape[1] // 4
        new_xyz_idx = pointops.furthestsampling(xyz, self.npoint)  # (B, N1)
        new_xyz = pointops.gathering(
            xyz_trans,
            new_xyz_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, 3)
        for i in range(len(self.groupers)):
            new_features, grouped_xyz, _ = self.groupers[i](xyz, new_xyz, features)
            # (B, Cin+3, N1, K), (B, 3, N1, K)
            if isinstance(self.mlps[i], paconv.SharedPAConv):
                new_features = self.mlps[i]((new_features, grouped_xyz))[0]  # (B, Cout, N1, K)
            else:
                new_features = self.mlps[i](new_features)  # (B, Cout, N1, K)
            if self.agg == 'max':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(-1)])  # (B, Cout, N1, 1)
            elif self.agg == 'sum':
                new_features = torch.sum(new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
            elif self.agg == 'avg':
                new_features = torch.mean(new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
            else:
                raise ValueError('Not implemented aggregation mode.')
            new_features = new_features.squeeze(-1)  # (B, Cout, N1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2SAModuleMSG(_PointNet2SAModuleBase):
    r"""Pointnet set abstraction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet_old before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True, use_xyz: bool = True, use_paconv: bool = False,
                 voxel_size=None, args=None, channel1: int = None, channel2: int = None, channel3: int = None, fmlps:[int], fmlp_channel: int = None, bs: int = None):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.bs = bs
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.fmlp_convs = nn.ModuleList()
        self.fmlp_bns = nn.ModuleList()
        self.use_xyz = use_xyz
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
        self.agg = args.get('agg', 'max')
        self.sampling = args.get('sampling', 'fps')
        self.voxel_size = voxel_size
        last_channel = fmlp_channel + 3
        for out_channel in fmlps:
            self.fmlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.fmlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointops.QueryAndGroup(radius, nsample, use_xyz=use_xyz, return_idx=True)
                # if npoint is not None else pointops.GroupAll(use_xyz=use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if use_paconv:
                self.mlps.append(paconv.SharedPAConv(mlp_spec, bn=bn, config=args))
            else:
                self.mlps.append(block.SharedMLP(mlp_spec, bn=bn))


class PointNet2SAModule(PointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True, use_paconv: bool = False,
                 args=None, channel1: int = None, channel2: int = None, channel3: int = None, fmlps:[int], fmlp_channel: int = None, bs: int = None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, use_paconv=use_paconv,
                         args=args, channel1=channel1, channel2=channel2, channel3=channel3, fmlps = fmlps, fmlp_channel=fmlp_channel, bs=bs)


class PointNet2SAModuleCUDA(PointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True, use_paconv: bool = False,
                 args=None, channel1: int = None, channel2: int = None, channel3: int = None, fmlps:[int], fmlp_channel: int = None, bs: int = None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, use_paconv=use_paconv,
                         args=args, channel1=channel1, channel2=channel2, channel3=channel3, fmlps = fmlps, fmlp_channel=fmlp_channel, bs=bs)
        self.bs = bs
        self.np = npoint
        #self.GA1 = GraphAttention(self.bs, self.np, self.channel1)
        #self.GA2 = GraphAttention(self.bs, self.np, self.channel2)
        self.GA3 = GraphAttention(self.bs, self.np, self.channel3)


    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N0, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, Cin, N0) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, N1, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, Cout, N1)) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        if self.npoint is None:
            self.npoint = xyz.shape[1] // 4
        new_xyz_idx = pointops.furthestsampling(xyz, self.npoint)  # (B, N1)
        new_xyz = pointops.gathering(
            xyz_trans,
            new_xyz_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, 3)
        new_features = features
        for i in range(len(self.groupers)):
            for j in range(len(self.mlps[i])):
                _, grouped_xyz, grouped_idx = self.groupers[i](xyz, new_xyz, new_features)
                # (B, Cin+3, N1, K), (B, 3, N1, K), (B, N1, K)
                if self.use_xyz and j == 0:
                    new_features = torch.cat((xyz.permute(0, 2, 1), new_features), dim=1)  # (B, Cin+3, N1, K)
                if isinstance(self.mlps[i], paconv.SharedPAConv):
                    grouped_new_features = self.mlps[i][j]((new_features, grouped_xyz, grouped_idx))[0]  # (B, Cout, N1, K)
                else:
                    raise NotImplementedError
                if self.agg == 'max':
                    new_features = F.max_pool2d(grouped_new_features, kernel_size=[1, grouped_new_features.size(3)])  # (B, Cout, N1, 1)
                elif self.agg == 'sum':
                    new_features = torch.sum(grouped_new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
                else:
                    raise ValueError('Not implemented aggregation mode.')
                xyz = new_xyz
                new_features = new_features.squeeze(-1).contiguous()  # (B, Cout, N1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2SAModuleCUDA_new(PointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True, use_paconv: bool = False,
                 args=None, channel1: int = None, channel2: int = None, channel3: int = None, fmlps:[int], fmlp_channel: int = None, bs: int = None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, use_paconv=use_paconv,
                         args=args, channel1=channel1, channel2=channel2, channel3=channel3, fmlps = fmlps, fmlp_channel=fmlp_channel, bs=bs)
        self.bs = bs
        self.np = npoint
        #self.GA1 = GraphAttention(self.bs, self.np, self.channel1)
        #self.GA2 = GraphAttention(self.bs, self.np, self.channel2)
        self.GA3 = GraphAttention(self.bs, self.np, self.channel3)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N0, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, Cin, N0) tensor of the descriptors of the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, N1, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, Cout, N1)) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        if self.npoint is None:
            self.npoint = xyz.shape[1] // 4
        new_xyz_idx = pointops.furthestsampling(xyz, self.npoint)  # (B, N1)
        new_xyz = pointops.gathering(
            xyz_trans,
            new_xyz_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, 3)
        new_features = features  # (B, N0, Cin)
        fps_features = pointops.gathering(
            features,
            new_xyz_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, Cin)

        for i in range(len(self.groupers)):
            for j in range(len(self.mlps[i])):
                _, grouped_xyz, grouped_idx = self.groupers[i](xyz, new_xyz, new_features)
                # (B, Cin+3, N1, K), (B, 3, N1, K), (B, N1, K)
                if self.use_xyz and j == 0:
                    new_features = torch.cat((xyz.permute(0, 2, 1), new_features), dim=1)  # (B, Cin+3, N0)
                    fps_features = torch.cat((new_xyz.permute(0, 2, 1), fps_features.permute(0, 2, 1)), dim=1) # (B, Cin+3, N1)
                    fps_features = torch.unsqueeze(fps_features, -1).repeat(1, 1, 1, 32)  # (B, Cin+3, N1, K)
                #print(new_features.shape)
                #print(fps_features.shape)
                fps_features = fps_features.permute(0, 1, 3, 2)  # [B, Cin+3, K ,N1]
                #print(fps_features.shape)s
                conv = self.fmlp_convs[j]
                bn = self.fmlp_bns[j]
                fps_features = F.relu(bn(conv(fps_features))).permute(0, 1, 3, 2) # [B, Cout, N1 ,K]
                #print(fps_features.shape)
                if isinstance(self.mlps[i], paconv.SharedPAConv):
                    grouped_new_features = self.mlps[i][j]((new_features, grouped_xyz, grouped_idx))[0]  # (B, Cout, N1, K)
                else:
                    raise NotImplementedError
                #print(fps_features.shape)
                #print(grouped_new_features.shape)
                delta_features = fps_features - grouped_new_features  # (B, Cout, N1, K)
                delta_xyz = torch.unsqueeze(new_xyz, -1).repeat(1, 1, 1, 32).permute(0,2,1,3) - grouped_xyz  # (B, 3, N1, K)
                #print(delta_features.shape)
                #print(delta_xyz.shape)
                k = j
                #print(k)
                fn = grouped_new_features.permute(0, 2, 3, 1)
                """
                if k == 0:
                    _, _, _, nc = fn.size()
                    new_points = self.GA1(grouped_feature=grouped_new_features.permute(0, 2, 3, 1),delta=delta_features.permute(0, 2, 3, 1))
                    new_points = new_points.permute(0, 2, 1)
                    # print(new_points.shape)
                    new_features = new_points  # (B, Cout, N1)
                    # print(new_features.shape)
                elif k == 1:
                    _, _, _, nc = fn.size()
                    new_points = self.GA2(grouped_feature=grouped_new_features.permute(0, 2, 3, 1),delta=delta_features.permute(0, 2, 3, 1))
                    new_points = new_points.permute(0, 2, 1)
                    # print(new_points.shape)
                    new_features = new_points  # (B, Cout, N1)
                    # print(new_features.shape)
                """
                if k == 2:
                    _, _, _, nc = fn.size()
                    new_points = self.GA3(center_feature=fps_features.permute(0, 2, 3, 1),
                                          grouped_feature=grouped_new_features.permute(0, 2, 3, 1),
                                          delta=delta_features.permute(0, 2, 3, 1),
                                          delta_xyz=delta_xyz.permute(0, 2, 3, 1))
                    new_points = new_points.permute(0, 2, 1)
                    # print(new_points.shape)
                    new_features = new_points  # (B, Cout, N1)
                    # print(new_features.shape)
                else:
                    if self.agg == 'max':
                        new_features = F.max_pool2d(grouped_new_features,
                                                    kernel_size=[1, grouped_new_features.size(3)])  # (B, Cout, N1, 1)
                        # fps_features = F.max_pool2d(grouped_fps_features, kernel_size=[1, grouped_fps_features.size(3)])  # (B, Cout, N1, 1)
                    elif self.agg == 'sum':
                        new_features = torch.sum(grouped_new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
                        # fps_features = torch.sum(grouped_fps_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
                    else:
                        raise ValueError('Not implemented aggregation mode.')
                    new_features = new_features.squeeze(-1).contiguous()  # (B, Cout, N1)
                    # fps_features = fps_features.squeeze(-1).contiguous()  # (B, Cout, N1)
                #print(new_features.shape)
                xyz = new_xyz
                new_features = new_features.squeeze(-1).contiguous()  # (B, Cout, N1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class SEModule(nn.Module):
    def __init__(self,feature_dim):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.a = nn.Parameter(torch.zeros(size=(2, 1, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Sequential(nn.Conv1d(feature_dim, feature_dim // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(feature_dim // 16, feature_dim, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        '''
        Input:
            feature: centered point feature [B, D, npoint]
        Return:
            SE_feature: results of graph pooling [B, D, npoint]
        '''
        feature = feature.permute(0,2,1)
        avg = self.avg_pool(feature.permute(0,2,1)).permute(0,2,1)  # [B, 1, D]
        max = self.max_pool(feature.permute(0,2,1)).permute(0,2,1)  # [B, 1, D]
        avg_out = self.fc(avg.permute(0,2,1)).permute(0,2,1)  # [B, 1, D]
        max_out = self.fc(max.permute(0,2,1)).permute(0,2,1)  # [B, 1, D]
        a1 = self.a[0, ..., ...]
        a2 = self.a[1, ..., ...]
        out = a1 * avg_out + a2 * max_out
        #out = self.sigmoid(avg_out + max_out)
        feature = ((out * feature) + feature).permute(0,2,1)  # [B, feature_dim, npoint]
        return feature


class PointNet2FPModule(nn.Module):
    r"""Propagates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], bn: bool = True,  use_paconv=False, args=None):
        super().__init__()
        self.use_paconv = use_paconv
        if self.use_paconv:
            self.mlp = paconv.SharedPAConv(mlp, bn=bn, config=args)
        else:
            self.mlp = block.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)


class FPPoolModule(nn.Module):
    def __init__(self,npoint,feature_dim):
        super(FPPoolModule, self).__init__()
        self.a = nn.Parameter(torch.ones(size=(4, 1, feature_dim, npoint)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        #self.conv1 = block.Conv1d(feature_dim, 1, bn=True)
        #self.conv2 = block.Conv1d(feature_dim, 1, bn=True)
        #self.conv3 = block.Conv1d(feature_dim, 1, bn=True)
        #self.conv4 = block.Conv1d(feature_dim, 1, bn=True)
        #self.bn1 = nn.BatchNorm1d(1)
        #self.bn2 = nn.BatchNorm1d(1)
        #self.bn3 = nn.BatchNorm1d(1)
        #self.bn4 = nn.BatchNorm1d(1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, l04_points, l03_points, l02_points, l01_points):
        '''
        Input:
            feature1、2、3、4: centered point feature [B, npoint, D]
        Return:
            GCN_feature: results of graph pooling [B, npoint, D]
        '''


        a = F.softmax(self.a, dim=0)  # [B, npoint, nsample,D]
        a1 = a[0, ..., ..., ...]
        a2 = a[1, ..., ..., ...]
        a3 = a[2, ..., ..., ...]
        a4 = a[3, ..., ..., ...]
        l_points = a1.expand_as(l01_points) * l01_points + a2.expand_as(l02_points) * l02_points + a3.expand_as(l03_points) * l03_points + a4.expand_as(l04_points) * l04_points
        """

        l04_weights = self.bn4(self.conv4(l04_points))
        l03_weights = self.bn3(self.conv3(l03_points))
        l02_weights = self.bn2(self.conv2(l02_points))
        l01_weights = self.bn1(self.conv1(l01_points))
        l_weights = self.softmax(torch.cat([l04_weights, l03_weights, l02_weights, l01_weights], dim=-1))
        l_weights4 = torch.unsqueeze(l_weights[:, :, 0], 1).expand_as(l04_points)
        l_weights3 = torch.unsqueeze(l_weights[:, :, 1], 1).expand_as(l03_points)
        l_weights2 = torch.unsqueeze(l_weights[:, :, 2], 1).expand_as(l02_points)
        l_weights1 = torch.unsqueeze(l_weights[:, :, 3], 1).expand_as(l01_points)
        l_points = l_weights1 * l01_points + l_weights2 * l02_points + l_weights3 * l03_points + l_weights4 * l04_points
        """

        return l_points


class ChannelGCNModule(nn.Module):
    def __init__(self,feature_dim,dropout=0.6, alpha=0.2):
        super(ChannelGCNModule, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(1, feature_dim, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.w = nn.Parameter(torch.zeros(size=(1, feature_dim, feature_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.conv1 = nn.Conv1d(feature_dim, feature_dim, 1, 1)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, 1, 1)
        self.conv3 = nn.Conv1d(feature_dim, feature_dim, 1, 1)
        self.conv4 = nn.Conv1d(feature_dim, feature_dim, 1, 1)
        self.conva = nn.Conv1d(feature_dim, feature_dim, 1, 1)
        self.convw = nn.Conv1d(feature_dim, feature_dim, 1, 1)
        self.bn1 = nn.BatchNorm1d(feature_dim)
        self.bn2 = nn.BatchNorm1d(feature_dim)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        self.bn4 = nn.BatchNorm1d(feature_dim)
        self.bna = nn.BatchNorm1d(feature_dim)
        self.bnw = nn.BatchNorm1d(feature_dim)

    def forward(self, feature):
        '''
        Input:
            feature: centered point feature [B, D, npoint]
        Return:
            GCN_feature: results of graph pooling [B, D, npoint]
        '''
        feature = feature.permute(0,2,1)
        #print(feature.shape)
        e1 = self.leakyrelu(self.bn1(self.conv1(feature.permute(0,2,1))).permute(0,2,1))
        e2 = torch.transpose(self.leakyrelu(self.bn2(self.conv2(feature.permute(0,2,1))).permute(0,2,1)), 2, 1)
        e0 = self.leakyrelu(torch.matmul(e2, e1))  # [B, feature_dim, feature_dim]
        a = self.leakyrelu(self.bna(self.conva(self.a.permute(0,2,1))).permute(0,2,1))
        w = self.leakyrelu(self.bnw(self.convw(self.w)))
        e = self.leakyrelu(torch.matmul(a,(torch.matmul(e0,w)))) # [B, feature_dim, feature_dim]
        e0 = e
        attention = F.dropout(F.softmax(e0, dim=1), self.dropout,training=self.training)  # [B, feature_dim, feature_dim]
        e3 = self.leakyrelu(self.bn3(self.conv3(feature.permute(0, 2, 1))).permute(0, 2, 1))  # [B, npoint, feature_dim]
        e4 = self.leakyrelu(torch.matmul(e3, attention))  # [B, npoint, feature_dim]
        new_feature = self.bn4(self.conv4(e4.permute(0, 2, 1))).permute(0, 2, 1) + feature
        #print(new_feature.shape)

        return new_feature.permute(0, 2, 1)

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = torch.randn(2, 9, 3, requires_grad=True).cuda()
    xyz_feats = torch.randn(2, 9, 6, requires_grad=True).cuda()

    test_module = PointNet2SAModuleMSG(npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]])
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    # test_module = PointNet2FPModule(mlp=[6, 6])
    # test_module.cuda()
    # from torch.autograd import gradcheck
    # inputs = (xyz, xyz, None, xyz_feats)
    # test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    # print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
