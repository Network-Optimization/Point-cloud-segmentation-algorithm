from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops
from util import block
from model.pointnet2 import paconv

import warnings
warnings.filterwarnings("ignore")

class SEG(nn.Module):
    def __init__(self, channel, r=4):
        super(SEG, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(channel, round(channel // r), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(round(channel // r), channel, bias=False),
        ).cuda()
        self.S = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        # print(x.shape)
        y1 = x.mean(dim=2)
        #y1 = y1.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # print(y1.shape)
        y2, e = x.max(dim=2)
        #y2 = y2.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # print(y2.shape)
        # Excitation
        y1 = y1.cuda()
        y2 = y2.cuda()
        y1 = self.fc(y1)
        y2 = self.fc(y2)
        y = self.S(y1 + y2)
        # Fscale
        y = torch.unsqueeze(y, 2).expand_as(x)
        y = torch.mul(x, y)
        torch.cuda.empty_cache()
        # print(y.shape)
        return y


class SEF(nn.Module):
    def __init__(self, inc, outc, r=4):
        super(SEF, self).__init__()
        self.conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=7, padding=3 ,stride=1).cuda()

        self.S = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        # print(x.shape)
        y1 = x.mean(dim=3)
        #y1 = y1.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #print(y1.shape)
        y2, e = x.max(dim=3)
        #y2 = y2.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #print(y2.shape)
        # Excitation
        #print((y1 + y2).shape)
        y = self.conv((y1 + y2).permute(0,2,1))
        y = self.S(y.permute(0,2,1))
        # Fscale
        y = torch.unsqueeze(y, 3).expand_as(x)
        y = torch.mul(x, y)
        torch.cuda.empty_cache()
        # print(y.shape)
        return y


class FeatureEnhancement(nn.Module):
    '''
    Input:
        g: sampled points position data [B, npoint, C]
        f: centered point feature [B, npoint, D]
        gn: group xyz data [B, npoint, nsample, C]
        fn_feature: sampled points feature [B, npoint, nsample, D]
    Return:
        F: results of feature enhancing [B, npoint, k, C*3+1+(D+3)*3+1]
    '''

    def __init__(self):
        super(FeatureEnhancement, self).__init__()
        self.S = nn.Softmax(dim=-1)

    def cal_feature_dis(self, feature, g_neighbours, f_neighbours):
        d = torch.unsqueeze(feature, 2).expand_as(f_neighbours)
        d = torch.abs(d - f_neighbours)
        feature_dist = torch.unsqueeze(d.mean(dim=-1), -1)
        _, _, _, gc = g_neighbours.size()
        feature_dist = torch.exp(-feature_dist).repeat(1, 1, 1, gc)
        g_neighbours = g_neighbours * feature_dist
        return g_neighbours

    def avg_g(self, geomertry, g_neighbours, f_neighbours, k):
        g = g_neighbours.mean(2)
        gd = torch.abs(self.S(g) - self.S(geomertry))
        gd = torch.unsqueeze(gd, -1).repeat(1, 1, 1, k)
        gd = gd.permute(0, 1, 3, 2)
        f = torch.cat((f_neighbours, gd), -1)
        return f

    def diso(self, A, B):
        C = torch.mul(A - B, A - B)
        # print(A2.shape)
        # print(B2.shape)
        #print(C)
        D = torch.unsqueeze(torch.sqrt(torch.sum(C, dim=3, out=None)), -1)
        #print(D.shape)
        return D

    """
    def diso(self, A, B, b, n):
        A2 = (A ** 2).sum(dim=3).reshape((b, n, 1, -1))
        B2 = (B ** 2).sum(dim=3).reshape((b, n, -1, 1))
        # print(A2.shape)
        # print(B2.shape)
        D = A2 + B2 - 2 * A.matmul(B.permute(0, 1, 3, 2))
        D = torch.unsqueeze(torch.sqrt(torch.sum(D, dim=3, out=None)), -1)
        return D
    """

    def forward(self, g, f, gn, fn, b, n, k, c):
        gn = self.cal_feature_dis(f, gn, fn)
        fn = self.avg_g(g, gn, fn, k)
        f =  torch.cat((f,g), -1)
        pg = torch.unsqueeze(g, 2).expand_as(gn)
        pf = torch.unsqueeze(f, 2).expand_as(fn)
        pg1 = pg - gn
        pf1 = pf - fn
        dg = self.diso(pg, gn)
        df = self.diso(pf, fn)
        # print(dg.shape)
        # print(df.shape)
        pg2 = torch.cat((gn, pg, pg1, dg), -1)
        pf2 = torch.cat((fn, pf, pf1, df), -1)
        #pg2 = torch.cat((gn, pg, pg1), -1)
        #pf2 = torch.cat((fn, pf, pf1), -1)
        #print(pg2.shape)
        #print(pf2.shape)
        #_, _, _, gc = pg2.size()
        #_, _, fn, fc = pf2.size()
        #SE1 = SEG(channel=gc)
        #pg3 = SE1(pg2)
        #SE2 = SEF(inc=fn, outc=fn)
        #pf3 = SE2(pf2)
        #print(pg3.shape)
        #print(pf3.shape)
        pg3 = pg2
        pf3 = pf2
        F = torch.cat((pg3, pf3), -1)
        #print(F.shape)
        #_, _, _, nc = F.size()
        #nconv = nn.Conv2d(in_channels=nc, out_channels=c, kernel_size=1, padding=0, stride=1).cuda()
        #F = nconv(F.permute(0,3,2,1)).permute(0,3,2,1)
        #print(F.shape)

        return F


class HeterogeneousAttention(nn.Module):
    '''
    Input:
        g: sampled points position data [B, npoint, C]
        f: centered point feature [B, npoint, D]
        gn: group xyz data [B, npoint, nsample, C]
        fn_feature: sampled points feature [B, npoint, nsample, D]
        enhanced_feature: sampled points feature [B, npoint, nsample, D']
    Return:
        F: results of attention pooling [B, npoint, k, 2*D']
    '''

    def __init__(self, all_channel, dropout=0.6, alpha=0.2):
        super(HeterogeneousAttention, self).__init__()
        self.all_channel = all_channel
        self.a = nn.Parameter(torch.ones(size=(self.all_channel, self.all_channel)))
        self.dropout = dropout
        self.alpha = alpha
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha).cuda()


    def diso(self, A, B):
        C = torch.mul(A - B, A - B)
        # print(A2.shape)
        # print(B2.shape)
        #print(C)
        D = torch.unsqueeze(torch.sqrt(torch.sum(C, dim=3, out=None)), -1)
        #print(D.shape)
        return D

    """
    def diso(self, A, B, b, n):
        A2 = (A ** 2).sum(dim=3).reshape((b, n, 1, -1))
        B2 = (B ** 2).sum(dim=3).reshape((b, n, -1, 1))
        # print(A2.shape)
        # print(B2.shape)
        D = A2 + B2 - 2 * A.matmul(B.permute(0, 1, 3, 2))
        D = torch.unsqueeze(torch.sqrt(torch.sum(D, dim=3, out=None)), -1)
        return D
    """

    def forward(self, g, f, gn, fn, nfn, b, n, k, c, nc):
        pg = torch.unsqueeze(g, 2).expand_as(gn) - gn
        pf = torch.unsqueeze(f, 2).expand_as(fn) - fn
        dg = self.diso(pg, gn)
        df = self.diso(pf, fn)
        dgf = torch.cat((pg, pf, dg, df), -1)
        #dgf = torch.cat((pg, pf), -1)
        _, _, _, dc = dgf.size()
        nconv = nn.Conv2d(in_channels=dc, out_channels=self.all_channel, kernel_size=1, padding=0, stride=1).cuda()
        disf = nconv(dgf.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        #print(disf.shape)

        #weight1 = torch.exp(-nfn)
        #print(weight1.shape)
        #weight2 = 1 - weight1
        #print(weight2.shape)
        #F1 = torch.mul(weight1, nfn)
        #F2 = torch.mul(weight2, DF)
        #print(F2.shape)
        #F =torch.cat((F1, F2), -1)
        #print(F.shape)
        #print(nfn.shape)
        #print(self.a.shape)

        e = self.leakyrelu(torch.matmul(nfn.cuda(), self.a.cuda())) # [B, npoint, nsample,D]
        attention = torch.nn.functional.softmax(e, dim=2) # [B, npoint, nsample,D]
        attention = torch.nn.functional.dropout(attention, self.dropout, training=self.training)
        #attention1 = torch.exp(-attention)
        #attention2 = 1 - attention1
        #print(attention1.shape)
        #print(attention2.shape)
        #print(nfn.shape)
        #graph_pooling1 = torch.sum(torch.mul(attention1, nfn), dim=2)
        #graph_pooling2 = torch.sum(torch.mul(attention2, disf), dim=2)
        #F = torch.cat((graph_pooling1, graph_pooling2), -1)

        F = torch.sum(torch.mul(attention, nfn), dim=2)  # [B, npoint, D]

        _, _ , nnc = F.size()
        nconv = nn.Conv1d(in_channels=nnc, out_channels=c, kernel_size=1, padding=0, stride=1).cuda()
        F = nconv(F.permute(0,2,1)).permute(0,2,1)

        return F


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
    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True, use_xyz: bool = True,
                 use_paconv: bool = False, voxel_size=None, args=None, channel1: int = None, channel2: int = None, channel3: int = None):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.use_xyz = use_xyz
        self.agg = args.get('agg', 'max')
        self.sampling = args.get('sampling', 'fps')
        self.voxel_size = voxel_size
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
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
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True,
                 use_xyz: bool = True, use_paconv: bool = False, args=None, channel1: int = None, channel2: int = None, channel3: int = None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
                         use_paconv=use_paconv, args=args, channel1=channel1, channel2=channel2, channel3=channel3)


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
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 use_xyz: bool = True, use_paconv: bool = False, args=None, channel1: int = None, channel2: int = None, channel3: int = None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample],
                         use_paconv=use_paconv, args=args, channel1=channel1, channel2=channel2, channel3=channel3)
        self.FE = FeatureEnhancement()
        self.HA1 = HeterogeneousAttention(self.channel1)
        self.HA2 = HeterogeneousAttention(self.channel2)
        self.HA3 = HeterogeneousAttention(self.channel3)

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
        fps_features = pointops.gathering(
            features,
            new_xyz_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, Cin)
        #print(new_xyz.shape)
        #print(fps_features.shape)
        #print(new_features.shape)
        for i in range(len(self.groupers)):
            for j in range(len(self.mlps[i])):
                _, grouped_xyz, grouped_idx = self.groupers[i](xyz, new_xyz, new_features)
                # (B, 3, N1, K), (B, N1, K)
                if self.use_xyz and j == 0:
                    new_features = torch.cat((xyz.permute(0, 2, 1), new_features), dim=1) # (B, Cin+3, N0, K)
                    fps_features = torch.cat((new_xyz, fps_features), dim=-1).permute(0,2,1) # (B, N1, Cin+3)

                # 加权升维
                if isinstance(self.mlps[i], paconv.SharedPAConv):
                    grouped_new_features = self.mlps[i][j]((new_features, grouped_xyz, grouped_idx))[0]  # (B, Cout, N1, K)

                else:
                    raise NotImplementedError
                #_, cout, _, _ = grouped_new_features.size()
                #_, cin, _ = fps_features.size()
                #nconv = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, padding=0, stride=1).cuda()
                #grouped_fps_features = nconv(fps_features)
                #grouped_fps_features = torch.unsqueeze(grouped_fps_features, -1).repeat(1, 1, 1, 32)  # (B, Cout, N1, K)
                #xyz = new_xyz
                #print(grouped_fps_features.shape)
                #print(grouped_new_features.shape)

                """
                # 模块加在这里，替换聚合特征
                #print(new_xyz.shape)  # (B, N1, 3)
                #print(grouped_xyz.shape) # (B, 3, N1, K)
                #print(grouped_fps_features.shape) # (B, Cout, N1)
                #print(grouped_new_features.shape) # (B, Cout, N1, K)
                #grouped_fps_features = torch.unsqueeze(grouped_fps_features, -1).repeat(1, 1, 1, 32)  # (B, Cout, N1, K)
                g = new_xyz
                f = grouped_fps_features.permute(0, 2, 1)
                gn = grouped_xyz.permute(0, 2, 3, 1)
                fn = grouped_new_features.permute(0, 2, 3, 1)
                b, n, k, c = fn.size()
                enhanced_points = self.FE(g=g, f=f, gn=gn, fn=fn, b=b, n=n, k=k, c=c)
                #print(grouped_new_features.shape)
                #grouped_new_features = enhanced_points.permute(0, 3, 1, 2)
                #print(grouped_new_features.shape)
                _, _, _, nc = fn.size()
                HA = HeterogeneousAttention(nc)
                new_points = HA(g=g, f=f, gn=gn, fn=fn, nfn=grouped_new_features.permute(0, 2, 3, 1), b=b, n=n, k=k, c=c, nc=nc)
                new_points = new_points.permute(0, 2, 1)
                #print(new_points.shape)
                new_features = new_points  # (B, Cout, N1)
                #print(new_features.shape)
                """

                """
                #print(grouped_new_features.shape)
                g = new_xyz
                f = grouped_fps_features.permute(0, 2, 1)
                gn = grouped_xyz.permute(0, 2, 3, 1)
                fn = grouped_new_features.permute(0, 2, 3, 1)
                b, n, k, c = fn.size()
                enhanced_points = self.FE(g=g, f=f, gn=gn, fn=fn, b=b, n=n, k=k, c=c)
                
                #print(enhanced_points.shape)
                grouped_new_features = enhanced_points.permute(0, 3, 1, 2)
                #print(j,"   ",c)
                """
                #g = new_xyz
                #f = grouped_fps_features.permute(0, 2, 1)
                #gn = grouped_xyz.permute(0, 2, 3, 1)
                #fn = grouped_new_features.permute(0, 2, 3, 1)
                #b, n, k, c = fn.size()
                #k = j

                # 聚合特征
                # grouped_new_features = grouped_new_features - grouped_fps_features
                if self.agg == 'max':
                    new_features = F.max_pool2d(grouped_new_features,
                                                kernel_size=[1, grouped_new_features.size(3)])  # (B, Cout, N1, 1)
                    # fps_features = F.max_pool2d(grouped_fps_features, kernel_size=[1, grouped_fps_features.size(3)])  # (B, Cout, N1, 1)
                elif self.agg == 'sum':
                    new_features = torch.sum(grouped_new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
                    # fps_features = torch.sum(grouped_fps_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
                else:
                    raise ValueError('Not implemented aggregation mode.')
                xyz = new_xyz
                new_features = new_features.squeeze(-1).contiguous()  # (B, Cout, N1)
                # fps_features = fps_features.squeeze(-1).contiguous()  # (B, Cout, N1)
                # print(new_features.shape)

                """
                if k == 0:
                    _, _, _, nc = fn.size()
                    new_points = self.HA1(g=g, f=f, gn=gn, fn=fn, nfn=grouped_new_features.permute(0, 2, 3, 1), b=b, n=n, k=k,
                                    c=c, nc=nc)
                    new_points = new_points.permute(0, 2, 1)
                    # print(new_points.shape)
                    new_features = new_points  # (B, Cout, N1)
                    # print(new_features.shape)
                elif k == 1:
                    _, _, _, nc = fn.size()
                    new_points = self.HA2(g=g, f=f, gn=gn, fn=fn, nfn=grouped_new_features.permute(0, 2, 3, 1), b=b, n=n, k=k,
                                    c=c, nc=nc)
                    new_points = new_points.permute(0, 2, 1)
                    # print(new_points.shape)
                    new_features = new_points  # (B, Cout, N1)
                    # print(new_features.shape)
                elif k == 2:
                    _, _, _, nc = fn.size()
                    new_points = self.HA3(g=g, f=f, gn=gn, fn=fn, nfn=grouped_new_features.permute(0, 2, 3, 1), b=b, n=n, k=k,
                                    c=c, nc=nc)
                    new_points = new_points.permute(0, 2, 1)
                    # print(new_points.shape)
                    new_features = new_points  # (B, Cout, N1)
                    # print(new_features.shape)
                else:
                    # 聚合特征
                    # grouped_new_features = grouped_new_features - grouped_fps_features
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
                    # print(new_features.shape)
                """
            new_features_list.append(new_features)
        #print(torch.cat(new_features_list, dim=1).shape)
        return new_xyz, torch.cat(new_features_list, dim=1)


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
