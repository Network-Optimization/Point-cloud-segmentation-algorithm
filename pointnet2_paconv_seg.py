from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pointnet2.pointnet2_paconv_modules import PointNet2FPModule,FPPoolModule,ChannelGCNModule,SEModule
from util import block


class PointNet2SSGSeg(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        k: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        c: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, bs, c=3, k=13, use_xyz=True, args=None):
        super().__init__()
        self.bs = bs
        self.nsamples = args.get('nsamples', [32, 32, 32, 32])
        self.npoints = args.get('npoints', [1024, 512, 256, 128])
        self.sa_mlps = args.get('sa_mlps', [[c, 32, 32, 64], [64, 64, 64, 128], [128, 128, 128, 256], [256, 256, 256, 512]])
        #self.fp_mlps = args.get('fp_mlps', [[128 + c, 128, 128, 128], [256 + 64, 256, 128], [256 + 128, 256, 256], [512 + 256, 256, 256]])
        self.fp_mlps4 = args.get('fp_mlps', [[128 + c, 128, 128, 128], [256 + 64, 256, 128], [256 + 128, 256, 256], [512 + 256, 256, 256]])
        self.fp_mlps3 = args.get('fp_mlps', [[256 + 128, 256, 128], [256 + 256, 256, 256], [512 + 256, 256, 256]])
        self.fp_mlps2 = args.get('fp_mlps', [[128 + 384, 256, 256], [512 + 256, 256, 256]])
        self.fp_mlps1 = args.get('fp_mlps', [[512 + 256, 256, 256]])
        self.paconv = args.get('pointnet2_paconv', [True, True, True, True, False, False, False, False])
        self.fc = args.get('fc', 128)

        if args.get('cuda', False):
            from model.pointnet2.pointnet2_paconv_modules import PointNet2SAModuleCUDA_new as PointNet2SAModule
        else:
            from model.pointnet2.pointnet2_paconv_modules import PointNet2SAModule

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[0], nsample=self.nsamples[0], mlp=self.sa_mlps[0], use_xyz=use_xyz, use_paconv=self.paconv[0],
                                                 args=args, channel1=32, channel2=32, channel3=64, fmlps=[32,32,64], fmlp_channel = 6, bs = self.bs))
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[1], nsample=self.nsamples[1], mlp=self.sa_mlps[1], use_xyz=use_xyz, use_paconv=self.paconv[1],
                                                 args=args, channel1=64, channel2=64, channel3=128, fmlps=[64,64,128], fmlp_channel = 64, bs = self.bs))
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[2], nsample=self.nsamples[2], mlp=self.sa_mlps[2], use_xyz=use_xyz, use_paconv=self.paconv[2],
                                                 args=args, channel1=128, channel2=128, channel3=256, fmlps=[128,128,256], fmlp_channel = 128, bs = self.bs))
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[3], nsample=self.nsamples[3], mlp=self.sa_mlps[3], use_xyz=use_xyz, use_paconv=self.paconv[3],
                                                 args=args, channel1=256, channel2=256, channel3=512, fmlps=[256,256,512], fmlp_channel = 256, bs = self.bs))

        #self.SE_modules = nn.ModuleList()
        #self.SE_modules.append(SEModule(feature_dim=64))
        #self.SE_modules.append(SEModule(feature_dim=128))
        #self.SE_modules.append(SEModule(feature_dim=256))
        #self.SE_modules.append(SEModule(feature_dim=512))
        """
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNet2FPModule(mlp=self.fp_mlps[0], use_paconv=self.paconv[4], args=args))
        self.FP_modules.append(PointNet2FPModule(mlp=self.fp_mlps[1], use_paconv=self.paconv[5], args=args))
        self.FP_modules.append(PointNet2FPModule(mlp=self.fp_mlps[2], use_paconv=self.paconv[6], args=args))
        self.FP_modules.append(PointNet2FPModule(mlp=self.fp_mlps[3], use_paconv=self.paconv[7], args=args))

        """
        self.FP_modules4 = nn.ModuleList()
        self.FP_modules4.append(PointNet2FPModule(mlp=self.fp_mlps4[0], use_paconv=self.paconv[4], args=args))
        self.FP_modules4.append(PointNet2FPModule(mlp=self.fp_mlps4[1], use_paconv=self.paconv[4], args=args))
        self.FP_modules4.append(PointNet2FPModule(mlp=self.fp_mlps4[2], use_paconv=self.paconv[4], args=args))
        self.FP_modules4.append(PointNet2FPModule(mlp=self.fp_mlps4[3], use_paconv=self.paconv[4], args=args))
        self.FP_modules3 = nn.ModuleList()
        self.FP_modules3.append(PointNet2FPModule(mlp=self.fp_mlps3[0], use_paconv=self.paconv[4], args=args))
        self.FP_modules3.append(PointNet2FPModule(mlp=self.fp_mlps3[1], use_paconv=self.paconv[4], args=args))
        self.FP_modules3.append(PointNet2FPModule(mlp=self.fp_mlps3[2], use_paconv=self.paconv[4], args=args))
        self.FP_modules2 = nn.ModuleList()
        self.FP_modules2.append(PointNet2FPModule(mlp=self.fp_mlps2[0], use_paconv=self.paconv[4], args=args))
        self.FP_modules2.append(PointNet2FPModule(mlp=self.fp_mlps2[1], use_paconv=self.paconv[4], args=args))
        self.FP_modules1 = nn.ModuleList()
        self.FP_modules1.append(PointNet2FPModule(mlp=self.fp_mlps1[0], use_paconv=self.paconv[4], args=args))
        self.FPPoolModule_layer = FPPoolModule(npoint=4096,feature_dim=128)
        self.Channel_layer = ChannelGCNModule(feature_dim=128)
        self.FC_layer = nn.Sequential(block.Conv2d(self.fc, self.fc, bn=True), nn.Dropout(), block.Conv2d(self.fc, k, activation=None))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            #print(li_features.shape)
            #li_features = self.SE_modules[i](li_features)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        #for i in range(len(self.SE_modules)):
            #print(l_features[i+1].shape)
            #l_features[i + 1] = self.SE_modules[i](l_features[i + 1])

        l_features4 = l_features
        l_features3 = l_features
        l_features2 = l_features
        l_features1 = l_features

        """
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        l_points = l_features[0]
        """
        for i in range(-1, -(len(self.FP_modules4) + 1), -1):
            l_features4[i - 1] = self.FP_modules4[i](l_xyz[i - 1], l_xyz[i], l_features4[i - 1], l_features4[i])
            #print(i)
        l04_points = l_features4[0]
        #print(l04_points.shape)
        for i in range(-1, -(len(self.FP_modules3) + 1), -1):
            l_features3[i - 1] = self.FP_modules3[i](l_xyz[i - 1], l_xyz[i], l_features3[i - 1], l_features3[i])
            #print(i)
        l03_points = l_features3[0]
        #print(l03_points.shape)
        for i in range(-1, -(len(self.FP_modules2) + 1), -1):
            l_features2[i - 1] = self.FP_modules2[i](l_xyz[i - 1], l_xyz[i], l_features2[i - 1], l_features2[i])
            #print(i)
        l02_points = l_features2[0]
        #print(l02_points.shape)
        for i in range(-1, -(len(self.FP_modules1) + 1), -1):
            l_features1[i - 1] = self.FP_modules1[i](l_xyz[i - 1], l_xyz[i], l_features1[i - 1], l_features1[i])
            #print(i)
        l01_points = l_features1[0]
        #print(l01_points.shape)

        l_points = self.FPPoolModule_layer(l04_points,l03_points,l02_points,l01_points)
        #print(l_points.shape)
        l_points = self.Channel_layer(l_points)
        return self.FC_layer(l_points.unsqueeze(-1)).squeeze(-1)


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            preds = model(inputs)
            loss = criterion(preds, labels)
            _, classes = torch.max(preds, 1)
            acc = (classes == labels).float().sum() / labels.numel()
            return ModelReturn(preds, loss, {"acc": acc.item(), 'loss': loss.item()})
    return model_fn


if __name__ == "__main__":
    import torch.optim as optim
    B, N, C, K = 2, 4096, 3, 13
    inputs = torch.randn(B, N, 6)#.cuda()
    labels = torch.randint(0, 3, (B, N))#.cuda()

    model = PointNet2SSGSeg(c=C, k=K)#.cuda()
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    print("Testing SSGCls with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()

    model = PointNet2SSGSeg(c=C, k=K, use_xyz=False).cuda()
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    print("Testing SSGCls without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()
