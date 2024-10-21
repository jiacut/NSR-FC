import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from einops import rearrange
import math

def l2norm(vec):
    vec_temp = torch.norm(vec, p=2, dim=1).reshape(-1, 1)
    return vec / vec_temp


def feature_Norm(vec):
    B, N, D = vec.shape
    vec = vec.reshape(-1, D)
    vec_temp = torch.norm(vec, p=2, dim=1).reshape(-1, 1)

    return (vec / vec_temp).reshape(B, N, D)


def A_Norm(A):
    B, N, _ = A.shape
    A = A.reshape(-1, N)
    sum = torch.sum(A, dim=1).unsqueeze(dim=1) + 1e-5
    A = (A / sum).reshape(-1, N, N)

    return A


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out
# GINConv
class GINGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)
        self.agg = agg()
        print('load GINConv')

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)

        # Step 1: Aggregate neighbors' features
        neighbors_features = torch.bmm(A, features)
        # Step 2: Combine own features and aggregated features
        combined_features = torch.cat([features, neighbors_features], dim=2)
        # Step 3: Apply weight matrix and activation function
        out = torch.matmul(combined_features, self.weight) + self.bias
        out = F.relu(out)
        return out



class triple_model(nn.Module):
    def __init__(self, feature_dim, phase):
        super(triple_model, self).__init__()
        self.dim = feature_dim
        self.nid_2 = feature_dim * 2
        self.phase = phase

        self.bn0 = nn.BatchNorm1d(self.dim)
        self.bn1 = nn.BatchNorm1d(self.dim * 2)

        self.conv1 = GraphConv(self.dim * 2, self.dim * 2, MeanAggregator)
        self.conv2 = GraphConv(self.dim * 2, self.dim * 2, MeanAggregator)

        self.local_conv1 = GraphConv(self.dim * 2, self.dim * 2, MeanAggregator)

        self.local_conv2 = GraphConv(self.dim * 2, self.dim * 2, MeanAggregator)

        self.trans1 = nn.Sequential(nn.Linear(self.dim, self.dim * 2),
                                   nn.GELU(),
                                   nn.Linear(self.dim * 2, self.dim * 2))

        self.trans2 = nn.Sequential(nn.Linear(self.dim * 2, self.dim * 2),
                                   nn.GELU(),
                                   nn.Linear(self.dim * 2, self.dim * 2))

        self.trans_x = nn.Linear(self.dim, self.dim * 2)

        self.Up_dim = nn.Linear(self.dim, self.dim * 2)

        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim * 2))

        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim * 2))

        self.classifier_out = nn.Linear(self.dim * 6, self.dim * 2)

        self.local_classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim * 2))
        self.local_classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim * 2))

        self.trans1.apply(self._initialize_weights)
        self.trans2.apply(self._initialize_weights)
        self.trans_x.apply(self._initialize_weights)
        self.classifier1.apply(self._initialize_weights)
        self.classifier2.apply(self._initialize_weights)
        self.classifier_out.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feature, center_idx, mask, K, gt_A, adj_local, nodes_label=[],t = 1):
        B, N, D = feature.shape
        # x = self.Ln0(feature)
        x = feature.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)
        x_res = self.Up_dim(x)

        x_trs = self.trans1(x)
        x_out = x_trs
        adj_local = A_Norm(adj_local)
        x_trs = self.local_conv1(x_trs, adj_local)
        x_trs = x_trs.view(-1, 2 * D)
        x_trs = self.local_classifier1(x_trs)
        x_trs = x_trs.view(B, N, 2 * D)
        x_trs = feature_Norm(x_trs)
        x_trs_cp = x_trs
        x_trs_cp = rearrange(x_trs_cp, 'B N D -> B D N')
        A = torch.bmm(x_trs, x_trs_cp).to(torch.double)
        A = torch.mul(A, mask).to(torch.double)
        A = torch.where(A >= 0.5, A, 0.0).to(torch.float)
        A = A_Norm(A)

        x = self.trans_x(x)

        x = self.conv1(x, A)
        x = x.view(-1, 2 * D)
        x = self.classifier1(x)
        x = x.view(B, N, 2 * D)
        x = x + x_res
        x_stage1 = x

        x_res = x
        x_trs = self.trans2(x)
        adj_local = A_Norm(adj_local)
        x_trs = self.local_conv2(x_trs, adj_local)
        x_trs = x_trs.view(-1, 2 * D)
        x_trs = self.local_classifier2(x_trs)
        x_trs = x_trs.view(B, N, 2 * D)
        x_trs = feature_Norm(x_trs)
        x_trs_cp = x_trs
        x_trs_cp = rearrange(x_trs_cp, 'B N D -> B D N')
        A = torch.bmm(x_trs, x_trs_cp).to(torch.double)
        A = torch.mul(A, mask).to(torch.double)
        A = torch.where(A >= 0.5, A, 0.0).to(torch.float)
        A = A_Norm(A)

        x = self.conv2(x, A)
        x = x.view(-1, 2 * D)
        x = self.classifier2(x)
        x = x.view(B, N, 2 * D)
        x = x + x_res

        x = torch.cat((x,x_stage1,x_out),dim=-1)
        x = self.classifier_out(x)

        new_D = x.shape[2]

        if self.phase == 'train':
            x = x.reshape(-1, new_D)
            x = l2norm(x)
            x = x.reshape(B, N, new_D)

            x_new_trs = x
            x_new_trs = rearrange(x_new_trs, 'B N D -> B D N')
            new_sim = torch.bmm(x, x_new_trs)
            same_sim = torch.mul(new_sim,gt_A).to(torch.double)
            same_sim = torch.where((same_sim == 0),1.0,same_sim)
            diff_sim = torch.mul(new_sim,1 - gt_A)

            same_id = torch.argmin(same_sim.reshape(-1, N), dim=1)
            diff_id = torch.argmax(diff_sim.reshape(-1, N), dim=1)
            for b in range(B):
                same_id[b * N:(b + 1) * N] = same_id[b * N:(b + 1) * N] + b * N
                diff_id[b * N:(b + 1) * N] = diff_id[b * N:(b + 1) * N] + b * N

            x = x.reshape(-1,new_D)
            pos = x[same_id]
            neg = x[diff_id]
            anchor = x

            return anchor, pos, neg
        else:
            x = x.reshape(-1, new_D)
            x = l2norm(x)
            x = x.reshape(B, N, new_D)

            x_new_trs = x
            x_new_trs = rearrange(x_new_trs, 'B N D -> B D N')
            new_sim = torch.bmm(x, x_new_trs)

            pos_id = np.zeros(shape=(B))
            neg_id = np.zeros(shape=(B))
            for b in range(B):
                center_sim = new_sim[b,center_idx[b]]
                batch_id = torch.argsort(center_sim, descending=True)
                batch_label = nodes_label[b, batch_id]
                batch_pos_id = torch.where(batch_label == True)[0]
                batch_neg_id = torch.where(batch_label == False)[0]
                pos_id[b] = batch_id[batch_pos_id[-1].item()] + (b * (K[0] * K[1] + 1))
                neg_id[b] = batch_id[batch_neg_id[0].item()] + (b * (K[0] * K[1] + 1))
                center_idx[b] = center_idx[b] + b * N

            x = x.reshape(-1, new_D)
            pos = x[pos_id]
            neg = x[neg_id]
            anchor = x[center_idx]

            return anchor, pos, neg


