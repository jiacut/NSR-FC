import torch
import torch.optim as optim
import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import Timer
from Dataset import build_dataset

from tqdm import tqdm
def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while (u != parent[u]):
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u

def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        v = int(v)
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id

def add_edge(pred, bid, cid):
    edge = []
    for idx, (bid_, cid_) in enumerate(zip(bid, cid)):
        if pred[idx] == 1:
            edge.append([bid_, cid_])
    return edge

def l2norm(vec):
    vec_temp = torch.norm(vec,p=2,dim=1).reshape(-1, 1)
    return vec / vec_temp


class MeanLoss(nn.Module):
    def __init__(self, an, ap):
        super(MeanLoss, self).__init__()
        self.an = an
        self.ap = ap

    def forward(self, anchor, pos, neg):
        an_gap = ((torch.sum(torch.mul(anchor, neg), dim=1)) - self.an).double()
        an_los = torch.where(an_gap >= 0.0, an_gap, 0.0)

        ap_gap = (self.ap - torch.sum(torch.mul(anchor, pos), dim=1)).double()
        ap_los = torch.where(ap_gap >= 0.0, ap_gap, 0.0)

        return torch.mean(an_los) + torch.mean(ap_los)

def test_triple(model, cfg):
    with Timer('1.prepare Dataset'):
        dataset = build_dataset(cfg.model['type'], cfg)

        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=True,
                                 shuffle=True)
        a = 0.9

        triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=2*a-1).cuda()

        Mean_loss = MeanLoss(1-a,a).cuda()

        model = model.cuda()

        model.load_state_dict(torch.load(cfg.save_path))
        print("=> Loaded checkpoint model '{}'".format(cfg.save_path))
        model.eval()

        final_feature = np.zeros(shape=(dataset.feature.shape[0],dataset.feature.shape[1]*2))
        bar = tqdm(data_loader)
        for (feature, center_idx, mask, idx, nodes_label, adj, adj_local) in bar:
            feature, center_idx, mask, idx, nodes_label, adj, adj_local = map(lambda x: x.cuda(), (feature, center_idx, mask, idx, nodes_label, adj, adj_local))

            anchor, pos, neg = model(feature, center_idx, mask, dataset.K, adj, adj_local, nodes_label)

            loss1 = triplet_loss(anchor, pos, neg)
            loss2 = Mean_loss(anchor, pos, neg)

            loss_sum = loss1 + loss2

            idx = idx.detach().cpu().numpy()

            anchor = anchor.detach().cpu().numpy()

            final_feature[idx] = anchor

            # print('loss_sum:{:.8f},loss1:{:.8f},loss2:{:.8f},{}/{}'.format(loss_sum.item(), loss1.item(), loss2.item(), i, int(dataset.inst_num / cfg.batch_size)))
            bar.set_description('loss_sum:{:.8f},loss1:{:.8f},loss2:{:.8f}'.format(loss_sum.item(), loss1.item(), loss2.item()))

        np.save(f"./data/Out_feature/{cfg.test_name}.npy", final_feature)

