import numpy as np
import torch
import random

from utils import (Timer, build_knn, knn2mat, build_symmetric_adj, row_normalize, fast_knns2spmat
, knns2ordered_nbrs, read_meta, intdict2ndarray, read_probs, label2spmat)
import torch.utils.data as data
import faiss
import scipy.sparse as sp
from einops import rearrange
from tqdm import tqdm


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


class Dataset(data.Dataset):
    def __init__(self, cfg, phase):
        self.feat_path = cfg.data['feat_path']
        self.label_path = cfg.data['label_path']
        self.knn_path = cfg.data['knn_path']
        self.feature_dim = cfg.data['feature_dim']
        self.reknn_path = cfg.data['reknn_path']
        self.resim_path = cfg.data['resim_path']
        self.K = cfg.data['K']
        self.depth = len(self.K)
        self.is_sort_knns = True
        self.phase = phase

        with Timer('load training data'):
            self.lb2idxs, self.idx2lb = read_meta(self.label_path)
            self.inst_num = len(self.idx2lb)
            self.label = intdict2ndarray(self.idx2lb).astype(np.int64)
            self.max_label = max(self.label)
            self.ignore_label = False

            self.feature = read_probs(self.feat_path, self.inst_num, self.feature_dim)

            self.feature = l2norm(self.feature)

            self.cls_num = len(self.lb2idxs)
            knns = np.load(self.knn_path)['data']
            assert self.inst_num == len(knns), "{} vs {}".format(self.inst_num, len(knns))

            self.knn = np.load(self.reknn_path).astype(int)
            self.sim = np.load(self.resim_path)

            self.active_connection = 10

        with Timer('Compute center feature'):
            self.center_fea = np.zeros((self.cls_num, self.feature_dim)).astype('float32')
            for i in range(self.cls_num):
                self.center_fea[i] = np.mean(self.feature[self.lb2idxs[i]], 0)

            index = faiss.IndexFlatIP(self.feature_dim)
            index.add(self.center_fea)
            self.center_sims, self.center_knn = index.search(self.center_fea, k=160)

            # load local knn col.npy and row.npy
            self.adj = label2spmat(self.label, False)

        with Timer('Get local knn feature'):
            adj = fast_knns2spmat(knns,5, 0, self.active_connection, True)
            self.local_adj = build_symmetric_adj(adj, self_loop=True)

    def __len__(self):
        return self.inst_num

    def __getitem__(self, index):
        hops = list()
        hops.append(set(self.knn[index][1:self.K[0]]))

        for d in range(1, self.depth):
            hops.append(set())
            for h in hops[-2]:
                hops[-1].update(set(self.knn[h][1:self.K[d] + 1]))

        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([self.knn[index][0], ])

        # if train add all nodes
        flag = 0
        if self.phase == 'train':
            if len(hops_set) < self.K[0] * self.K[1]:
                for class_id in self.center_knn[self.idx2lb[index],:]:
                    if (len(hops_set.union(self.lb2idxs[class_id])) >= self.K[0] * self.K[1]):
                        for new_node in self.lb2idxs[class_id]:
                            hops_set.update([new_node])
                            if len(hops_set) == self.K[0] * self.K[1]:
                                flag = 1
                                break
                            else:
                                continue
                    else:
                        hops_set.update(self.lb2idxs[class_id])

                    if flag == 1:
                        break
            # Prevent the neg point from being found
            while len(hops_set) < self.K[0] * self.K[1] + 1:
                temp_id = random.randint(0,self.inst_num-1)
                if self.label[index] != self.label[temp_id]:
                    hops_set.update([temp_id])

        max_num_nodes = self.K[0] * self.K[1] + 1
        node_list = list(hops_set)

        node_num = len(node_list)

        mask = torch.zeros(max_num_nodes, max_num_nodes)
        mask[:node_num,:node_num] = 1

        center_idx = node_list.index(index)

        feat = torch.Tensor(self.feature[node_list]).type(torch.float)

        feat = torch.cat([feat, torch.zeros(max_num_nodes - node_num, self.feature_dim)], dim=0)

        adj_total = torch.zeros(max_num_nodes, max_num_nodes)
        adj = self.adj[node_list, :][:, node_list].toarray().astype(np.float32)
        adj_total[:node_num,:node_num] = torch.tensor(adj)

        adj_local = torch.zeros(max_num_nodes, max_num_nodes)
        adj_local_temp = self.local_adj[node_list, :][:, node_list].toarray().astype(np.float32)
        adj_local[:node_num, :node_num] = torch.tensor(adj_local_temp)

        if self.phase == 'train':
            return feat, center_idx, mask, self.label[node_list], adj_total, adj_local
        else:
            nodes_label = torch.tensor(self.label[index] == self.label[node_list])
            nodes_label = torch.cat([nodes_label, torch.zeros(max_num_nodes - node_num)], dim=0)
            return feat, center_idx, mask, index, nodes_label, adj_total, adj_local
