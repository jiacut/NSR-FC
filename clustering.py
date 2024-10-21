import sys
import os.path as osp
import torch
import faiss
import numpy as np
from sklearn.metrics.cluster import (contingency_matrix,
                                     normalized_mutual_info_score)

def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec

def read_meta(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr

def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs

def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs

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

def pairwise(gt_labels, pred_labels, sparse=True):
    print(gt_labels.shape)
    print(pred_labels.shape)
    n_samples, = gt_labels.shape
    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel()**2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel()**2) - n_samples

    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def _get_lb2idxs(labels):
    lb2idxs = {}
    for idx, lb in enumerate(labels):
        if lb not in lb2idxs:
            lb2idxs[lb] = []
        lb2idxs[lb].append(idx)
    return lb2idxs


def _compute_fscore(pre, rec):
    return 2. * pre * rec / (pre + rec)


def bcubed(gt_labels, pred_labels):
    gt_labels = gt_labels.numpy()
    gt_lb2idxs = _get_lb2idxs(gt_labels)
    pred_lb2idxs = _get_lb2idxs(pred_labels)

    num_lbs = len(gt_lb2idxs)
    pre = np.zeros(num_lbs)
    rec = np.zeros(num_lbs)
    gt_num = np.zeros(num_lbs)

    for i, gt_idxs in enumerate(gt_lb2idxs.values()):
        all_pred_lbs = np.unique(pred_labels[gt_idxs])
        gt_num[i] = len(gt_idxs)
        for pred_lb in all_pred_lbs:
            pred_idxs = pred_lb2idxs[pred_lb]
            n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
            pre[i] += n**2 / len(pred_idxs)
            rec[i] += n**2 / gt_num[i]

    gt_num = gt_num.sum()
    avg_pre = pre.sum() / gt_num
    avg_rec = rec.sum() / gt_num
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def nmi(gt_labels, pred_labels):
    return normalized_mutual_info_score(pred_labels, gt_labels)

def evaluate(gt_labels, pred_labels, metric='pairwise'):
    if metric == 'pairwise':
        return pairwise(gt_labels, pred_labels)
    elif metric == 'bcubed':
        return bcubed(gt_labels, pred_labels)
    elif metric == 'nmi':
        return nmi(gt_labels, pred_labels)

def cal_confidence(sim, knn):
    conf = np.zeros((sim.shape[0],), dtype=np.float32)
    for i, (k_out, s) in enumerate(zip(knn, sim)):
        sum = 0
        for j, k_in in enumerate(k_out):
            sum += s[j]
        conf[i] = sum
        if i % 1000 == 0:
            print(str(i) + " Finish!")
    conf /= np.abs(conf).max()

    return conf


def main(name, feature_pth):
    test_name = name
    data_path = './data'
    label_path = osp.join(data_path, 'labels', '{}.meta'.format(test_name))

    lb2idxs, idx2lb = read_meta(label_path)
    inst_num = len(idx2lb)
    label = intdict2ndarray(idx2lb)
    feature = np.load(feature_pth)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, 512, flat_config)
    feature = feature.astype("float32")
    index.add(feature)
    sim, knn = index.search(feature, k=5)

    tau_0 = 0.75
    edge = []
    conf = cal_confidence(sim, knn)
    for idx in range(inst_num):
        first_nei_id = -1
        # get node confidence
        node_conf = conf[idx]
        # get neighbour id
        neighbour = knn[idx, :].astype(dtype=int)
        # get neighbour confidence
        nei_conf = conf[neighbour]
        # find the first nearest neighbour
        nei_idx = np.where(nei_conf > node_conf)
        if len(nei_idx[0]):
            first_nei_id = neighbour[nei_idx[0][0]]
        else:
            first_nei_id = -1
        if first_nei_id != -1:
            flag = 80
            if sim[idx, nei_idx[0][0]] >= tau_0:
                edge.append([idx, first_nei_id])


        if idx % 10000 == 0:
            print("calculate confidence (" + str(idx) + "/" + str(inst_num) + ")")


    edges = np.array(edge)
    pre_labels = edge_to_connected_graph(edges, inst_num)

    labels = torch.LongTensor(label)
    print("---------------------------------------------------")
    pre, rec, fscore = evaluate(labels, pre_labels, 'pairwise')
    print("P F-score: pre:", pre, " recall:", rec, "F-score:", fscore)
    print("---------------------------------------------------")
    pre, rec, fscore = evaluate(labels, pre_labels, 'bcubed')
    print("Bcubed F-score: pre:", pre, " recall:", rec, "F-score:", fscore)
    print("---------------------------------------------------")
    nmi = evaluate(labels, pre_labels, 'nmi')
    print("NMI:", nmi)
    print("---------------------------------------------------")
    print("nearest")


if __name__ == '__main__':
    test_name, feature_pth = sys.argv[1], sys.argv[2]
    main(test_name, feature_pth)