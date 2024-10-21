#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import faiss
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch

from utils import (load_data, dump_data, mkdir_if_no_exists, Timer)
from utils import faiss_search_knn

__all__ = [
    'knn_brute_force', 'knn_hnsw', 'knn_faiss', 'knn_faiss_gpu', 'knns2spmat',
    'fast_knns2spmat', 'knns2sub_spmat', 'build_knns', 'filter_knns',
    'knns2ordered_nbrs'
]


def knns_recall(nbrs, idx2lb, lb2idxs):
    with Timer('compute recall'):
        recs = []
        cnt = 0
        for idx, (n, _) in enumerate(nbrs):
            lb = idx2lb[idx]
            idxs = lb2idxs[lb]
            n = list(n)
            if len(n) == 1:
                cnt += 1
            s = set(idxs) & set(n)
            recs += [1. * len(s) / len(idxs)]
        print('there are {} / {} = {:.3f} isolated anchors.'.format(
            cnt, len(nbrs), 1. * cnt / len(nbrs)))
    recall = np.mean(recs)
    return recall


def filter_knns(knns, k, th):
    pairs = []
    scores = []
    n = len(knns)
    nbrs = np.zeros([n, k], dtype=np.int32) - 1
    simi = np.zeros([n, k]) - 1
    for i, (nbr, dist) in enumerate(knns):
        assert len(nbr) == len(dist)
        nbrs[i, :len(nbr)] = nbr
        simi[i, :len(nbr)] = 1. - dist
    anchor = np.tile(np.arange(n).reshape(n, 1), (1, k))

    # filter
    selidx = np.where((simi >= th) & (nbrs != -1) & (nbrs != anchor))
    pairs = np.hstack((anchor[selidx].reshape(-1,
                                              1), nbrs[selidx].reshape(-1, 1)))
    scores = simi[selidx]

    if len(pairs) > 0:
        # keep uniq pairs
        pairs = np.sort(pairs, axis=1)
        pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
        scores = scores[unique_idx]
    return pairs, scores


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


def knns2spmat(knns, k, th_sim=0.7, use_sim=False):
    # convert knns to symmetric sparse matrix
    from scipy.sparse import csr_matrix
    eps = 1e-5
    n = len(knns)
    row, col, data = [], [], []
    for row_i, knn in enumerate(knns):
        nbrs, dists = knn
        for nbr, dist in zip(nbrs, dists):
            assert -eps <= dist <= 1 + eps, "{}: {}".format(row_i, dist)
            w = dist
            if 1 - w < th_sim or nbr == -1:
                continue
            if row_i == nbr:
                assert abs(dist) < eps
                continue
            row.append(row_i)
            col.append(nbr)
            if use_sim:
                w = 1 - w
            data.append(w)
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def fast_knns2spmat(knns, k, th_sim=0.7, use_sim=False, fill_value=None):
    # convert knns to symmetric sparse matrix
    from scipy.sparse import csr_matrix
    eps = 1e-5
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    if len(knns.shape) == 2:
        # knns saved by hnsw has different shape
        n = len(knns)
        ndarr = np.ones([n, 2, k])
        ndarr[:, 0, :] = -1  # assign unknown dist to 1 and nbr to -1
        for i, (nbr, dist) in enumerate(knns):
            size = len(nbr)
            assert size == len(dist)
            ndarr[i, 0, :size] = nbr[:size]
            ndarr[i, 1, :size] = dist[:size]
        knns = ndarr
    nbrs = knns[:, 0, :]
    dists = knns[:, 1, :]
    assert -eps <= dists.min() <= dists.max(
    ) <= 1 + eps, "min: {}, max: {}".format(dists.min(), dists.max())
    if use_sim:
        sims = 1. - dists
    else:
        sims = dists
    if fill_value is not None:
        print('[fast_knns2spmat] edge fill value:', fill_value)
        sims.fill(fill_value)
    row, col = np.where(sims >= th_sim)
    # remove the self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def knns2sub_spmat(idxs, knns, th_sim=0.7, use_sim=False):
    # convert knns to symmetric sparse sub-matrix
    from scipy.sparse import csr_matrix
    n = len(idxs)
    row, col, data = [], [], []
    abs2rel = {}
    for rel_i, abs_i in enumerate(idxs):
        assert abs_i not in abs2rel
        abs2rel[abs_i] = rel_i

    for row_i, idx in enumerate(idxs):
        nbrs, dists = knns[idx]
        for nbr, dist in zip(nbrs, dists):
            if idx == nbr:
                assert abs(dist) < 1e-6, "{}: {}".format(idx, dist)
                continue
            if nbr not in abs2rel:
                continue
            col_i = abs2rel[nbr]
            assert -1e-6 <= dist <= 1
            w = dist
            if 1 - w < th_sim or nbr == -1:
                continue
            row.append(row_i)
            col.append(col_i)
            if use_sim:
                w = 1 - w
            data.append(w)
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def build_knns(knn_prefix,
               feats,
               knn_method,
               k,
               num_process=None,
               is_rebuild=False,
               feat_create_time=None):
    knn_prefix = os.path.join(knn_prefix, '{}_k_{}'.format(knn_method, k))
    mkdir_if_no_exists(knn_prefix)
    knn_path = knn_prefix + '.npz'
    if os.path.isfile(
            knn_path) and not is_rebuild and feat_create_time is not None:
        knn_create_time = os.path.getmtime(knn_path)
        if knn_create_time <= feat_create_time:
            print('[warn] knn is created before feats ({} vs {})'.format(
                format_time(knn_create_time), format_time(feat_create_time)))
            is_rebuild = True
    if not os.path.isfile(knn_path) or is_rebuild:
        index_path = knn_prefix + '.index'
        with Timer('build index'):
            if knn_method == 'hnsw':
                index = knn_hnsw(feats, k, index_path)
            elif knn_method == 'faiss':
                index = knn_faiss(feats,
                                  k,
                                  index_path,
                                  omp_num_threads=num_process,
                                  rebuild_index=True)
            elif knn_method == 'faiss_gpu':
                index = knn_faiss_gpu(feats,
                                      k,
                                      index_path,
                                      num_process=num_process)
            else:
                raise KeyError(
                    'Only support hnsw and faiss currently ({}).'.format(
                        knn_method))
            knns = index.get_knns()
        with Timer('dump knns to {}'.format(knn_path)):
            dump_data(knn_path, knns, force=True)
    else:
        print('read knn from {}'.format(knn_path))
        knns = load_data(knn_path)
    return knns


class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return (th_nbrs, th_dists)

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_brute_force(knn):
    def __init__(self, feats, k, index_path='', verbose=True):
        self.verbose = verbose
        with Timer('[brute force] build index', verbose):
            feats = feats.astype('float32')
            sim = feats.dot(feats.T)
        with Timer('[brute force] query topk {}'.format(k), verbose):
            nbrs = np.argpartition(-sim, kth=k)[:, :k]
            idxs = np.array([i for i in range(nbrs.shape[0])])
            dists = 1 - sim[idxs.reshape(-1, 1), nbrs]
            self.knns = [(np.array(nbr, dtype=np.int32),
                          np.array(dist, dtype=np.float32))
                         for nbr, dist in zip(nbrs, dists)]


class knn_hnsw(knn):
    def __init__(self, feats, k, index_path='', verbose=True, **kwargs):
        import nmslib
        self.verbose = verbose
        with Timer('[hnsw] build index', verbose):
            ''' higher ef leads to better accuracy, but slower search
                higher M leads to higher accuracy/run_time at fixed ef,
                but consumes more memory
            '''
            # space_params = {
            #     'ef': 100,
            #     'M': 16,
            # }
            # index = nmslib.init(method='hnsw',
            #                     space='cosinesimil',
            #                     space_params=space_params)
            index = nmslib.init(method='hnsw', space='cosinesimil')
            if index_path != '' and os.path.isfile(index_path):
                index.loadIndex(index_path)
            else:
                index.addDataPointBatch(feats)
                index.createIndex({
                    'post': 2,
                    'indexThreadQty': 1
                },
                                  print_progress=verbose)
                if index_path:
                    print('[hnsw] save index to {}'.format(index_path))
                    mkdir_if_no_exists(index_path)
                    index.saveIndex(index_path)
        with Timer('[hnsw] query topk {}'.format(k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                print('[hnsw] read knns from {}'.format(knn_ofn))
                self.knns = np.load(knn_ofn)['data']
            else:
                self.knns = index.knnQueryBatch(feats, k=k)


class knn_faiss(knn):
    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 index_key='',
                 nprobe=128,
                 omp_num_threads=None,
                 rebuild_index=True,
                 verbose=True,
                 **kwargs):
        import faiss
        if omp_num_threads is not None:
            faiss.omp_set_num_threads(omp_num_threads)
        self.verbose = verbose
        with Timer('[faiss] build index', verbose):
            if index_path != '' and not rebuild_index and os.path.exists(
                    index_path):
                print('[faiss] read index from {}'.format(index_path))
                index = faiss.read_index(index_path)
            else:
                feats = feats.astype('float32')
                size, dim = feats.shape
                index = faiss.IndexFlatIP(dim)
                if index_key != '':
                    assert index_key.find(
                        'HNSW') < 0, 'HNSW returns distances insted of sims'
                    metric = faiss.METRIC_INNER_PRODUCT
                    nlist = min(4096, 8 * round(math.sqrt(size)))
                    if index_key == 'IVF':
                        quantizer = index
                        index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                                   metric)
                    else:
                        index = faiss.index_factory(dim, index_key, metric)
                    if index_key.find('Flat') < 0:
                        assert not index.is_trained
                    index.train(feats)
                    index.nprobe = min(nprobe, nlist)
                    assert index.is_trained
                    print('nlist: {}, nprobe: {}'.format(nlist, nprobe))
                index.add(feats)
                if index_path != '':
                    print('[faiss] save index to {}'.format(index_path))
                    mkdir_if_no_exists(index_path)
                    faiss.write_index(index, index_path)
        with Timer('[faiss] query topk {}'.format(k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                print('[faiss] read knns from {}'.format(knn_ofn))
                self.knns = np.load(knn_ofn)['data']
            else:
                sims, nbrs = index.search(feats, k=k)
                self.knns = [(np.array(nbr, dtype=np.int32),
                              1 - np.array(sim, dtype=np.float32))
                             for nbr, sim in zip(nbrs, sims)]


class knn_faiss_gpu(knn):
    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 index_key='',
                 nprobe=128,
                 num_process=4,
                 is_precise=False,
                 sort=True,
                 verbose=True,
                 **kwargs):
        with Timer('[faiss_gpu] query topk {}'.format(k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                print('[faiss_gpu] read knns from {}'.format(knn_ofn))
                self.knns = np.load(knn_ofn)['data']
            else:
                dists, nbrs = faiss_search_knn(feats,
                                               k=k,
                                               nprobe=nprobe,
                                               num_process=num_process,
                                               is_precise=is_precise,
                                               sort=sort,
                                               verbose=False)

                self.knns = [(np.array(nbr, dtype=np.int32),
                              np.array(dist, dtype=np.float32))
                             for nbr, dist in zip(nbrs, dists)]

################################

def build_knn(feats, k):
    # faiss.omp_set_num_threads(threads)

    feats = feats.astype('float32')
    size, dim = feats.shape
    index = faiss.IndexFlatIP(dim)
    index.add(feats)
    sims, nbrs = index.search(feats, k=k)
    knns = [(np.array(nbr, dtype=np.int32),
             1 - np.minimum(np.maximum(np.array(sim, dtype=np.float32), 0), 1))
            for nbr, sim in zip(nbrs, sims)]
    return knns


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # if rowsum <= 0, keep its previous value
    rowsum[rowsum <= 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_indices_values(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    values = sparse_mx.data
    shape = np.array(sparse_mx.shape)
    return indices, values, shape



def label2spmat(label, load=True):
    from scipy.sparse import csr_matrix
    n = len(label)

    row = np.array([])
    col = np.array([])
    if load != True:
        row_temp = np.array([])
        col_temp = np.array([])
        for idx in tqdm(range(n),desc="label2spmat"):
            row_nodeid = np.where(label[idx] == label)[0]
            row_temp = np.concatenate((row_temp, row_nodeid), axis=-1)

            col_nodeid = np.full((len(row_nodeid)), idx)
            col_temp = np.concatenate((col_temp, col_nodeid), axis=-1)
            if idx % 10000 == 0 or idx + 1 == n:
                row = np.concatenate((row, row_temp), axis=-1)
                col = np.concatenate((col, col_temp), axis=-1)
                row_temp = np.array([])
                col_temp = np.array([])
                # print(idx)

        np.save(f"data/row_{n}.npy", row)
        np.save(f"data/col_{n}.npy", col)
        print("Save Finish")
    else:
        row = np.load(f"data/row_{n}.npy")
        col = np.load(f"data/col_{n}.npy")

    data = np.ones(shape=(row.shape))
    spmat = csr_matrix((data, (row, col)), shape=(n, n))

    return spmat



def knn2spmat(dists, nbrs, k, th_sim, use_sim, self_loop):
    eps = 1e-2
    n = len(nbrs)
    if use_sim:
        sims = 1. - dists
    row, col = np.where(sims >= th_sim)
    # remove self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    # make symmetric adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    # div row sum to norm
    adj = row_normalize(adj)
    # sparse_mx2torch_sparse
    indices, values, shape = sparse_mx_to_indices_values(adj)
    indices = torch.from_numpy(indices)
    values = torch.from_numpy(values)
    shape = torch.Size(shape)

    return torch.sparse.FloatTensor(indices, values, shape)



def fast_knns2spmat(knns, k, th_sim=0.7, active_connection=10, use_sim=False, fill_value=None):
    # convert knns to symmetric sparse matrix
    from scipy.sparse import csr_matrix
    eps = 1e-5
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    if len(knns.shape) == 2:
        # knns saved by hnsw has different shape
        n = len(knns)
        ndarr = np.ones([n, 2, k])
        ndarr[:, 0, :] = -1  # assign unknown dist to 1 and nbr to -1
        for i, (nbr, dist) in enumerate(knns):
            size = len(nbr)
            assert size == len(dist)
            ndarr[i, 0, :size] = nbr[:size]
            ndarr[i, 1, :size] = dist[:size]
        knns = ndarr
    nbrs = knns[:, 0, :active_connection]
    dists = knns[:, 1, :active_connection]
    assert -eps <= dists.min() <= dists.max(
    ) <= 1 + eps, "min: {}, max: {}".format(dists.min(), dists.max())
    if use_sim:
        sims = 1. - dists
    else:
        sims = dists
    if fill_value is not None:
        print('[fast_knns2spmat] edge fill value:', fill_value)
        sims.fill(fill_value)
    row, col = np.where(sims >= th_sim)
    # remove the self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def knn2mat(knns, k, use_sim, self_loop):
    eps = 1e-2
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :]
    dists = knns[:, 1, :]
    if use_sim:
        sims = 1. - dists
    row, col = np.where(sims >= 0)
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    # make symmetric adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    # div row sum to norm
    adj = row_normalize(adj)
    adj = adj.todense().A

    return adj



def build_symmetric_adj(adj, self_loop=True):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj


