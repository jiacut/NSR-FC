import os.path as osp

phase = 'test'
train_name = 'part0_train'
test_name = 'part0_train'

# data path
data_path = './data'
# save model path
save_path = './saves/checkpoint.pth'

# global config
knn_method = 'faiss'
seed = 1

batch_size = 64
workers = 20

# model
model = dict(type='triple',
             kwargs=dict(feature_dim=256, phase=phase))

# cal faiss
cal_faiss = False

data = dict(feat_path=osp.join(data_path, 'features', '{}.bin'.format(test_name)),
            label_path=osp.join(data_path, 'labels', '{}.meta'.format(test_name)),
            knn_path=osp.join(data_path, 'knns', test_name,
                            '{}_k_{}.npz'.format(knn_method, 80)),
            reknn_path=osp.join(data_path, 'reranking_knn_{}.npy'.format(test_name)),
            resim_path=osp.join(data_path, 'reranking_sim_{}.npy'.format(test_name)),
            feature_dim=256,
            K=[80,10])
