import os.path as osp

phase = 'train'
train_name = 'part0_train'

# data path
data_path = './data'
# save model path
save_path = './saves'

# global config
knn_method = 'faiss'
seed = 1

batch_size = 64
workers = 10
epochs = 10
lr = 0.01
warmup_step = 5000

# model
model = dict(type='triple',
             kwargs=dict(feature_dim=256, phase=phase))

# training args
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)

STAGES = [int(epochs*0.4), int(epochs*0.6), int(epochs*0.8)]
lr_config = dict(
    policy='step',
    step=[int(r * epochs) for r in [0.4, 0.6, 0.8]]
)

data = dict(feat_path=osp.join(data_path, 'features', '{}.bin'.format(train_name)),
            label_path=osp.join(data_path, 'labels', '{}.meta'.format(train_name)),
            knn_path=osp.join(data_path, 'knns', train_name,
                             '{}_k_{}.npz'.format(knn_method, 80)),
            reknn_path=osp.join(data_path, 'reranking_knn_{}.npy'.format(train_name)),
            resim_path=osp.join(data_path, 'reranking_sim_{}.npy'.format(train_name)),
            feature_dim=256,
            K=[80,10])


