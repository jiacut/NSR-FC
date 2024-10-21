# NSR-FC

This is an official implementation for "NSR-FC": "Enhanced face clustering with neighbor structure refinement"

## Introduction
In NSR-FC, the cascaded graph convolutional networks (C-GCN) enhance featureextraction while optimizing the graph structure to reduce noise and generate discriminative features.We then construct a local density graph using refined distances in the updated k-nearest neighbor(KNN) graph, eliminating negative pairs and preserving positive ones. Additionally, we evaluate edgeconnectivity within the density graph to produce a global connectivity graph. Finally, the breadth first search (BFS) algorithm is utilized to derive the clustering results. Our experimental resultsdemonstrate that NSR-FC achieves state-of-the-art performance on large-scale face image datasetslike MS-Celeb-1M, highlighting its effectiveness in face clustering.
The main framework of PSE-GCN is shown as following:

<img src=assets/framework.png width=900 />

## Main Results

<img src=assets/result.png width=900 />


## Getting Started

### Install

+ Clone this repo

```
git clone https://github.com/jiacut/NSR-FC.git
cd NSR-FC
```

+ Create a conda virtual environment and activate it

```
conda create -n nsrfc python=3.8 -y
conda activate nsrfc
```

+ Install `Pytorch` and other requirements.
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```


### Data preparation

The process of clustering on the MS-Celeb part1 is as follows:

- [MS1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
    - part0_train & part1_test (584K): [GoogleDrive](https://drive.google.com/open?id=16WD4orcF9dqjNPLzST2U3maDh2cpzxAY).
    - part0_train & part1/3/5/7/9_test: [GoogleDrive](https://drive.google.com/file/d/10boLBiYq-6wKC_N_71unlMyNrimRjpVa/view?usp=sharing).
    - Precomputed KNN: [GoogleDrive](https://drive.google.com/file/d/1CRwzy899vkLqIYm60AzDsaDEBuwgxNlY/view?usp=sharing).

The file structure should look like:

```
|——data
   |——features
      |——part0_train.bin
      |——part1_test.bin
      |——...
      |——part9_test.bin
   |——labels
      |——part0_train.meta
      |——part1_test.meta
      |——...
      |——part9_test.meta
   |——knns
      |——part0_train/faiss_k_80.npz
      |——part1_test/faiss_k_80.npz
      |——...
      |——part9_test/faiss_k_80.npz
```

## Training
### Re-ranking knn

```
python re-ranking.py 80, part0_train
```

### Training model
```
python main1.py --config ./config/cfg_train.py
```

### Generate refined feature
You need to change `test_name` in ./config/cfg_test.py from `part0_train` to `part1/3/5/7/9_test` to get the fine feature.
```
mkdir Out_feature
python re-ranking.py 80, $test_name
python main1.py --config ./config/cfg_test.py
```
After running the code, you will get the fine feature in `./Out_feature/`. Put them to the `./data/features/` folder. 

### Re-training model
```
python main.py -c ./config/config_train_ms1m.yaml
```

### Get new knn
``` bash
python eval.py -c ./config/config_eval_ms1m_part*.yaml
```

## Test

Simply run shell scripts.

```
sh inference_ms1m.sh
```