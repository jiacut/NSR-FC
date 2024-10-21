import torch
import torch.optim as optim
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import Timer
from Dataset import build_dataset
import math
from torch.optim.lr_scheduler import CosineAnnealingLR


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

def l2norm(vec):
    vec_temp = torch.norm(vec,p=2,dim=1).reshape(-1, 1)
    return vec / vec_temp

class TripletLoss(nn.Module):
    def __init__(self,distance_function,margin):
        super(TripletLoss, self).__init__()
        self.distance_function = distance_function
        self.margin = margin
    def forward(self, anchor, positive, negative):
        positive_dist = self.distance_function(anchor, positive)
        negative_dist = self.distance_function(anchor, negative)

        output = torch.clamp(positive_dist - negative_dist + self.margin, min=0.0)

        return output.mean()

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

class F_Norm(nn.Module):
    def __init__(self):
        super(F_Norm, self).__init__()
    def forward(self,A,gt):
        loss = torch.mean(torch.norm(A-gt,'fro',dim=[1,2]))
        return loss

def warmup_lr(current_step, optimizer, cfg):
    if current_step <= cfg.warmup_step:
        rate = 1.0 * current_step / cfg.warmup_step
        lr = cfg.lr * rate
    for each in optimizer.param_groups:
        each['lr'] = lr

def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        lr_per_epoch.append(scheduler.get_epoch_values(epoch))
    return lr_per_epoch

def train_triple(model, cfg):
    t = 1.5
    dataset = build_dataset(cfg.model['type'], cfg)

    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=True, shuffle=True)

    OPTIMIZER = optim.SGD([{'params': model.parameters(), 'weight_decay': cfg.optimizer['weight_decay']}],
                          lr=cfg.optimizer['lr'], momentum=cfg.optimizer['momentum'])

    a = 0.9
    model = model.cuda()

    CE_loss = nn.CrossEntropyLoss().cuda()

    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
                                                    margin=2*a-1).cuda()

    Mean_loss = MeanLoss(1-a,a).cuda()
    Mean_loss_A = MeanLoss(0, 1).cuda()

    with Timer('train'):
        for epoch in range(cfg.epochs):
            print("Epoch: " + str(epoch))
            if epoch == cfg.STAGES[0]:
                schedule_lr(OPTIMIZER)
            if epoch == cfg.STAGES[1]:
                schedule_lr(OPTIMIZER)
            if epoch == cfg.STAGES[2]:
                schedule_lr(OPTIMIZER)
            print("Learning rate: " + str(OPTIMIZER.state_dict()['param_groups'][0]['lr']))

            for i, (feature, center_idx, mask, label, adj, adj_local) in enumerate(data_loader):

                feature, center_idx, mask, label, adj, adj_local = map(lambda x: x.cuda(), (feature,
                        center_idx, mask, label, adj, adj_local))

                label = label.view(-1)

                anchor, pos, neg = model(feature, center_idx, mask, dataset.K, adj, adj_local, t)

                loss1 = triplet_loss(anchor, pos, neg)
                loss2 = Mean_loss(anchor, pos, neg)

                loss = loss1 + loss2

                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                print('[{}]loss_sum:{:.8f},loss1:{:.8f},loss2:{:.8f},{}/{}'.format(epoch, loss.item(), loss1.item(), loss2.item(), i,
                                                                         int(dataset.inst_num / cfg.batch_size)))

            torch.save(model.state_dict(), os.path.join(cfg.save_path, "triple_{}.pth".format(epoch)))