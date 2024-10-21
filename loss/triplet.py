import torch
import torch.nn as nn

def l2norm(vec):
    vec_temp = torch.norm(vec,p=2,dim=1).reshape(-1, 1)
    return vec / vec_temp

class MeanLoss(nn.Module):
    def __init__(self, an, ap):
        super(MeanLoss, self).__init__()
        self.an = an
        self.ap = ap

    def forward(self, anchor, pos, neg):
        anchor = l2norm(anchor)
        pos = l2norm(pos)
        neg = l2norm(neg)
        an_gap = ((torch.sum(torch.mul(anchor, neg), dim=1)) - self.an).double()
        an_los = torch.where(an_gap >= 0.0,an_gap, 0.0)
        ap_gap = (self.ap - torch.sum(torch.mul(anchor, pos), dim=1)).double()
        ap_los = torch.where(ap_gap >= 0.0, ap_gap, 0.0)
        return torch.mean(an_los) + torch.mean(ap_los)

class TripletMarginWithDistanceLoss(nn.Module):
    def __init__(self, *, distance_function, margin: float = 1.0, swap: bool = False):
        super(TripletMarginWithDistanceLoss, self).__init__()
        self.distance_function = distance_function
        self.margin = margin
        self.swap = swap

    def forward(self, anchor, positive, negative):
        positive_dist = self.distance_function(anchor, positive)
        negative_dist = self.distance_function(anchor, negative)

        output = torch.clamp(positive_dist - negative_dist + self.margin, min=0.0)
        return output.mean()
