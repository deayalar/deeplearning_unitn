import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class OverallLossWrapper(nn.Module):
    def __init__(self,num_classes,feat_dim):
        super(OverallLossWrapper, self).__init__()
        # self.id_loss = TripletLoss()
        self.id_loss = CenterLoss(num_classes,feat_dim)
        self.attr_loss = AttributesLossWrapper(0)

    def forward(self, output_attrs, target_attrs, output_features, target_ids):
        return self.id_loss(output_features, target_ids) + self.attr_loss(output_attrs, target_attrs)

class AttributesLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(AttributesLossWrapper, self).__init__()
        self.task_num = task_num
        # This is to learn the weights
        #self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, attrs):

        bce = nn.BCELoss()
        crossEntropy = nn.CrossEntropyLoss()

        loss_age = crossEntropy(preds[0], attrs[:,0])

        binary_losses = 0
        for idx in range(1, len(preds)):
            binary_losses += bce(preds[idx], attrs[:, idx].unsqueeze(1).to(torch.float32))
        return loss_age + binary_losses

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        #print(targets)
        #print(inputs.size())
        targets = torch.Tensor(np.array([int(el) for el in targets]))
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        #print("dist_an", dist_an)
        #print("dist_ap", dist_ap)
        #print(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
    
class CenterLoss(nn.Module):
    """Center loss.
    code imported: https://github.com/KaiyangZhou/pytorch-center-loss
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, output_features, target_ids):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            self, x, labels -> self, output_features, target_ids
        """
        batch_size = output_features.size(0)
        distmat = torch.pow(output_features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, output_features, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        target_ids = target_ids.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = target_ids.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

