import torch
import torch.nn as nn
import torch.nn.functional as F

class OverallLossWrapper(nn.Module):
    def __init__(self):
        super(OverallLossWrapper, self).__init__()
        self.id_loss = TripletLoss()
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

        #loss_backpack = bce(preds[1], attrs[:,1].unsqueeze(1).to(torch.float32))
        #loss_bag = bce(preds[2], attrs[:,2].unsqueeze(1).to(torch.float32))
        #loss_handbag = bce(preds[3], attrs[:,3].unsqueeze(1).to(torch.float32))
        #loss_clothes = bce(preds[4], attrs[:,4].unsqueeze(1).to(torch.float32))
        #loss_down = bce(preds[5], attrs[:,5].unsqueeze(1).to(torch.float32))
        #loss_up = bce(preds[6], attrs[:,6].unsqueeze(1).to(torch.float32))
        #loss_hair = bce(preds[7], attrs[:,7].unsqueeze(1).to(torch.float32))
        #loss_hat = bce(preds[8], attrs[:,8].unsqueeze(1).to(torch.float32))
        #loss_gender = bce(preds[9], attrs[:,9].unsqueeze(1).to(torch.float32))
        #loss_upblack = bce(preds[10], attrs[:,10].unsqueeze(1).to(torch.float32))
        #loss_upwhite = bce(preds[11], attrs[:,11].unsqueeze(1).to(torch.float32))
        #loss_upred = bce(preds[12], attrs[:,12].unsqueeze(1).to(torch.float32))
        #loss_uppurple = bce(preds[13], attrs[:,13].unsqueeze(1).to(torch.float32))
        #loss_upyellow = bce(preds[14], attrs[:,14].unsqueeze(1).to(torch.float32))
        #loss_upgray = bce(preds[15], attrs[:,15].unsqueeze(1).to(torch.float32))
        #loss_upblue = bce(preds[16], attrs[:,16].unsqueeze(1).to(torch.float32))
        #loss_upgreen = bce(preds[17], attrs[:,17].unsqueeze(1).to(torch.float32))
        #loss_downblack = bce(preds[18], attrs[:,18].unsqueeze(1).to(torch.float32))
        #loss_downwhite = bce(preds[19], attrs[:,19].unsqueeze(1).to(torch.float32))
        #loss_downpink = bce(preds[20], attrs[:,20].unsqueeze(1).to(torch.float32))
        #loss_downpurple = bce(preds[21], attrs[:,21].unsqueeze(1).to(torch.float32))
        #loss_downyellow = bce(preds[22], attrs[:,22].unsqueeze(1).to(torch.float32))
        #loss_downgray = bce(preds[23], attrs[:,23].unsqueeze(1).to(torch.float32))
        #loss_downblue = bce(preds[24], attrs[:,24].unsqueeze(1).to(torch.float32))
        #loss_downgreen = bce(preds[25], attrs[:,25].unsqueeze(1).to(torch.float32))
        #loss_downbrown = bce(preds[26], attrs[:,26].unsqueeze(1).to(torch.float32))

        binary_losses = 0
        for idx in range(1, len(preds)):
            binary_losses += bce(preds[idx], attrs[:, idx].unsqueeze(1).to(torch.float32))

        #precision0 = torch.exp(-self.log_vars[0])
        #loss0 = precision0*loss0 + self.log_vars[0]

        # precision1 = torch.exp(-self.log_vars[1])
        # loss1 = precision1*loss1 + self.log_vars[1]

        # precision2 = torch.exp(-self.log_vars[2])
        # loss2 = precision2*loss2 + self.log_vars[2]
        
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
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
