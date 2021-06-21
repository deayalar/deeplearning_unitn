import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy():
    return nn.CrossEntropyLoss()

def bin_cross_entropy():
    return nn.BCELoss()

def bin_cross_entropy_logit():
    return nn.BCEWithLogitsLoss()

def triplet_margin_loss():
    return nn.TripletMarginLoss()

class AttributesLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        #self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, attrs):

        #bce, crossEntropy = nn.BCELoss(), CrossEntropyFlat()
        bce = nn.BCELoss()
        # mock = torch.empty(8, 1).random_(2)

        #print(attrs[:,1].unsqueeze(1).size())
        #print(attrs[:,1].unsqueeze(1))
        # loss_backpack = bce(preds[1], mock.to("cuda:0"))
        loss_backpack = bce(preds[1], attrs[:,1].unsqueeze(1).to(torch.float32))
        loss_bag = bce(preds[0], attrs[:,2].unsqueeze(1).to(torch.float32))

        #precision0 = torch.exp(-self.log_vars[0])
        #loss0 = precision0*loss0 + self.log_vars[0]

        # precision1 = torch.exp(-self.log_vars[1])
        # loss1 = precision1*loss1 + self.log_vars[1]

        # precision2 = torch.exp(-self.log_vars[2])
        # loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss_bag + loss_backpack
