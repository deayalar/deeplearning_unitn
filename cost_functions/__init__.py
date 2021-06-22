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


class OverallLoss(nn.Module):
    def __init__(self):
        id_loss = IdentificationLoss()
        attr_loss = AttributesLossWrapper(0)

    def forward(self, outputs_attr, attr, ids):
        return self.id_loss(outputs_attr, ids) + self.attr_loss(outputs_attr, attr)

class IdentificationLoss(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, preds, ids):
        return 0

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
