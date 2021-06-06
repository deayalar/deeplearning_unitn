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