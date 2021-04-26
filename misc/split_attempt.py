'''
The idea here is to split the train set into training and validation
All images of the same person should be either in the train or in the
 validation dataset,

TODO: Refactor this and move to a separate class that can be extended
'''

from scipy import stats
import torch
import math
from torch.utils.data import DataLoader
import os
from collections import Counter
import matplotlib.pyplot as plt

train_path = "/media/deayalar/Data/Documents/Unitn/Deep Learning/Assignment/dataset/train"

images_list = os.listdir(train_path)

ids = [name.split("_")[0] for name in images_list] # get the id

count = Counter(ids)
#TODO: when this is in a separate class, change count by unique, counts were only for ploting purposes
val_size = math.ceil(len(count.keys()) * .3)
train_size = math.floor(len(count.keys()) * .7)

train_set, val_set = torch.utils.data.random_split(list(count.keys()), [train_size, val_size], generator=torch.Generator().manual_seed(42))
len(val_set)

val_set_count = {i: count[i] for i in val_set}
train_set_count = {i: count[i] for i in train_set}

plt.hist(val_set_count.values(), bins=30)
plt.hist(train_set_count.values(), bins=30)
plt.hist(count.values(), bins=30)

fig, ax = plt.subplots()
ax.boxplot((list(count.values()), list(train_set_count.values()), list(val_set_count.values())), 
           showmeans=True, meanline=True,
           labels=('total', 'train', 'val'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

stats.describe(list(count.values()), ddof=1, bias=False)
stats.describe(list(train_set_count.values()), ddof=1, bias=False)
stats.describe(list(val_set_count.values()), ddof=1, bias=False)